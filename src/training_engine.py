import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import GradScaler
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from utils import *

def dice_loss_per_sample(pred, target, smooth=1e-6):
    """Per-sample Dice loss so invalid masks can be gated out safely."""
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return 1 - dice

def evaluate(model, loader, loss_fn, device, is_physics=False):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    device_type = str(device).split(':')[0]
    
    with torch.no_grad():
        for batch in loader:
            if is_physics: X, mask, y, valid = batch
            else: X, y = batch
            X, y = X.to(device), y.to(device)
            with torch.amp.autocast(device_type=device_type):
                outputs = model(X)
                loss = loss_fn(outputs, y)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    avg_loss = val_loss / len(loader)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = correct / total
    return avg_loss, weighted_f1, accuracy

def train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, epochs, device, 
                history=None, checkpoint_dir=None, start_epoch=0, config=None, is_physics=False):
    
    if checkpoint_dir is None: checkpoint_dir = Path.cwd()
    if history is None: history = {}
    
    keys = ['train_loss', 'train_f1', 'train_acc', 'val_loss', 'val_f1', 'val_acc', 'lr', 'physics_loss']
    for k in keys:
        if k not in history: history[k] = []
    if 'best_val_f1' not in history: history['best_val_f1'] = 0.0
    if 'best_val_loss' not in history: history['best_val_loss'] = float('inf')
    if 'patience_counter' not in history: history['patience_counter'] = 0
    
    device_type = str(device).split(':')[0]
    scaler = GradScaler(device=device_type, enabled=device_type == "cuda")
    best_val_f1 = history['best_val_f1']
    final_physics_lambda = config.get('model_hp', {}).get('physics_lambda', 0.25)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss, running_phys = 0.0, 0.0
        train_preds, train_labels = [], []
        train_correct, train_total = 0, 0
        
        current_lambda = final_physics_lambda * min(1.0, (epoch + 1) / 5.0) if is_physics else 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(loop, start=1):
            if is_physics:
                X, target_mask, y, is_valid_mask = batch
                target_mask = target_mask.to(device)
                is_valid_mask = is_valid_mask.to(device)
            else:
                X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device_type):
                if is_physics:
                    outputs, attn_map = model(X, return_attention=True)
                    class_loss = loss_fn(outputs, y)
                else:
                    outputs = model(X)
                    class_loss = loss_fn(outputs, y)
            
            if is_physics:
                
                attn_map_f32 = attn_map.float()
                target_mask_f32 = F.interpolate(
                    target_mask.float(),
                    size=attn_map.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
                p_loss_raw = dice_loss_per_sample(attn_map_f32, target_mask_f32)
                
                probs = F.softmax(outputs, dim=1)
                confidences = probs.gather(1, y.view(-1, 1)).squeeze()
                
                combined_weight = confidences.detach() * is_valid_mask.float()
                batch_p_loss = (p_loss_raw * combined_weight).sum() / combined_weight.sum().clamp_min(1e-8)
                
                loss = class_loss.float() + (current_lambda * batch_p_loss)
                running_phys += batch_p_loss.item()
            else:
                loss = class_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('model_hp', {}).get('clip_grad_norm', 1.0))
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y.cpu().numpy())
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            
            loop.set_postfix(
                loss=loss.item(),
                f1=f1_score(train_labels, train_preds, average='weighted', zero_division=0),
                phys=running_phys / batch_idx if is_physics else 0.0,
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
        epoch_train_acc = train_correct / train_total
        epoch_phys = running_phys / len(train_loader) if is_physics else 0.0
        val_loss, epoch_val_f1, epoch_val_acc = evaluate(model, val_loader, loss_fn, device, is_physics=is_physics)
        
        if isinstance(scheduler, CosineAnnealingLR): scheduler.step()
        else: scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(epoch_loss)
        history['train_f1'].append(epoch_train_f1)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(epoch_val_f1)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)
        history['physics_loss'].append(epoch_phys)
        
        print(
            f"Summary -> Train Loss: {epoch_loss:.4f} | Train F1: {epoch_train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val F1: {epoch_val_f1:.4f} | Physics: {epoch_phys:.4f} | "
            f"Physics Lambda: {current_lambda:.3f} | LR: {current_lr:.2e}"
        )

        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), str(checkpoint_dir / "best_model.pth"))
            print("--- New Best Model Saved ---")
        history['best_val_f1'] = best_val_f1

        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['patience_counter'] = 0
        else:
            history['patience_counter'] += 1

        checkpoint_payload = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'history': history,
            'best_f1': best_val_f1,
            'best_val_loss': history['best_val_loss'],
        }
        torch.save(checkpoint_payload, str(checkpoint_dir / "latest_checkpoint.pkl"))
        save_pickle(history, checkpoint_dir / "training_history.pkl")
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            torch.save(checkpoint_payload, str(checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"))

        if history['patience_counter'] >= config.get('model_hp', {}).get('early_stopping_patience', 25):
            print("Early stopping triggered.")
            break
        
    return history, model
