import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import GradScaler
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import f1_score
from utils import *

def dice_loss(pred, target, smooth=1e-6):
    """Numerically stable Dice loss for spatial alignment."""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
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
    
    scaler = GradScaler()
    best_val_f1 = history['best_val_f1']
    device_type = str(device).split(':')[0]
    final_physics_lambda = config.get('model_hp', {}).get('physics_lambda', 0.25)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss, running_phys = 0.0, 0.0
        train_preds, train_labels = [], []
        train_total = 0
        
        current_lambda = final_physics_lambda * min(1.0, (epoch + 1) / 5.0) if is_physics else 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in loop:
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
                target_mask_f32 = F.interpolate(target_mask.float(), size=attn_map.shape[-1], mode='bilinear')
                p_loss_raw = dice_loss(attn_map_f32, target_mask_f32)
                
                y
                probs = F.softmax(outputs, dim=1)
                confidences = probs.gather(1, y.view(-1, 1)).squeeze()
                
                combined_weight = confidences.detach() * is_valid_mask.float()
                batch_p_loss = (p_loss_raw * combined_weight).mean()
                
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
            train_total += y.size(0)
            
            loop.set_postfix(loss=loss.item(), f1=f1_score(train_labels, train_preds, average='weighted', zero_division=0), phys=running_phys/(len(train_preds)/X.size(0)))

        epoch_loss = running_loss / len(train_loader)
        epoch_phys = running_phys / len(train_loader) if is_physics else 0.0
        val_loss, epoch_val_f1, epoch_val_acc = evaluate(model, val_loader, loss_fn, device, is_physics=is_physics)
        
        if isinstance(scheduler, CosineAnnealingLR): scheduler.step()
        else: scheduler.step(val_loss)
        
        history['train_loss'].append(epoch_loss)
        history['val_f1'].append(epoch_val_f1)
        history['physics_loss'].append(epoch_phys)
        
        print(f"Summary -> Total: {epoch_loss:.4f} | Physics: {epoch_phys:.4f} | Val F1: {epoch_val_f1:.4f} | Physics Lambda: {current_lambda:.3f}")

        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), str(checkpoint_dir / "best_model.pth"))
            print("--- New Best Model Saved ---")

        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['patience_counter'] = 0
        else:
            history['patience_counter'] += 1
            if history['patience_counter'] >= config.get('model_hp', {}).get('early_stopping_patience', 25):
                break
        
    return history, model