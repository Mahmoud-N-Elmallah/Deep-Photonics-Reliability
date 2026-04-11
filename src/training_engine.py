import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import GradScaler
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import f1_score
from utils import *

def evaluate(model, loader, loss_fn, device, is_physics=False):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    device_type = str(device).split(':')[0]
    
    with torch.no_grad():
        for batch in loader:
            if is_physics:
                X, mask, y = batch
            else:
                X, y = batch
                
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
    
    if checkpoint_dir is None:
        checkpoint_dir = Path.cwd()
    if history is None:
        history = {}
    
    # Initialize history keys
    keys = ['train_loss', 'train_f1', 'train_acc', 'val_loss', 'val_f1', 'val_acc', 'lr', 'physics_loss']
    for k in keys:
        if k not in history: history[k] = []
    if 'best_val_f1' not in history: history['best_val_f1'] = 0.0
    if 'best_val_loss' not in history: history['best_val_loss'] = float('inf')
    if 'patience_counter' not in history: history['patience_counter'] = 0
    
    scaler = GradScaler()
    best_val_f1 = history['best_val_f1']
    device_type = str(device).split(':')[0]
    
    # Physics Loss Hyperparams
    physics_lambda = config.get('model_hp', {}).get('physics_lambda', 0.1) if config else 0.1
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        running_phys = 0.0
        train_preds, train_labels = [], []
        train_correct, train_total = 0, 0
        
        desc = f"Epoch {epoch+1}/{epochs} {'(Physics-Constrained)' if is_physics else ''}"
        loop = tqdm(train_loader, desc=desc)
        
        for batch in loop:
            if is_physics:
                X, target_mask, y = batch
                target_mask = target_mask.to(device)
            else:
                X, y = batch
                
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device_type):
                if is_physics:
                    outputs, attn_map = model(X, return_attention=True)
                    class_loss = loss_fn(outputs, y)
                    
                    # Compute Physics Loss (Alignment between Attention and Pseudo-mask)
                    # Resize target mask to match attention map size (usually 7x7)
                    attn_size = attn_map.shape[-1]
                    target_mask_low = F.interpolate(target_mask, size=(attn_size, attn_size), mode='bilinear', align_corners=False)
                    
                    phys_loss = F.mse_loss(attn_map, target_mask_low)
                    loss = class_loss + (physics_lambda * phys_loss)
                    running_phys += phys_loss.item()
                else:
                    outputs = model(X)
                    loss = loss_fn(outputs, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            max_norm = config.get('model_hp', {}).get('clip_grad_norm', 1.0) if config else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y.cpu().numpy())
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            
            current_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
            loop.set_postfix(loss=loss.item(), f1=current_f1)

        # Epoch Summaries
        epoch_loss = running_loss / len(train_loader)
        epoch_phys = running_phys / len(train_loader) if is_physics else 0.0
        epoch_train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
        epoch_train_acc = train_correct / train_total
        val_loss, epoch_val_f1, epoch_val_acc = evaluate(model, val_loader, loss_fn, device, is_physics=is_physics)
        
        if isinstance(scheduler, CosineAnnealingLR): scheduler.step()
        else: scheduler.step(val_loss)
        
        history['train_loss'].append(epoch_loss)
        history['physics_loss'].append(epoch_phys)
        history['train_f1'].append(epoch_train_f1)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(epoch_val_f1)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"\nSummary -> Total Loss: {epoch_loss:.4f} | Physics: {epoch_phys:.4f} | Val F1: {epoch_val_f1:.4f} | Val Acc: {epoch_val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']}")

        # Save Best Model Logic
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            history['best_val_f1'] = best_val_f1
            best_model_name = config.get('experiment', {}).get('best_model_name', 'best_model.pth') if config else 'best_model.pth'
            torch.save(model.state_dict(), str(checkpoint_dir / best_model_name))
            print("New Best Model Saved!")

        # Checkpoint Logic
        if (epoch + 1) % 5 == 0:
            checkpoint = {'epoch': epoch+1, 'model_state_dict': model.state_dict(), 
                          'optimizer_state_dict': optimizer.state_dict(), 'history': history, 'best_f1': best_val_f1}
            torch.save(checkpoint, str(checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"))
            torch.save(checkpoint, str(checkpoint_dir / "latest_checkpoint.pkl"))
            
        # Early Stopping check
        best_val_loss = history['best_val_loss']
        patience = config.get('model_hp', {}).get('early_stopping_patience', 15)
        if val_loss < best_val_loss:
            history['best_val_loss'] = val_loss
            history['patience_counter'] = 0
        else:
            history['patience_counter'] += 1
            if history['patience_counter'] >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
    return history, model