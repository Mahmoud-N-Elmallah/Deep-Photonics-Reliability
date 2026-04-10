import torch
from tqdm import tqdm
from torch.amp import GradScaler
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

def evaluate(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    device_type = str(device).split(':')[0]  # Extract 'cuda' since this is how the new methhod works instead of the old deprecated one
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            with torch.amp.autocast(device_type=device_type):
                outputs = model(X)
                loss = loss_fn(outputs, y)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = val_loss / len(loader)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, weighted_f1

def train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, epochs, device, history=None, checkpoint_dir=None, start_epoch=0):
    if checkpoint_dir is None:
        checkpoint_dir = Path.cwd()
    if history is None:
        history = {}
    
    # Initialize history keys if not present
    if 'train_loss' not in history:
        history['train_loss'] = []
    if 'train_f1' not in history:
        history['train_f1'] = []
    if 'val_loss' not in history:
        history['val_loss'] = []
    if 'val_f1' not in history:
        history['val_f1'] = []
    if 'lr' not in history:
        history['lr'] = []
    if 'best_val_f1' not in history:
        history['best_val_f1'] = 0.0
    
    scaler = GradScaler()
    best_val_f1 = history['best_val_f1']
    device_type = str(device).split(':')[0]  # Extract 'cuda' or 'cpu'
    
    # Define Scheduler: 
    # Reduce LR by factor of 0.1 if Val Loss doesn't improve for 3 epochs.
    
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device_type):
                outputs = model(X)
                loss = loss_fn(outputs, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y.cpu().numpy())
            
            current_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
            loop.set_postfix(loss=loss.item(), f1=current_f1)

        # Evaluation
        epoch_loss = running_loss / len(train_loader)
        epoch_train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
        val_loss, epoch_val_f1 = evaluate(model, val_loader, loss_fn, device)
        

        # monitor val_loss to decide when to drop the LR
        scheduler.step(val_loss)
        
        history['train_loss'].append(epoch_loss)
        history['train_f1'].append(epoch_train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(epoch_val_f1)
        history['lr'].append(optimizer.param_groups[0]['lr']) # Track LR changes
        
        print(f"\nSummary -> Loss: {epoch_loss:.4f} | Train F1: {epoch_train_f1:.4f} | Val F1: {epoch_val_f1:.4f} | Val loss: {val_loss:.4f}| LR: {optimizer.param_groups[0]['lr']}")

        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            history['best_val_f1'] = best_val_f1
            best_model_path = checkpoint_dir / "best_photonics_fft_resnet50.pth"
            torch.save(model.state_dict(), str(best_model_path))
            print("New Best Model Saved!")

        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'best_f1': best_val_f1
            }
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, str(checkpoint_path))
            # Also save as latest checkpoint for easy resume
            latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pkl"
            torch.save(checkpoint, latest_checkpoint_path)
        
    return history, model