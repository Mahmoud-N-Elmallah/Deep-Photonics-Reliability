import torch
from tqdm import tqdm
from torch.amp import GradScaler
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau

def evaluate(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    device_type = str(device).split(':')[0]  # Extract 'cuda' since this is how the new methhod works instead of the old deprecated one
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            with torch.amp.autocast(device_type=device_type):
                outputs = model(X)
                loss = loss_fn(outputs, y)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    avg_loss = val_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, epochs, device, history=None, checkpoint_dir=None, start_epoch=0):
    if checkpoint_dir is None:
        checkpoint_dir = Path.cwd()
    if history is None:
        history = {}
    
    # Initialize history keys if not present
    if 'train_loss' not in history:
        history['train_loss'] = []
    if 'train_acc' not in history:
        history['train_acc'] = []
    if 'val_loss' not in history:
        history['val_loss'] = []
    if 'val_acc' not in history:
        history['val_acc'] = []
    if 'lr' not in history:
        history['lr'] = []
    if 'best_val_acc' not in history:
        history['best_val_acc'] = 0.0
    
    scaler = GradScaler()
    best_val_acc = history['best_val_acc']
    device_type = str(device).split(':')[0]  # Extract 'cuda' or 'cpu'
    
    # Define Scheduler: 
    # Reduce LR by factor of 0.1 if Val Loss doesn't improve for 3 epochs.
    
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
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
            correct_train += (preds == y).sum().item()
            total_train += y.size(0)
            
            loop.set_postfix(loss=loss.item(), acc=correct_train/total_train)

        # Evaluation
        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        val_loss, epoch_val_acc = evaluate(model, val_loader, loss_fn, device)
        

        # monitor val_loss to decide when to drop the LR
        scheduler.step(val_loss)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr']) # Track LR changes
        
        print(f"\nSummary -> Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f} | Val loss: {val_loss:.4f}| LR: {optimizer.param_groups[0]['lr']}")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            history['best_val_acc'] = best_val_acc
            best_model_path = checkpoint_dir / "best_photonics_fft_resnet50.pth"
            torch.save(model.state_dict(), str(best_model_path))
            print("New Best Model Saved!")

        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'best_acc': best_val_acc
            }
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, str(checkpoint_path))
            # Also save as latest checkpoint for easy resume
            latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pkl"
            torch.save(checkpoint, latest_checkpoint_path)
        
    return history, model