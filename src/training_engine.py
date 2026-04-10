import torch
from tqdm import tqdm
from torch.amp import GradScaler
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import f1_score
from utils import unfreeze_layer4

def evaluate(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
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
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    avg_loss = val_loss / len(loader)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = correct / total
    
    return avg_loss, weighted_f1, accuracy

def train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, epochs, device, history=None, checkpoint_dir=None, start_epoch=0, config=None, stage1_epochs=None, stage2_lr=None):
    if checkpoint_dir is None:
        checkpoint_dir = Path.cwd()
    if history is None:
        history = {}
    
    # Initialize history keys if not present
    if 'train_loss' not in history:
        history['train_loss'] = []
    if 'train_f1' not in history:
        history['train_f1'] = []
    if 'train_acc' not in history:
        history['train_acc'] = []
    if 'val_loss' not in history:
        history['val_loss'] = []
    if 'val_f1' not in history:
        history['val_f1'] = []
    if 'val_acc' not in history:
        history['val_acc'] = []
    if 'lr' not in history:
        history['lr'] = []
    if 'best_val_f1' not in history:
        history['best_val_f1'] = 0.0
    if 'training_stage' not in history:
        history['training_stage'] = 'stage1' if stage1_epochs else 'full'
    
    scaler = GradScaler()
    best_val_f1 = history['best_val_f1']
    device_type = str(device).split(':')[0]  # Extract 'cuda' or 'cpu'
    
    # Check using staged training
    use_staged = stage1_epochs is not None and stage2_lr is not None
    
    # Detect if stage transition already happened (on resume)
    stage_transitioned = False
    if use_staged and start_epoch >= stage1_epochs:
        stage_transitioned = True
        print(f"Stage transition already completed. Resuming in Stage 2.\n")
    
    for epoch in range(start_epoch, epochs):
        # === STAGE TRANSITION: Move to stage 2 after stage1_epochs ===
        if use_staged and not stage_transitioned and epoch == stage1_epochs:
            print(f"\n{'='*60}")
            print(f"TRANSITIONING TO STAGE 2: Fine-tuning layer4 + head")
            print(f"{'='*60}\n")
            
            # Unfreeze layer4
            unfreeze_layer4(model)
            
            # Create new optimizer with stage2 LR
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage2_lr,
                                        weight_decay=config.get('model_hp', {}).get('weight_decay', 1e-4))
            
            # Switch to ReduceLROnPlateau for stage 2
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
            
            history['training_stage'] = 'stage2'
            stage_transitioned = True
            print(f"Stage 2 LR: {stage2_lr}, Epochs remaining: {epochs - epoch}\n")
        
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        train_correct = 0
        train_total = 0
        
        stage_label = "Stage 1" if (not stage_transitioned and use_staged) else ("Stage 2" if (stage_transitioned and use_staged) else "")
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} {stage_label}")
        
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
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            
            current_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
            current_acc = train_correct / train_total
            loop.set_postfix(loss=loss.item(), f1=current_f1, acc=current_acc)

        # Evaluation
        epoch_loss = running_loss / len(train_loader)
        epoch_train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
        epoch_train_acc = train_correct / train_total
        val_loss, epoch_val_f1, epoch_val_acc = evaluate(model, val_loader, loss_fn, device)
        
        # Scheduler step (handle both types of steps we r using)
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        else:
            scheduler.step(val_loss)
        
        history['train_loss'].append(epoch_loss)
        history['train_f1'].append(epoch_train_f1)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(epoch_val_f1)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr']) # Track LR changes
        
        print(f"\nSummary -> Loss: {epoch_loss:.4f} | Train F1: {epoch_train_f1:.4f} | Train Acc: {epoch_train_acc:.4f} | Val F1: {epoch_val_f1:.4f} | Val Acc: {epoch_val_acc:.4f} | Val loss: {val_loss:.4f}| LR: {optimizer.param_groups[0]['lr']}")

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
            # latest checkpoint for later  resuming of the training
            latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pkl"
            torch.save(checkpoint, latest_checkpoint_path)
        
    return history, model