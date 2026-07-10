import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SegmentationMetrics:
    """Calculate comprehensive segmentation metrics"""
    
    @staticmethod
    def dice_coefficient(pred, target, smooth=1e-6):
        """Calculate Dice coefficient"""
        pred = torch.sigmoid(pred)
        # red = torch.tanh(pred)
        pred = (pred > 0.5).float()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.mean()
    
    @staticmethod
    def iou_score(pred, target, smooth=1e-6):
        """Calculate IoU (Jaccard) score"""
        pred = torch.sigmoid(pred)
        # pred = torch.tanh(pred)
        pred = (pred > 0.5).float()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()
    
    @staticmethod
    def pixel_accuracy(pred, target):
        """Calculate pixel-wise accuracy"""
        pred = torch.sigmoid(pred)
        # pred = torch.tanh(pred)
        pred = (pred > 0.5).float()
        
        correct = (pred == target).float()
        accuracy = correct.sum() / correct.numel()
        return accuracy
    
    @staticmethod
    def sensitivity_specificity(pred, target):
        """Calculate sensitivity (recall) and specificity"""
        pred = torch.sigmoid(pred)
        # pred = torch.tanh(pred)
        pred = (pred > 0.5).float()
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True positives, false positives, true negatives, false negatives
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        sensitivity = tp / (tp + fn + 1e-6)  # Recall/True Positive Rate
        specificity = tn / (tn + fp + 1e-6)  # True Negative Rate
        precision = tp / (tp + fp + 1e-6)
        
        return sensitivity, specificity, precision

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image as grayscale
        image = Image.open(self.image_paths[idx]).convert('L')
        mask = np.array(Image.open(self.mask_paths[idx]))
        
        # Convert grayscale to 3 channels
        if self.transform:
            image = self.transform(image)
            # Duplicate grayscale channel to 3 channels
            image = image.repeat(3, 1, 1)
        
        # Process mask: convert to binary
        processed_mask = (mask > 0).astype(np.float32)
            
        # Resize mask
        mask_pil = Image.fromarray(processed_mask)
        mask_resized = transforms.Resize((512, 512), 
                                       interpolation=transforms.InterpolationMode.NEAREST)(mask_pil)
        mask_tensor = torch.from_numpy(np.array(mask_resized))
        
        return image, mask_tensor.float()

def evaluate_metrics(model, data_loader, criterion, device):
    """Comprehensive evaluation of segmentation metrics"""
    model.eval()
    
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_accuracy = 0
    total_sensitivity = 0
    total_specificity = 0
    total_precision = 0
    
    batch_count = 0
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Skip batches with size 1 during evaluation to avoid batch norm issues
            if images.size(0) == 1:
                continue
                
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks.unsqueeze(1))
            
            # Calculate metrics
            dice = SegmentationMetrics.dice_coefficient(outputs, masks.unsqueeze(1))
            iou = SegmentationMetrics.iou_score(outputs, masks.unsqueeze(1))
            accuracy = SegmentationMetrics.pixel_accuracy(outputs, masks.unsqueeze(1))
            sensitivity, specificity, precision = SegmentationMetrics.sensitivity_specificity(outputs, masks.unsqueeze(1))
            
            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()
            total_accuracy += accuracy.item()
            total_sensitivity += sensitivity.item()
            total_specificity += specificity.item()
            total_precision += precision.item()
            
            batch_count += 1
    
    # Handle case where no valid batches were processed
    if batch_count == 0:
        print("Warning: No valid batches for evaluation (all had batch size 1)")
        return {
            'loss': 0.0,
            'dice': 0.0,
            'iou': 0.0,
            'accuracy': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'precision': 0.0,
            'f1_score': 0.0
        }
    
    # Calculate averages
    metrics = {
        'loss': total_loss / batch_count,
        'dice': total_dice / batch_count,
        'iou': total_iou / batch_count,
        'accuracy': total_accuracy / batch_count,
        'sensitivity': total_sensitivity / batch_count,
        'specificity': total_specificity / batch_count,
        'precision': total_precision / batch_count
    }
    
    # Calculate F1 score from precision and sensitivity
    if metrics['precision'] + metrics['sensitivity'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['sensitivity']) / (metrics['precision'] + metrics['sensitivity'])
    else:
        metrics['f1_score'] = 0.0
    
    return metrics

def create_prediction_plots(model, val_loader, device, epoch, save_path='predictions_plot.png'):
    """Create visualization of predictions vs ground truth"""
    model.eval()
    
    # Get a batch of validation data with size > 1
    for images, masks in val_loader:
        if images.size(0) > 1:  # Only use batches with more than 1 sample
            break
    else:
        print("Warning: No suitable batch found for prediction plots")
        return save_path
    
    images = images.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.sigmoid(outputs)
        # predictions = torch.tanh(outputs)
        predictions = (predictions > 0.5).float()
    
    # Create subplot for first 4 images
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i in range(min(4, images.shape[0])):
        # Original image (convert from 3-channel back to grayscale for display)
        img = images[i][0].cpu().numpy()  # Take first channel
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Input Image {i+1}')
        axes[0, i].axis('off')
        
        # Ground truth mask
        mask = masks[i].cpu().numpy()
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Ground Truth {i+1}')
        axes[1, i].axis('off')
        
        # Prediction
        pred = predictions[i][0].cpu().numpy()
        axes[2, i].imshow(pred, cmap='gray')
        axes[2, i].set_title(f'Prediction {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

def setup_mlflow():
    """Setup MLflow tracking"""
    print("\n" + "="*60)
    print("MLFLOW SETUP")
    print("="*60)
    
    # Set tracking URI to your MLflow server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Set experiment name
    experiment_name = "Efficient_NetV2_echos"
    try:
        mlflow.set_experiment(experiment_name)
        print(f"Using experiment: {experiment_name}")
    except Exception as e:
        print(f"Error setting experiment: {e}")
        return False
    
    print("MLflow server should be running at: http://127.0.0.1:5000")
    print("="*60)
    
    # Test MLflow functionality
    print("\nTesting MLflow connection...")
    try:
        with mlflow.start_run(run_name="connection_test") as test_run:
            mlflow.log_param("test_param", "connection_working")
            mlflow.log_metric("test_metric", 1.0)
            print(f"✓ MLflow connection successful! Run ID: {test_run.info.run_id}")
        print("✓ MLflow server connection is working!\n")
        return True
    except Exception as e:
        print(f"✗ MLflow connection failed: {e}")
        print("Please ensure:")
        print("1. MLflow server is running: mlflow server --host 127.0.0.1 --port 5000")
        print("2. Server is accessible at http://127.0.0.1:5000")
        return False

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100):
    """Train the segmentation model with comprehensive MLflow logging"""
    
    # Setup MLflow
    if not setup_mlflow():
        print("MLflow setup failed. Training without logging.")
        use_mlflow = False
    else:
        use_mlflow = True
    
    
    model.to(device)
    best_val_dice = 0.0
    patience = 100
    patience_counter = 0
    scaler = torch.amp.GradScaler('cuda')
    
    # Lists to store metrics
    train_metrics_history = []
    val_metrics_history = []
    
    try:
        # Start MLflow run
        if use_mlflow:
            run = mlflow.start_run(run_name=f"DeepLabV3Plus_training_{int(time.time())}")
            print(f"Started MLflow run: {run.info.run_id}")
            print(f"View at: http://127.0.0.1:5000/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{run.info.run_id}")
            
            # Log hyperparameters
            mlflow.log_params({
                "model_architecture": "DeepLabV3Plus",
                "encoder": "RegNet_Y_16",
                "num_epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "initial_lr": optimizer.param_groups[0]['lr'],
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingWarmRestarts",
                "loss_function": "BCEWithLogitsLoss",
                "input_size": "512x512",
                "patience": patience,
                "device": str(device),
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset)
            })
        
        # Training loop
        print(f"\nStarting training for {num_epochs} epochs...")
        print("="*80)
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0
            batch_count = 0
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(device)
                masks = masks.to(device)
                
                # Skip batches with size 1 to avoid batch normalization issues
                if images.size(0) == 1:
                    print(f"Skipping batch {batch_idx} with size 1")
                    continue
                
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks.unsqueeze(1))
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                batch_count += 1
            
            # Evaluation
            train_metrics = evaluate_metrics(model, train_loader, criterion, device)
            val_metrics = evaluate_metrics(model, val_loader, criterion, device)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store metrics
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print progress
            print(f'Epoch [{epoch+1:3d}/{num_epochs}] | Time: {epoch_time:5.1f}s | LR: {current_lr:.2e}')
            print(f'  Train: Loss={train_metrics["loss"]:.4f} | Dice={train_metrics["dice"]:.4f} | IoU={train_metrics["iou"]:.4f}')
            print(f'  Val  : Loss={val_metrics["loss"]:.4f} | Dice={val_metrics["dice"]:.4f} | IoU={val_metrics["iou"]:.4f} | F1={val_metrics["f1_score"]:.4f}')
            
            # Log to MLflow
            if use_mlflow:
                try:
                    mlflow.log_metrics({
                        # Training metrics
                        "train_loss": train_metrics["loss"],
                        "train_dice": train_metrics["dice"],
                        "train_iou": train_metrics["iou"],
                        "train_accuracy": train_metrics["accuracy"],
                        "train_sensitivity": train_metrics["sensitivity"],
                        "train_specificity": train_metrics["specificity"],
                        "train_precision": train_metrics["precision"],
                        "train_f1_score": train_metrics["f1_score"],
                        
                        # Validation metrics
                        "val_loss": val_metrics["loss"],
                        "val_dice": val_metrics["dice"],
                        "val_iou": val_metrics["iou"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_sensitivity": val_metrics["sensitivity"],
                        "val_specificity": val_metrics["specificity"],
                        "val_precision": val_metrics["precision"],
                        "val_f1_score": val_metrics["f1_score"],
                        
                        # Training info
                        "learning_rate": current_lr,
                        "epoch_time": epoch_time
                    }, step=epoch)
                except Exception as e:
                    print(f"  Warning: MLflow logging failed: {e}")
            
            # Early stopping and model saving
            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                patience_counter = 0
                print(f'  ✓ New best model saved (Dice: {best_val_dice:.4f})')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'  Early stopping triggered after {patience} epochs without improvement')
                    break
            
            # Create visualizations every 10 epochs
            if (epoch + 1) % 10 == 0:
                try:
                    pred_plot_path = f'predictions_epoch_{epoch+1}.png'
                    create_prediction_plots(model, val_loader, device, epoch, pred_plot_path)
                    
                    if use_mlflow:
                        mlflow.log_artifact(pred_plot_path)
                    
                    print(f'  ✓ Prediction plots saved')
                except Exception as e:
                    print(f'  Warning: Failed to create prediction plots: {e}')
            
            # Create metrics plots every 5 epochs
            if (epoch + 1) % 50 == 0 and len(train_metrics_history) > 1:
                try:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                    
                    epochs_range = range(1, len(train_metrics_history) + 1)
                    
                    # Loss plot
                    ax1.plot(epochs_range, [m['loss'] for m in train_metrics_history], 'b-', label='Train Loss')
                    ax1.plot(epochs_range, [m['loss'] for m in val_metrics_history], 'r-', label='Val Loss')
                    ax1.set_title('Loss Over Time')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Dice and IoU plot
                    ax2.plot(epochs_range, [m['dice'] for m in train_metrics_history], 'b-', label='Train Dice')
                    ax2.plot(epochs_range, [m['dice'] for m in val_metrics_history], 'r-', label='Val Dice')
                    ax2.plot(epochs_range, [m['iou'] for m in train_metrics_history], 'b--', label='Train IoU')
                    ax2.plot(epochs_range, [m['iou'] for m in val_metrics_history], 'r--', label='Val IoU')
                    ax2.set_title('Dice & IoU Over Time')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Score')
                    ax2.legend()
                    ax2.grid(True)
                    
                    # Precision, Recall, F1 plot
                    ax3.plot(epochs_range, [m['precision'] for m in val_metrics_history], 'g-', label='Precision')
                    ax3.plot(epochs_range, [m['sensitivity'] for m in val_metrics_history], 'b-', label='Recall')
                    ax3.plot(epochs_range, [m['f1_score'] for m in val_metrics_history], 'r-', label='F1 Score')
                    ax3.set_title('Precision, Recall & F1 Score')
                    ax3.set_xlabel('Epoch')
                    ax3.set_ylabel('Score')
                    ax3.legend()
                    ax3.grid(True)
                    
                    # Accuracy and Specificity plot
                    ax4.plot(epochs_range, [m['accuracy'] for m in val_metrics_history], 'purple', label='Accuracy')
                    ax4.plot(epochs_range, [m['specificity'] for m in val_metrics_history], 'orange', label='Specificity')
                    ax4.set_title('Accuracy & Specificity')
                    ax4.set_xlabel('Epoch')
                    ax4.set_ylabel('Score')
                    ax4.legend()
                    ax4.grid(True)
                    
                    plt.tight_layout()
                    metrics_plot_path = f'metrics_plot_epoch_{epoch+1}.png'
                    plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    if use_mlflow:
                        mlflow.log_artifact(metrics_plot_path)
                    
                    print(f'  ✓ Metrics plots saved')
                except Exception as e:
                    print(f'  Warning: Failed to create metrics plots: {e}')
            
            print("-" * 80)
        
        # Log final summary
        if use_mlflow and train_metrics_history:
            try:
                final_summary = {
                    "final_train_dice": train_metrics_history[-1]["dice"],
                    "final_val_dice": val_metrics_history[-1]["dice"],
                    "final_train_iou": train_metrics_history[-1]["iou"],
                    "final_val_iou": val_metrics_history[-1]["iou"],
                    "best_val_dice_achieved": best_val_dice,
                    "total_epochs_trained": len(train_metrics_history),
                    "early_stopped": patience_counter >= patience
                }
                mlflow.log_params(final_summary)
                print("✓ Final summary logged to MLflow")
            except Exception as e:
                print(f"Warning: Failed to log final summary: {e}")
        
        # Print final results
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        if train_metrics_history:
            print(f"Best Validation Dice Score: {best_val_dice:.4f}")
            print(f"Final Validation Dice Score: {val_metrics_history[-1]['dice']:.4f}")
            print(f"Final Validation IoU Score: {val_metrics_history[-1]['iou']:.4f}")
            print(f"Final Validation F1 Score: {val_metrics_history[-1]['f1_score']:.4f}")
            print(f"Total Epochs Trained: {len(train_metrics_history)}")
            print(f"Early Stopped: {'Yes' if patience_counter >= patience else 'No'}")
        
        if use_mlflow:
            print(f"\n✓ All results logged to MLflow server")
            print(f"  View results at: http://127.0.0.1:5000")
            print(f"  Experiment: Efficient_NetV2_auged_mic_drop_meet_V1")
        print("="*80)
    
    except Exception as e:
        print(f"Training error: {str(e)}")
        if use_mlflow:
            try:
                mlflow.log_param("training_error", str(e))
            except:
                print("Could not log error to MLflow")
        raise e
    finally:
        if use_mlflow:
            try:
                mlflow.end_run()
                print("✓ MLflow run ended successfully")
            except:
                print("Error ending MLflow run")
    
    return model

def main():
    """Main training function"""
    print("="*80)
    print("LARVA SEGMENTATION TRAINING WITH DEEPLABV3PLUS")
    print("="*80)
    
    # Dataset Paths
    image_dir = r"C:\Users\Pushkar Bansal\Downloads\Ventral_eye.v10i.yolo26\train\images"
    mask_dir = r"C:\Users\Pushkar Bansal\Downloads\Ventral_eye.v10i.yolo26\masks"
    
    # Debug directory existence
    print("\nChecking directories...")
    print(f"Image directory exists: {os.path.exists(image_dir)}")
    print(f"Mask directory exists: {os.path.exists(mask_dir)}")
    
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print("ERROR: One or both directories do not exist!")
        return
    
    # Get all files
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)
    
    print(f"\nFound {len(image_files)} image files and {len(mask_files)} mask files")
    print("First few files:")
    print("Images:", image_files[:5])
    print("Masks:", mask_files[:5])
    
    # Get file paths with supported extensions
    image_paths = sorted([os.path.join(image_dir, f) for f in image_files 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in mask_files 
                        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))])
    
    print(f"\nValid image files: {len(image_paths)}")
    print(f"Valid mask files: {len(mask_paths)}")
    
    # Create matching pairs
    image_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
    mask_dict = {os.path.splitext(os.path.basename(p))[0].replace('_mask_0', ''): p for p in mask_paths}
    
    print("\nExample processed filenames:")
    print("Image names:", list(image_dict.keys())[:5])
    print("Mask names:", list(mask_dict.keys())[:5])
    
    # Find common filenames
    common_names = set(image_dict.keys()) & set(mask_dict.keys())
    print(f"\nFound {len(common_names)} matching image-mask pairs")
    
    if len(common_names) == 0:
        print("ERROR: No matching image-mask pairs found!")
        print("Check naming conventions and file extensions")
        return
    
    # Create lists of matched pairs
    valid_pairs = []
    for name in common_names:
        img_path = image_dict[name]
        mask_path = mask_dict[name]
        if os.path.exists(img_path) and os.path.exists(mask_path):
            valid_pairs.append((img_path, mask_path))
    
    print(f"Valid pairs found: {len(valid_pairs)}")
    
    if len(valid_pairs) == 0:
        print("ERROR: No valid image-mask pairs found!")
        return
    
    # Show some examples
    print("\nFirst few matched pairs:")
    for i, (img, mask) in enumerate(valid_pairs[:3]):
        print(f"{i+1}. Image: {os.path.basename(img)}")
        print(f"   Mask:  {os.path.basename(mask)}")
    
    # Unzip the valid pairs
    image_paths, mask_paths = zip(*valid_pairs)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Split dataset
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Create datasets
    train_dataset = SegmentationDataset(train_images, train_masks, transform)
    val_dataset = SegmentationDataset(val_images, val_masks, transform)
    
    # Create data loaders with drop_last=True to avoid batch size 1 issues
    batch_size = 32 #8
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True  # This ensures we never get batches of size 1
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False  # Keep all validation samples
    )
    
    print(f"Data loaders created:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    print(f"\nInitializing DeepLabV3Plus model...")
    model = smp.DeepLabV3Plus(
        encoder_name="timm-regnety_160",
        encoder_weights="swsl",   # or None
        in_channels=3,
        classes=1,
    )
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6) #lr=1e-4, weight_decay=1e-4
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    print("✓ Model, loss, optimizer, and scheduler initialized")
    
    # Train model
    print(f"\nStarting training...")
    try:
        trained_model = train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer,
            scheduler,
            num_epochs=300
        )
        
        # Save final model
        final_save_path = r"D:\Behavioral genetics_V1\Metamorph_scans\Workflow_codes_V2\V52_pericardium_skips_testing_with_dual_seg_models\V50_same as_V49\CODE_FILES\Best_Models_regularly_updated\Class_seg_models\regnety16_swsl_echo_Ventral_eye_patience100_1.pth"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'num_classes': 1,
            'model_architecture': 'DeepLabV3Plus',
            'encoder': 'resnet50',
            'input_size': (512, 512),
            'training_completed': True,
            'timestamp': time.time()
        }, final_save_path)
        
        print(f"\n✓ Training completed successfully!")
        print(f"✓ Final model saved to: {final_save_path}")
        print(f"\nTo view training results:")
        print(f"1. Open: http://127.0.0.1:5000")
        print(f"2. Navigate to 'Efficient_NetV2_auged_mic_drop_meet_V1' experiment")
        print(f"3. View your training run for detailed metrics and plots")
        
    except Exception as e:
        print(f"\n✗ Training failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()