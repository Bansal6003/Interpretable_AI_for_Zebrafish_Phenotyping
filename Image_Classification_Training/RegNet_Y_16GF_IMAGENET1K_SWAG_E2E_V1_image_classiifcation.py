import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import seaborn as sns
from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights
# from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights
from av.video.frame import VideoFrame

# Allow PyTorch to leverage cuDNN for optimization
torch.backends.cudnn.benchmark = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

mlflow.set_experiment("RegNet_Y_16GF_SWAG_E2E_V1")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

class CustomRegNetY16GF(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomRegNetY16GF, self).__init__()
        # Load RegNet_Y_16GF with IMAGENET1K_SWAG_E2E_V1 weights
        weights = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
        self.regnet = regnet_y_16gf(weights=weights)
        
        # Modify the classifier head
        num_features = self.regnet.fc.in_features
        self.regnet.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.regnet(x)

def main():
    # Timing the whole code run
    start_time = time.time()

    # Path to the dataset
    train_image_path = r"D:\Behavioral genetics_V1\Metamorph_scans\Echo Images\orientations\True_Orientations_1500_BG\2_classes\output"

    # Define transformations (including grayscale)
    # EfficientNetV2-S optimal input size is 384x384
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    good_dataset = ImageFolder(root=train_image_path, transform=transform)

    print("Total number of samples in the original dataset:", len(good_dataset))

    # Define batch size - EfficientNetV2 can handle larger batch sizes
    BS = 32

    # Create the custom model
    model = CustomRegNetY16GF(num_classes=2)
    model = model.to(device)
    # model = torch.compile(model)

    # Hyperparameters 
    lr = 5e-5
    weight_decay = 1e-5
    T_max = 100
    # eta_min = 1e-7
#
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)

    # Use mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0

    # Lists to track metrics
    Loss = []
    Validation_Loss = []
    Accuracy = []
    all_preds = []
    all_labels = []

    # Number of epochs
    num_epochs = 200

    # Initialize MLflow and start the experiment
    mlflow.start_run()

    # Log parameters
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("batch_size", BS)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("weight_decay", weight_decay)
    mlflow.log_param("Patience", patience)
    mlflow.log_param("T Max", T_max)
    # mlflow.log_param("eta_min", eta_min)

    # Split dataset
    train_size = int(0.8 * len(good_dataset))  
    test_size = len(good_dataset) - train_size
    train_dataset, test_dataset = random_split(good_dataset, [train_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False, num_workers=8, pin_memory=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # Training phase
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                output = model(img)
                loss = criterion(output, label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

        scheduler.step()
        epoch_accuracy = 100. * correct / total
        Accuracy.append(epoch_accuracy)
        Loss.append(train_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0
        preds = []
        labels = []
        
        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(device), label.to(device)
                
                with torch.amp.autocast('cuda'):
                    output = model(img)
                    loss = criterion(output, label)
                    
                    preds.append(torch.argmax(output, dim=1).cpu().numpy())
                    labels.append(label.cpu().numpy())
                
                val_loss += loss.item()

        all_preds.extend(np.concatenate(preds))
        all_labels.extend(np.concatenate(labels))
        
        val_loss = val_loss / len(test_loader)
        Validation_Loss.append(val_loss)

        # Log metrics
        mlflow.log_metric("train_loss", train_loss / len(train_loader), step=epoch)
        mlflow.log_metric("validation_loss", val_loss, step=epoch)
        mlflow.log_metric("accuracy", epoch_accuracy, step=epoch)

        # Print progress
        if epoch % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}, '
                  f'Val Loss: {val_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = r'D:\Behavioral genetics_V1\Metamorph_scans\Workflow_codes_V2\Pretrained_and_non_custom_algorithms\RegNet_Y_16_Pretrained\RegNet_Y_16GF_2_class_orients_cropped_P50_2.pt'
            torch.save(model.state_dict(), save_path)
            mlflow.log_artifact(save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Plot and save metrics
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(Loss, label='Training Loss')
    plt.plot(Validation_Loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(Accuracy, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    mlflow.log_artifact("training_metrics.png")

    # Classification report
    class_report = classification_report(all_labels, all_preds, target_names=good_dataset.classes, digits=4)
    print("Classification Report:\n", class_report)
    with open("classification_report.txt", "w") as f:
        f.write(class_report)
    mlflow.log_artifact("classification_report.txt")

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=good_dataset.classes,
                yticklabels=good_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # End MLflow run
    mlflow.end_run()

    print(f"Total training time: {(time.time() - start_time) / 3600:.2f} hours")

if __name__ == '__main__':
    main()