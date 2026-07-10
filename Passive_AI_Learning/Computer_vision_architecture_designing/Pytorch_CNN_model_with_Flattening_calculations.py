import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision
import os
import matplotlib.pyplot as plt
# import mlflow
# import mlflow.pytorch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import seaborn as sns

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Allow PyTorch to leverage cuDNN for optimization
torch.backends.cudnn.benchmark = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# mlflow.set_experiment("Run_2_frozen_layers_image_dimensions_regnet_16GF")
# mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Define the main training process
def main():
    # Timing the whole code run
    start_time = time.time()
    
    model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=5),  # Output: 32 x 220 x 220
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # Output: 32 x 110 x 110
    
    nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: 64 x 110 x 110
    nn.ReLU(),
    
    nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: 128 x 110 x 110
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # Output: 128 x 55 x 55
    
    nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Output: 256 x 55 x 55
    nn.ReLU(),
    
    nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Output: 512 x 55 x 55
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # Output: 512 x 27 x 27
    
    nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # Output: 1024 x 27 x 27
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # Output: 1024 x 13 x 13
    
    nn.Flatten(),  # Output: 1024 * 13 * 13 = 173056
    nn.Linear(173056, 3400),
    nn.ReLU(),
    nn.Linear(3400, 5)
)

    # Path to the dataset
    train_image_path = r'C:\Users\pkrap\Desktop\Behavioral genetics\AI_Project_Python_Templates\output'

    # Define transformations (including grayscale)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
        transforms.ToTensor()
    ])

    # Load dataset
    good_dataset = ImageFolder(root=train_image_path, transform=transform)

    # Split dataset into training and testing subsets
    train_dataset, test_dataset = random_split(good_dataset, [int(0.8 * len(good_dataset)), len(good_dataset) - int(0.8 * len(good_dataset))])

    print("Total number of samples in the original dataset:", len(good_dataset))
    print("Number of samples in the training subset:", len(train_dataset))
    print("Number of samples in the testing subset:", len(test_dataset))

    # Define batch size
    BS = 8  # Larger batch size to fully utilize GPU memory

    # Dataloaders with multiple workers and pinned memory for faster data loading
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True)

   

    # Move the model to GPU
    model = model.to(device)

    lr = 0.001

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    # Use mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Early stopping parameters
    best_val_loss = float('inf')

    # Lists to track loss for plotting
    Loss = []
    Validation_Loss = []
    all_preds = []
    all_labels = []

    # Number of epochs to train
    num_epochs = 5

    # Initialize MLflow and start the experiment
    # mlflow.start_run()
    
    # Log parameters
    # mlflow.log_param("learning_rate", lr)
    # mlflow.log_param("batch_size", BS)
    # mlflow.log_param("num_epochs", num_epochs)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Training phase with mixed precision
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                output = model(img)
                loss = criterion(output, label)

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        Loss.append(train_loss / len(train_loader.dataset))

        # Validation phase
        model.eval()
        val_loss = 0
        preds = []
        labels = []
        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(device), label.to(device)

                with torch.cuda.amp.autocast():  # Use mixed precision in validation too
                    output = model(img)
                    loss = criterion(output, label)
                    preds.append(torch.argmax(output, dim=1).cpu().numpy())
                    labels.append(label.cpu().numpy())

                val_loss += loss.item()

        all_preds.extend(np.concatenate(preds))
        all_labels.extend(np.concatenate(labels))

        Validation_Loss.append(val_loss / len(test_loader.dataset))

        # Log training and validation loss to MLflow
        # mlflow.log_metric("train_loss", train_loss / len(train_loader.dataset), step=epoch)
        # mlflow.log_metric("validation_loss", val_loss / len(test_loader.dataset), step=epoch)

        # Print losses for this epoch
        if epoch % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}, Validation Loss: {val_loss / len(test_loader.dataset):.4f}')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Log the model as an artifact
            model_save_path = r'C:\Users\pkrap\Desktop\Behavioral genetics\AI_Project_Python_Templates\Vit_B_16_6_class_frozen_layers_test_alien.pt'
            torch.save(model.state_dict(), model_save_path)
            # mlflow.log_artifact(model_save_path)

    # Plot training and validation losses
    plt.plot(Loss, label='Training Loss')
    plt.plot(Validation_Loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig("loss_plot.png")
    
    # Log the loss plot as an artifact in MLflow
    # mlflow.log_artifact("loss_plot.png")

    # Classification report and confusion matrix
    class_report = classification_report(all_labels, all_preds, target_names=good_dataset.classes)
    print("Classification Report:\n", class_report)

    # # Log classification report to a text file in MLflow
    # with open("classification_report.txt", "w") as f:
    #     f.write(class_report)
    # mlflow.log_artifact("classification_report.txt")

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=good_dataset.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    # plt.savefig("confusion_matrix.png")
    # mlflow.log_artifact("confusion_matrix.png")

    # Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(recalls, precisions, marker='.', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    # plt.savefig("precision_recall_curve.png")
    # mlflow.log_artifact("precision_recall_curve.png")

# End the MLflow run
# mlflow.end_run()

# Safe entry point for Windows and other environments
if __name__ == '__main__':
    main()