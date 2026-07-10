from pathlib import Path
import numpy as np
import os, shutil
import matplotlib.pyplot as plt

from PIL import Image

from tqdm.auto import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data.dataset import Subset
from torch import nn

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(1024 * 7 * 7, 1024)
        self.fc_logvar = nn.Linear(1024 * 7 * 7, 1024)
        self.fc_decode = nn.Linear(1024, 1024 * 7 * 7)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.fc_decode(z)
        x = x.view(x.size(0), 1024, 7, 7)
        x = self.decoder(x)
        return x, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

torch.manual_seed(41)

model = VAE().cuda()

train_image_path = r'C:\Users\pkrap\Desktop\Behavioral genetics\AI_Project_Python_Templates\Alibi_detect\Tejia_data_alibi_detect'

transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()])

good_dataset = ImageFolder(root=train_image_path, transform=transform)

x, y = good_dataset[0]
print("Image Shape:", x.shape)
print("Label:", y)

train_dataset, test_dataset = torch.utils.data.random_split(good_dataset, [0.8, 0.2])

print("Total number of samples in the original dataset:", len(good_dataset))
print("Number of samples in the training subset:", len(train_dataset))
print("Number of samples in the testing subset:", len(test_dataset))

BS = 4

train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

image_batch, label_batch = next(iter(train_loader))

print(f'Shape of input images: {image_batch.shape}')
print(f'Shape of labels: {label_batch.shape}')

grid = torchvision.utils.make_grid(image_batch[0:4], padding=5, nrow=4)
plt.imshow(grid.permute(1, 2, 0))
plt.title('Good Samples')
plt.show()

criterion = loss_function
optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001)

Loss = []
Validation_Loss = []

num_epochs = 400
for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0
    for img, _ in train_loader:
        img = img.cuda()
        
        output, mu, logvar = model(img)
        loss = criterion(output, img, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    Loss.append(train_loss / len(train_loader.dataset))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for img, _ in test_loader:
            img = img.cuda()
            output, mu, logvar = model(img)
            loss = criterion(output, img, mu, logvar)
            val_loss += loss.item()
    Validation_Loss.append(val_loss / len(test_loader.dataset))
    
    if epoch % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / len(train_loader.dataset), val_loss / len(test_loader.dataset)))

plt.plot(Loss, label='Training Loss')
plt.plot(Validation_Loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

torch.save(model.state_dict(), r'C:\Users\pkrap\Desktop\Behavioral genetics\AI_Project_Python_Templates\CNN_image_identification_zebrafish\vae_autoencoder.pth')
model.eval()

ckpoints = torch.load(r'C:\Users\pkrap\Desktop\Behavioral genetics\AI_Project_Python_Templates\CNN_image_identification_zebrafish\vae_autoencoder.pth')
model.load_state_dict(ckpoints)

def load_and_transform_image(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = transform(image)
    return image

test_image_1 = load_and_transform_image(r'C:\Users\pkrap\Desktop\Behavioral genetics\AI_Project_Python_Templates\Alibi_detect\Test\hexa_fish2_4.tif')
test_image_2 = load_and_transform_image(r'C:\Users\pkrap\Desktop\Behavioral genetics\AI_Project_Python_Templates\Alibi_detect\Test\hexa_fish1_8.tif')
test_image_3 = load_and_transform_image(r'C:\Users\pkrap\Desktop\Behavioral genetics\AI_Project_Python_Templates\Alibi_detect\Test\eyeless.png')

data = torch.stack([test_image_1, test_image_2, test_image_3])

with torch.no_grad():
    data = data.cuda()
    recon, _, _ = model(data)

recon_error = ((data - recon) ** 2).mean(dim=1)

cmap = 'jet'
plt.figure(dpi=2600)
fig, ax = plt.subplots(3, 3, figsize=(5 * 4, 4 * 4))
for i in range(3):
    ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
    ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
    ax[2, i].imshow(recon_error[i].cpu().numpy(), cmap=cmap, vmax=torch.max(recon_error[i]))
    ax[0, i].axis('OFF')
    ax[1, i].axis('OFF')
    ax[2, i].axis('OFF')
plt.show()

with torch.no_grad():
    for data, _ in train_loader:
        data = data.cuda()
        recon, _, _ = model(data)
        break

recon_error = ((data - recon) ** 2).mean(dim=1)
print(recon_error.shape)

plt.figure(dpi=600)
fig, ax = plt.subplots(3, 3, figsize=(5 * 4, 4 * 4))
for i in range(3):
    ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
    ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
    ax[2, i].imshow(recon_error[i].cpu().numpy(), cmap=cmap, vmax=torch.max(recon_error[i]))
    ax[0, i].axis('OFF')
    ax[1, i].axis('OFF')
    ax[2, i].axis('OFF')
plt.show()
