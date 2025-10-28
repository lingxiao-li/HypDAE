import os
import torch
import shutil
import random
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

data_dir = '/mnt/workspace/gongkaixiong/fws/kaixuan/animal_faces/train_30' 
output_dir = '/mnt/workspace/gongkaixiong/fws/kaixuan/animal_faces/output'  
pretrained_model_path = '/mnt/workspace/gongkaixiong/fws/kaixuan/models/resnet/ptmodel_bs128.pth'  

train_size = 30
val_size = 35
test_size = 35
batch_size = 8
learning_rate = 0.001

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

def split_data(data_dir, train_size, val_size, test_size):
    subfolders = sorted(os.listdir(data_dir)) 
    train_images = []
    val_images = []
    test_images = []

    for subfolder in subfolders:
        subfolder_path = os.path.join(data_dir, subfolder)
        images = [os.path.join(subfolder_path, img) for img in os.listdir(subfolder_path)]
        random.shuffle(images)

        train_images += images[:train_size]
        val_images += images[train_size:train_size+val_size]
        test_images += images[train_size+val_size:train_size+val_size+test_size]
    
    return train_images, val_images, test_images

train_images, val_images, test_images = split_data(data_dir, train_size, val_size, test_size)

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        # self.folder_to_label = {folder.split('_')[0]: idx for idx, folder in enumerate(sorted(os.listdir(image_paths)))}
        self.folder_to_label = {}
        idx = 0
        for folder in self.image_paths:
            # print('folder: ',folder)
            if folder.split('/')[-2].split('_')[0] in self.folder_to_label:
                continue
            self.folder_to_label[folder.split('/')[-2].split('_')[0]] = idx
            idx+=1
        print(self.folder_to_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = image_path.split('/')[-2]  
        label = int(self.folder_to_label[label])
        return image, label

train_dataset = CustomDataset(train_images, transform)
val_dataset = CustomDataset(val_images, transform)
test_dataset = CustomDataset(test_images, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet18(pretrained=False)  
model.fc = nn.Linear(model.fc.in_features, 119)

model.load_state_dict(torch.load(pretrained_model_path))
model.fc = nn.Linear(model.fc.in_features, 30)  
nn.init.kaiming_normal_(model.fc.weight)
nn.init.zeros_(model.fc.bias)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total * 100
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total * 100

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print("Best model saved!")

    return model

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total * 100
    print(f"Test Accuracy: {test_acc:.2f}%")

def load_best_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model = model.to(device) 
    return model

model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=20)

model = load_best_model(model, os.path.join(output_dir, 'best_model.pth'))
test_model(model, test_loader)
