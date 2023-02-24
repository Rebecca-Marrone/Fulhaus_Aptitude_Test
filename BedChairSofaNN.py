
"""
Created on Thu Feb 23 17:28:17 2023

@author: Rebecca Marrone
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

# Image processing
class BedChairSofa(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_labels = df
        self.img_dir = img_dir
        self.dataset = ImageFolder(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0], self.img_labels.iloc[idx, 2])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
    
# Define the path to the folder containing the images
img_dir = 'Data for test'

# Get the list of subdirectories in the folder and create a list of class names
subfolders = [x[0] for x in os.walk(img_dir)][1:]

# Create pd DataFrame to store labels from folder provided
classes = []
subfolder = [x[0] for x in os.walk('Data for test')]
temp1 = []
for i in range(1,4):
    for filename in os.listdir(subfolder[i]):
        sub = subfolder[i][14:]
        temp1.append([sub, i, filename])
    classes.append(sub)      
imagesdf = pd.DataFrame(temp1, columns = ['Category', 'Class No.', 'File Name'])

# Split the dataset into training set and testing set
train_df, test_df = train_test_split(imagesdf, test_size=0.1, random_state=20)

# Define the transformations to be applied to the training set and testing set
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor()])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

# Create dataset and dataloader for the training set and testing set
train_dataset = BedChairSofa(train_df, img_dir, train_transform)
test_dataset = BedChairSofa(test_df, img_dir, test_transform)

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define function to set device (CPU or GPU)
def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

# Define function to train the neural network
def train_nn(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    device = set_device()
    best_acc = 0
    for epoch in range(num_epochs):
        print("Epoch %d" % (epoch + 1))
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            optimizer.zero_grad()
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels == pred).sum().item()
    
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * running_correct / total 
        test_acc = evaluate_test(model, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, epoch, optimizer, best_acc)
        print("Training Loss: {:.4f}, Training Accuracy: {:.2f}, Testing Accuracy: {:.2f}".format(epoch_loss, epoch_acc, test_acc))
    return model    

# Define function to evaluate the neural network
def evaluate_test(model, test_loader):
    model.eval()
    pred_correct_epoch = 0
    total = 0 
    device = set_device()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            _, pred = torch.max(outputs.data, 1)
            pred_correct_epoch += (pred == labels).sum().item()
    epoch_acc = 100.00 * pred_correct_epoch / total
    return epoch_acc

def save_checkpoint(model, epoch, optimizer, best_acc):
    state = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'best accuracy': best_acc,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, 'model_best_state.pth.tar')
    
resnet18_model = models.resnet18()
no_feat = resnet18_model.fc.in_features
no_class = 4
resnet18_model.fc = nn.Linear(no_feat, no_class) 
device = set_device()
resnet18_model = resnet18_model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.005)

train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimizer, 25)

checkpoint = torch.load('model_best_state.pth.tar')

resnet18_model.load_state_dict(checkpoint['model'])

torch.save(resnet18_model, 'best_model.pth')
