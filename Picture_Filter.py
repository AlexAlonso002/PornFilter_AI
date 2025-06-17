import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Define transformations (same for both datasets)
transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    #transforms.RandomResizedCrop(400, scale=(0.8, 1.0)),  # Vary sizes slightly
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

# Path to the  dataset
data_path = "Binary_Classification"
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Print dataset info
print(f"Number of images in dataset: {len(dataset)}")
print(f"Classes in dataset: {dataset.classes}")  # Should show ['0', '1', ...]


class_indices = {label: [] for label in range(len(dataset.classes))}
for idx, (_, label) in enumerate(dataset.imgs):
    class_indices[label].append(idx)


max_samples_per_class = 4900
limited_indices = []

for class_label, indices in class_indices.items():
    print(f"Original size of class {class_label}: {len(indices)}")
    limited_indices.extend(indices[:max_samples_per_class])
    print(f"Limited size of class {class_label}: {len(indices[:max_samples_per_class])}")


dataset_limited = torch.utils.data.Subset(dataset, limited_indices)


#train_size = int(0.8 * len(dataset_limited))  
#test_size = len(dataset_limited) - train_size  
#train_dataset, test_dataset = random_split(dataset_limited, [train_size, test_size])

train_size = int(0.7 * len(dataset_limited))  
valid_size = int(0.1 * len(dataset_limited))  
test_size = len(dataset_limited) - train_size - valid_size  


train_dataset, valid_dataset, test_dataset = random_split(dataset_limited, [train_size, valid_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)


print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

device = 'cuda' if torch.cuda.is_available else 'cpu'
print(device)

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 333, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout = nn.Dropout(.5)  
        self.fc = nn.Linear(500000   , 2)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        #out2 = self.dropout(out2)
        out2 = self.layer3(out2)
        out3 = self.dropout(out2)
       # out3 = self.layer4(out3)
        out3 = out3.view(out3.size(0), -1)
        out4 = self.fc(out3)
        return out4


# Instantiate the CNN model
model = CNN().to(device)

# Print the model architecture
print(model)

import torch.optim as optim
#class_weights = torch.tensor([1.0, 1.7, 1.0, 1.0]).to(device)
#criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4 )
criterion = nn.CrossEntropyLoss().to(device)
#criterion = nn.BCEWithLogitsLoss().to(device)


num_epochs = 15
for epoch in range(num_epochs):
    # Training phase
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad() 
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total * 100


    model.eval()  
    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

    valid_epoch_loss = valid_loss / len(valid_loader)
    valid_epoch_accuracy = valid_correct / valid_total * 100

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    print(f"Validation Loss: {valid_epoch_loss:.4f}, Validation Accuracy: {valid_epoch_accuracy:.2f}%")



model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


torch.save(model.state_dict(), 'FinalModel???.pth')

print("Model saved to FinalModel???.pth")

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predicted = (outputs > 0).long().squeeze(1)  

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
