import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import random
from typing import Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class MembershipDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.membership = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        label = self.labels[index]
        member = self.membership[index]
        if self.transform is not None:
            img = self.transform(img)
        return id_, img, label, member

    def __len__(self):
        return len(self.ids)


pub_data = torch.load("/home/adugast/tml_2025_task1/pub.pt", map_location=torch.device('cpu'))


mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]
transform = transforms.Compose([
    transforms.Normalize(mean, std)
])

# Train models on non members only
non_member_entries = [entry for entry in pub_data if entry[3] == 0]


def create_dataset(entries, transform):
    dataset = MembershipDataset(transform=transform)
    for id_, img, label, member in entries:
        dataset.ids.append(id_)
        dataset.imgs.append(img)
        dataset.labels.append(label)
        dataset.membership.append(member)
    return dataset

full_dataset = create_dataset(non_member_entries, transform)
full_loader = DataLoader(full_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

def create_resnet18():
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 44)
    return model.to(device)

# Model 1 
model1 = create_resnet18()
optimizer1 = optim.SGD(model1.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
scheduler1 = CosineAnnealingLR(optimizer1, T_max=100)
criterion1 = nn.CrossEntropyLoss()

# Model 2
model2 = create_resnet18()
optimizer2 = optim.AdamW(model2.parameters(), lr=0.0005, weight_decay=5e-4)
scheduler2 = CosineAnnealingLR(optimizer2, T_max=100)
criterion2 = nn.CrossEntropyLoss()

def train_model(model, loader, optimizer, scheduler, criterion, model_name):
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for _, inputs, labels, _ in loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        

train_model(model1, full_loader, optimizer1, scheduler1, criterion1, "model1_resnet18_sgd")
train_model(model2, full_loader, optimizer2, scheduler2, criterion2, "model2_resnet18_adamw")
