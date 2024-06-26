import pytest
import torch
from torch import nn
from fadam import FAdam
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from efficientnet_pytorch import EfficientNet
from torchvision import datasets, transforms
import os
import numpy as np
import random

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class EFNet_L2(nn.Module):
    def __init__(self, num_classes):
        super(EFNet_L2, self).__init__()
        self.effnet = EfficientNet.from_pretrained('efficientnet-b2')
        self.effnet._fc = nn.Linear(self.effnet._fc.in_features, num_classes)

    def forward(self, x):
        x = self.effnet(x)
        return x

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends
set_seed(40)
@pytest.fixture
def model():
    model = EFNet_L2(num_classes=2).to(device)
    return model

@pytest.fixture
def data_loaders():
    train_data, train_labels = torch.load('train_data.pt')
    test_data, test_labels = torch.load('test_data.pt')

    print(f'Train data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')

    if train_data.shape[1] == 1:
        train_data = train_data.repeat(1, 3, 1, 1)
    if test_data.shape[1] == 1:
        test_data = test_data.repeat(1, 3, 1, 1)

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


@pytest.fixture
def optimizer_and_criterion(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion


#train
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


#test
def test(model, data_loaders, optimizer_and_criterion):
    model.eval()
    train_loader, test_loader = data_loaders
    optimizer, criterion = optimizer_and_criterion
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 배치 손실 합산
            pred = output.argmax(dim=1, keepdim=True)  # 가장 높은 로그 확률 선택
            correct += pred.eq(target.view_as(pred)).sum().item()
            print(f'pred: {pred} target: {target.view_as(pred)}')  # 출력 추가

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    model_save_path = "/Users/hongseunghyuk/PycharmProjects/practice 1/data/model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
# 테스트 이외의 용도로 사용할 함수 정의
def load_data():
    train_data, train_labels = torch.load('train_data.pt')
    test_data, test_labels = torch.load('test_data.pt')

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

def create_model_and_optimizer():
    model = EFNet_L2(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


if __name__ == "__main__":
    model, optimizer, criterion = create_model_and_optimizer()
    train_loader, test_loader = load_data()
    for epoch in range(1, 11):
        print(train_loader)
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, (train_loader, test_loader), (optimizer, criterion))