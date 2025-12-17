import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 16 * 5 * 5)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        return t


mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)

test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=200, shuffle=True, num_workers=2, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=200, shuffle=False, num_workers=2, pin_memory=True
)
model = LeNet().to(device)
torch.save(model.state_dict(), "lenet_init_state.pth")


criterion = nn.CrossEntropyLoss()
optimizer = SGD(
    model.parameters(),
    lr=0.05,
    momentum=0.9,
    weight_decay=5e-4,
)
scheduler = MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)

num_epochs = 60
best_acc = 0.0


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    val_loss, val_acc = evaluate(model, test_loader, device)
    scheduler.step()

    print(
        f"Epoch [{epoch}/{num_epochs}] "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
        f"lr={scheduler.get_last_lr()[0]:.5f}"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "lenet_cifar10_best_pytorch.pth")
        print(f"  -> new best model saved (val_acc={best_acc:.4f})")
