import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)


test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)


test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True
)


criterion = nn.CrossEntropyLoss()


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)
        x = x.reshape(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


@torch.no_grad()
def evaluate_pytorch_fp32(model, test_loader, warmup_batches=5, device="cuda"):
    model.eval().to(device).float()
    total = correct = 0
    run_loss = 0.0

    fwd_time_sum = 0.0
    e2e_time_sum = 0.0
    measured = 0
    bs_seen = None

    for bi, (images_cpu, labels_cpu) in enumerate(test_loader):

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        images = images_cpu.to(device, non_blocking=True)
        outputs = model(images)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        e2e_dt = t1 - t0

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        _ = model(images)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        fwd_dt = t3 - t2

        labels = labels_cpu.to(device, non_blocking=True)
        loss = criterion(outputs, labels)
        run_loss += loss.item() * labels.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if bs_seen is None:
            bs_seen = labels.size(0)

        if bi >= warmup_batches:
            fwd_time_sum += fwd_dt
            e2e_time_sum += e2e_dt
            measured += 1

    avg_loss = run_loss / total
    acc = correct / total

    if measured > 0 and bs_seen:
        fwd_t_batch = fwd_time_sum / measured
        fwd_t_img = fwd_t_batch / bs_seen
        fwd_fps = 1.0 / fwd_t_img

        e2e_t_batch = e2e_time_sum / measured
        e2e_t_img = e2e_t_batch / bs_seen
        e2e_fps = 1.0 / e2e_t_img
    else:
        fwd_t_batch = fwd_t_img = fwd_fps = float("nan")
        e2e_t_batch = e2e_t_img = e2e_fps = float("nan")

    return {
        "loss": avg_loss,
        "acc": acc,
        "fwd_batch_s": fwd_t_batch,
        "fwd_img_s": fwd_t_img,
        "fwd_fps": fwd_fps,
        "e2e_batch_s": e2e_t_batch,
        "e2e_img_s": e2e_t_img,
        "e2e_fps": e2e_fps,
    }


model = LeNet()
model.load_state_dict(torch.load("lenet_cifar10_best_pytorch.pth", map_location="cpu"))

KOL = 100

fwd_t_batch_sum = 0
fwd_t_img_sum = 0
fwd_fps_sum = 0
e2e_t_batch_sum = 0
e2e_t_img_sum = 0
e2e_fps_sum = 0
for _ in range(KOL):
    time.sleep(1)
    stats = evaluate_pytorch_fp32(model, test_loader)
    print(
        f"PyTorch FP32: "
        f"forward: {stats['fwd_batch_s']*1000:.3f} ms/batch, {stats['fwd_img_s']*1000:.4f} ms/img, {stats['fwd_fps']:.1f} FPS | "
        f"e2e: {stats['e2e_batch_s']*1000:.3f} ms/batch, {stats['e2e_img_s']*1000:.4f} ms/img, {stats['e2e_fps']:.1f} FPS"
    )
    fwd_t_batch_sum += stats["fwd_batch_s"]
    fwd_t_img_sum += stats["fwd_img_s"]
    fwd_fps_sum += stats["fwd_fps"]
    e2e_t_batch_sum += stats["e2e_batch_s"]
    e2e_t_img_sum += stats["e2e_img_s"]
    e2e_fps_sum += stats["e2e_fps"]

stats = {
    "fwd_batch_s": fwd_t_batch_sum / KOL,
    "fwd_img_s": fwd_t_img_sum / KOL,
    "fwd_fps": fwd_fps_sum / KOL,
    "e2e_batch_s": e2e_t_batch_sum / KOL,
    "e2e_img_s": e2e_t_img_sum / KOL,
    "e2e_fps": e2e_fps_sum / KOL,
}
print("Mean______________________________________________________________")
print(
    f"PyTorch FP32: "
    f"forward: {stats['fwd_batch_s']*1000:.3f} ms/batch, {stats['fwd_img_s']*1000:.4f} ms/img, {stats['fwd_fps']:.1f} FPS | "
    f"e2e: {stats['e2e_batch_s']*1000:.3f} ms/batch, {stats['e2e_img_s']*1000:.4f} ms/img, {stats['e2e_fps']:.1f} FPS"
)
