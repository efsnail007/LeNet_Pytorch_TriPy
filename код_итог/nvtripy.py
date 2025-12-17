import time

import nvtripy as tp
import torch
import torchvision
import torchvision.transforms as transforms

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
    dataset=test_set, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True
)


class TripyLeNet(tp.Module):
    def __init__(self, dtype=tp.float32):
        super().__init__()

        self.conv1 = tp.Conv(
            in_channels=3,
            out_channels=6,
            kernel_dims=(5, 5),
            dtype=dtype,
        )
        self.conv2 = tp.Conv(
            in_channels=6,
            out_channels=16,
            kernel_dims=(5, 5),
            dtype=dtype,
        )

        self.fc1 = tp.Linear(16 * 5 * 5, 120, dtype=dtype)
        self.fc2 = tp.Linear(120, 84, dtype=dtype)
        self.out = tp.Linear(84, 10, dtype=dtype)

    def forward(self, x):

        x = self.conv1(x)
        x = tp.relu(x)
        x = tp.maxpool(x, kernel_dims=(2, 2), stride=(2, 2))

        x = self.conv2(x)
        x = tp.relu(x)
        x = tp.maxpool(x, kernel_dims=(2, 2), stride=(2, 2))

        x = tp.reshape(x, (x.shape[0], -1))

        x = self.fc1(x)
        x = tp.relu(x)

        x = self.fc2(x)
        x = tp.relu(x)

        x = self.out(x)
        return x


def convert_state_torch_to_tripy(torch_state):
    tripy_state = {}
    for name, param in torch_state.items():
        np_value = param.detach().cpu().numpy().astype("float32")
        tripy_state[name] = tp.Tensor(np_value)
    return tripy_state


def to_tripy_tensor(x_torch):
    list_x = x_torch.tolist()
    return tp.Tensor(list_x)


def to_torch_tensor(x_tripy):
    list_x = x_tripy.tolist()
    return torch.Tensor(list_x)


torch_state = torch.load("lenet_cifar10_best_pytorch.pth")
tripy_state = convert_state_torch_to_tripy(torch_state)

tripy_model = TripyLeNet()
tripy_model.load_state_dict(tripy_state)

inp_info = tp.InputInfo(shape=(1000, 3, 32, 32), dtype=tp.float32)
fast_model = tp.compile(tripy_model, args=[inp_info])


@torch.no_grad()
def evaluate_tripy_time(exe, test_loader, warmup_batches=5, device="cuda", fp16=False):

    if device != "cuda":
        raise ValueError("Для бенча ожидается device='cuda'.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна.")

    fwd_time_sum = 0.0
    e2e_time_sum = 0.0
    measured = 0
    bs_seen = None

    for bi, (images_cpu, _) in enumerate(test_loader):

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        images = images_cpu.to(device, non_blocking=True)
        if fp16:
            images = images.half()
        images = images.contiguous()

        images_tp = tp.Tensor(images).eval()
        _ = exe(images_tp).eval()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        e2e_dt = t1 - t0

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        _ = exe(images_tp).eval()

        torch.cuda.synchronize()
        t3 = time.perf_counter()
        fwd_dt = t3 - t2

        if bs_seen is None:
            bs_seen = int(images.shape[0])

        if bi >= warmup_batches:
            fwd_time_sum += fwd_dt
            e2e_time_sum += e2e_dt
            measured += 1

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
        "fwd_batch_s": fwd_t_batch,
        "fwd_img_s": fwd_t_img,
        "fwd_fps": fwd_fps,
        "e2e_batch_s": e2e_t_batch,
        "e2e_img_s": e2e_t_img,
        "e2e_fps": e2e_fps,
    }


KOL = 100

fwd_t_batch_sum = 0
fwd_t_img_sum = 0
fwd_fps_sum = 0
e2e_t_batch_sum = 0
e2e_t_img_sum = 0
e2e_fps_sum = 0
for _ in range(KOL):
    time.sleep(1)
    stats = evaluate_tripy_time(fast_model, test_loader, warmup_batches=5, fp16=False)
    print(
        f"TriPy FP32: "
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
    f"TriPy FP32: "
    f"forward: {stats['fwd_batch_s']*1000:.3f} ms/batch, {stats['fwd_img_s']*1000:.4f} ms/img, {stats['fwd_fps']:.1f} FPS | "
    f"e2e: {stats['e2e_batch_s']*1000:.3f} ms/batch, {stats['e2e_img_s']*1000:.4f} ms/img, {stats['e2e_fps']:.1f} FPS"
)
