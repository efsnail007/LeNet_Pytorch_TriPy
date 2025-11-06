import nvtripy as tp

class TripyLeNet(tp.Module):
    def __init__(self, dtype=tp.float32):
        super().__init__()

        # те же размеры каналов и ядер, что и в PyTorch-версии
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
        # вход: (N, 3, 32, 32)
        x = self.conv1(x)
        x = tp.relu(x)
        x = tp.maxpool(x, kernel_dims=(2, 2), stride=(2, 2))

        x = self.conv2(x)
        x = tp.relu(x)
        x = tp.maxpool(x, kernel_dims=(2, 2), stride=(2, 2))

        # выпрямление фичей: (N, 16*5*5)
        x = tp.reshape(x, (x.shape[0], -1))

        x = self.fc1(x)
        x = tp.relu(x)

        x = self.fc2(x)
        x = tp.relu(x)

        x = self.out(x)
        return x