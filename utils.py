import re
from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"^Epoch\s*\[(\d+)/(\d+)\]\s*"
    r"train_loss=([0-9]*\.?[0-9]+)\s+"
    r"train_acc=([0-9]*\.?[0-9]+)\s+"
    r"val_loss=([0-9]*\.?[0-9]+)\s+"
    r"val_acc=([0-9]*\.?[0-9]+)\s+"
    r"lr=([0-9]*\.?[0-9]+)\s*$"
)
import matplotlib.pyplot as plt

# Глобальные настройки размера шрифтов (поставь свои значения)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,

    "font.size": 16,        # базовый размер
    "axes.titlesize": 18,   # заголовок графика
    "axes.labelsize": 18,   # подписи осей
    "xtick.labelsize": 14,  # подписи делений по X
    "ytick.labelsize": 14,  # подписи делений по Y
    "legend.fontsize": 18,  # легенда
})
plt.legend(loc="best", frameon=True)

@dataclass
class History:
    epoch: List[int]
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_acc: List[float]
    lr: List[float]

def parse_training_log(text: str) -> History:
    epoch, tr_l, tr_a, va_l, va_a, lr = [], [], [], [], [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("Epoch"):
            continue
        m = LINE_RE.match(line)
        if not m:
            # если формат внезапно поменяется, не падаем молча
            raise ValueError(f"Не смог распарсить строку:\n{line}")
        e = int(m.group(1))
        epoch.append(e)
        tr_l.append(float(m.group(3)))
        tr_a.append(float(m.group(4)))
        va_l.append(float(m.group(5)))
        va_a.append(float(m.group(6)))
        lr.append(float(m.group(7)))
    if not epoch:
        raise ValueError("Не найдено ни одной строки Epoch [...]. Проверь текст лога.")
    return History(epoch, tr_l, tr_a, va_l, va_a, lr)

def plot_history(h, show_lr: bool = False) -> None:
    # Чтобы кириллица не превращалась в квадратики
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    # 1) Loss
    plt.figure()
    plt.plot(h.epoch, h.train_loss, label="Потери (обучение)")
    plt.plot(h.epoch, h.val_loss, label="Потери (валидация)")
    plt.xlabel("Эпоха")
    plt.ylabel("Значение функции потерь")
    plt.title("Изменение функции потерь по эпохам")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 2) Accuracy
    plt.figure()
    plt.plot(h.epoch, h.train_acc, label="Точность (обучение)")
    plt.plot(h.epoch, h.val_acc, label="Точность (валидация)")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность (доля верных ответов)")
    plt.title("Изменение точности по эпохам")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 3) LR (опционально)
    if show_lr:
        plt.figure()
        plt.plot(h.epoch, h.lr, label="Скорость обучения (LR)")
        plt.xlabel("Эпоха")
        plt.ylabel("Скорость обучения")
        plt.title("Изменение скорости обучения по эпохам")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    best_i = max(range(len(h.val_acc)), key=lambda i: h.val_acc[i])
    print(f"Лучшая точность на валидации = {h.val_acc[best_i]:.4f} (эпоха {h.epoch[best_i]})")
    print(f"Финал: val_acc = {h.val_acc[-1]:.4f} | val_loss = {h.val_loss[-1]:.4f} | lr = {h.lr[-1]:.5f}")

    plt.show()


if __name__ == "__main__":
    log_text = r"""
Epoch [1/60] train_loss=1.8535 train_acc=0.3092 val_loss=1.5341 val_acc=0.4452 lr=0.05000
  -> new best model saved (val_acc=0.4452)
Epoch [2/60] train_loss=1.5643 train_acc=0.4292 val_loss=1.4005 val_acc=0.4923 lr=0.05000
  -> new best model saved (val_acc=0.4923)
Epoch [3/60] train_loss=1.4619 train_acc=0.4746 val_loss=1.3780 val_acc=0.5003 lr=0.05000
  -> new best model saved (val_acc=0.5003)
Epoch [4/60] train_loss=1.4076 train_acc=0.4950 val_loss=1.2948 val_acc=0.5418 lr=0.05000
  -> new best model saved (val_acc=0.5418)
Epoch [5/60] train_loss=1.3576 train_acc=0.5156 val_loss=1.2279 val_acc=0.5722 lr=0.05000
  -> new best model saved (val_acc=0.5722)
Epoch [6/60] train_loss=1.3263 train_acc=0.5289 val_loss=1.2295 val_acc=0.5750 lr=0.05000
  -> new best model saved (val_acc=0.5750)
Epoch [7/60] train_loss=1.2965 train_acc=0.5410 val_loss=1.2435 val_acc=0.5602 lr=0.05000
Epoch [8/60] train_loss=1.2679 train_acc=0.5525 val_loss=1.1958 val_acc=0.5853 lr=0.05000
  -> new best model saved (val_acc=0.5853)
Epoch [9/60] train_loss=1.2646 train_acc=0.5538 val_loss=1.1468 val_acc=0.6041 lr=0.05000
  -> new best model saved (val_acc=0.6041)
Epoch [10/60] train_loss=1.2429 train_acc=0.5610 val_loss=1.1549 val_acc=0.5989 lr=0.05000
Epoch [11/60] train_loss=1.2243 train_acc=0.5672 val_loss=1.1338 val_acc=0.6047 lr=0.05000
  -> new best model saved (val_acc=0.6047)
Epoch [12/60] train_loss=1.2115 train_acc=0.5753 val_loss=1.0874 val_acc=0.6236 lr=0.05000
  -> new best model saved (val_acc=0.6236)
Epoch [13/60] train_loss=1.1991 train_acc=0.5791 val_loss=1.1139 val_acc=0.6181 lr=0.05000
Epoch [14/60] train_loss=1.2068 train_acc=0.5780 val_loss=1.1398 val_acc=0.6117 lr=0.05000
Epoch [15/60] train_loss=1.1915 train_acc=0.5835 val_loss=1.1063 val_acc=0.6285 lr=0.05000
  -> new best model saved (val_acc=0.6285)
Epoch [16/60] train_loss=1.1827 train_acc=0.5881 val_loss=1.1191 val_acc=0.6102 lr=0.05000
Epoch [17/60] train_loss=1.1663 train_acc=0.5917 val_loss=1.1091 val_acc=0.6176 lr=0.05000
Epoch [18/60] train_loss=1.1690 train_acc=0.5922 val_loss=1.0614 val_acc=0.6336 lr=0.05000
  -> new best model saved (val_acc=0.6336)
Epoch [19/60] train_loss=1.1656 train_acc=0.5940 val_loss=1.1286 val_acc=0.6113 lr=0.05000
Epoch [20/60] train_loss=1.1660 train_acc=0.5948 val_loss=1.0930 val_acc=0.6197 lr=0.05000
Epoch [21/60] train_loss=1.1567 train_acc=0.5991 val_loss=1.0507 val_acc=0.6344 lr=0.05000
  -> new best model saved (val_acc=0.6344)
Epoch [22/60] train_loss=1.1354 train_acc=0.6051 val_loss=1.0365 val_acc=0.6390 lr=0.05000
  -> new best model saved (val_acc=0.6390)
Epoch [23/60] train_loss=1.1329 train_acc=0.6049 val_loss=1.0768 val_acc=0.6274 lr=0.05000
Epoch [24/60] train_loss=1.1406 train_acc=0.6021 val_loss=1.0229 val_acc=0.6514 lr=0.05000
  -> new best model saved (val_acc=0.6514)
Epoch [25/60] train_loss=1.1351 train_acc=0.6026 val_loss=1.0918 val_acc=0.6202 lr=0.05000
Epoch [26/60] train_loss=1.1322 train_acc=0.6063 val_loss=1.0249 val_acc=0.6415 lr=0.05000
Epoch [27/60] train_loss=1.1353 train_acc=0.6056 val_loss=1.0421 val_acc=0.6411 lr=0.05000
Epoch [28/60] train_loss=1.1252 train_acc=0.6084 val_loss=1.0299 val_acc=0.6397 lr=0.05000
Epoch [29/60] train_loss=1.1187 train_acc=0.6112 val_loss=1.0810 val_acc=0.6306 lr=0.05000
Epoch [30/60] train_loss=1.1296 train_acc=0.6062 val_loss=1.1148 val_acc=0.6151 lr=0.00500
Epoch [31/60] train_loss=0.9823 train_acc=0.6571 val_loss=0.8991 val_acc=0.6916 lr=0.00500
  -> new best model saved (val_acc=0.6916)
Epoch [32/60] train_loss=0.9437 train_acc=0.6710 val_loss=0.8760 val_acc=0.6988 lr=0.00500
  -> new best model saved (val_acc=0.6988)
Epoch [33/60] train_loss=0.9201 train_acc=0.6805 val_loss=0.8669 val_acc=0.7033 lr=0.00500
  -> new best model saved (val_acc=0.7033)
Epoch [34/60] train_loss=0.9159 train_acc=0.6798 val_loss=0.8622 val_acc=0.7042 lr=0.00500
  -> new best model saved (val_acc=0.7042)
Epoch [35/60] train_loss=0.9025 train_acc=0.6847 val_loss=0.8589 val_acc=0.7047 lr=0.00500
  -> new best model saved (val_acc=0.7047)
Epoch [36/60] train_loss=0.9005 train_acc=0.6851 val_loss=0.8524 val_acc=0.7032 lr=0.00500
Epoch [37/60] train_loss=0.8857 train_acc=0.6890 val_loss=0.8522 val_acc=0.7073 lr=0.00500
  -> new best model saved (val_acc=0.7073)
Epoch [38/60] train_loss=0.8926 train_acc=0.6874 val_loss=0.8431 val_acc=0.7098 lr=0.00500
  -> new best model saved (val_acc=0.7098)
Epoch [39/60] train_loss=0.8885 train_acc=0.6905 val_loss=0.8333 val_acc=0.7154 lr=0.00500
  -> new best model saved (val_acc=0.7154)
Epoch [40/60] train_loss=0.8793 train_acc=0.6912 val_loss=0.8426 val_acc=0.7113 lr=0.00500
Epoch [41/60] train_loss=0.8754 train_acc=0.6934 val_loss=0.8263 val_acc=0.7148 lr=0.00500
Epoch [42/60] train_loss=0.8688 train_acc=0.6986 val_loss=0.8294 val_acc=0.7173 lr=0.00500
  -> new best model saved (val_acc=0.7173)
Epoch [43/60] train_loss=0.8636 train_acc=0.6980 val_loss=0.8272 val_acc=0.7184 lr=0.00500
  -> new best model saved (val_acc=0.7184)
Epoch [44/60] train_loss=0.8608 train_acc=0.6983 val_loss=0.8198 val_acc=0.7179 lr=0.00500
Epoch [45/60] train_loss=0.8596 train_acc=0.6997 val_loss=0.8224 val_acc=0.7185 lr=0.00050
  -> new best model saved (val_acc=0.7185)
Epoch [46/60] train_loss=0.8432 train_acc=0.7048 val_loss=0.8090 val_acc=0.7224 lr=0.00050
  -> new best model saved (val_acc=0.7224)
Epoch [47/60] train_loss=0.8394 train_acc=0.7084 val_loss=0.8061 val_acc=0.7238 lr=0.00050
  -> new best model saved (val_acc=0.7238)
Epoch [48/60] train_loss=0.8356 train_acc=0.7068 val_loss=0.8052 val_acc=0.7235 lr=0.00050
Epoch [49/60] train_loss=0.8370 train_acc=0.7083 val_loss=0.8037 val_acc=0.7224 lr=0.00050
Epoch [50/60] train_loss=0.8327 train_acc=0.7095 val_loss=0.8050 val_acc=0.7237 lr=0.00050
Epoch [51/60] train_loss=0.8319 train_acc=0.7069 val_loss=0.8029 val_acc=0.7242 lr=0.00050
  -> new best model saved (val_acc=0.7242)
Epoch [52/60] train_loss=0.8343 train_acc=0.7081 val_loss=0.8040 val_acc=0.7246 lr=0.00050
  -> new best model saved (val_acc=0.7246)
Epoch [53/60] train_loss=0.8309 train_acc=0.7093 val_loss=0.8006 val_acc=0.7248 lr=0.00050
  -> new best model saved (val_acc=0.7248)
Epoch [54/60] train_loss=0.8241 train_acc=0.7129 val_loss=0.8000 val_acc=0.7250 lr=0.00050
  -> new best model saved (val_acc=0.7250)
Epoch [55/60] train_loss=0.8339 train_acc=0.7084 val_loss=0.8020 val_acc=0.7252 lr=0.00050
  -> new best model saved (val_acc=0.7252)
Epoch [56/60] train_loss=0.8319 train_acc=0.7104 val_loss=0.8033 val_acc=0.7252 lr=0.00050
Epoch [57/60] train_loss=0.8267 train_acc=0.7118 val_loss=0.7993 val_acc=0.7250 lr=0.00050
Epoch [58/60] train_loss=0.8328 train_acc=0.7079 val_loss=0.7994 val_acc=0.7270 lr=0.00050
  -> new best model saved (val_acc=0.7270)
Epoch [59/60] train_loss=0.8249 train_acc=0.7099 val_loss=0.7997 val_acc=0.7233 lr=0.00050
Epoch [60/60] train_loss=0.8262 train_acc=0.7109 val_loss=0.8010 val_acc=0.7256 lr=0.00050
""".strip("\n")

    hist = parse_training_log(log_text)
    plot_history(hist, show_lr=False)  # поставь True, если нужен график lr
