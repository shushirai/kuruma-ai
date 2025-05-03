import os
import time
import torch
import torch.nn as nn
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ✅ 設定
data_dir = "dataset_split"
batch_size = 16
num_epochs = 50
lr = 1e-4
patience = 5
model_name = "resnet101"  # resnet50, resnet101, resnext50_32x4d, wide_resnet50_2
cutmix_alpha = 1.0  # CutMixの強さ（0で無効）

# ✅ デバイス自動設定
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"🔶 使用デバイス: {device}")

# ✅ 保存フォルダ作成
os.makedirs("./models", exist_ok=True)

# ✅ データ拡張と前処理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
])
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ データ読み込み
train_dataset = ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
val_dataset = ImageFolder(os.path.join(data_dir, "val"), transform=val_test_transform)
test_dataset = ImageFolder(os.path.join(data_dir, "test"), transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

num_classes = len(train_dataset.classes)
print(f"🔶 クラス数: {num_classes}, クラス名: {train_dataset.classes}")

# ✅ クラス名保存
with open("class_names.txt", "w") as f:
    for name in train_dataset.classes:
        f.write(f"{name}\n")

# ✅ モデル構築関数
def get_model(model_name, num_classes):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Unsupported model name")

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

print(f"🔶 モデル選択: {model_name}")
model = get_model(model_name, num_classes).to(device)

# ✅ 損失関数・最適化
criterion = nn.CrossEntropyLoss()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

# ✅ 学習記録用
train_losses = []
train_accuracies = []
val_accuracies = []

# ✅ CutMix関数定義
def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam

import math

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = random.randint(0, W)
    cy = random.randint(0, H)

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)

    return bbx1, bby1, bbx2, bby2


# ✅ 学習ループ（EarlyStopping & 保存）
best_acc = 0.0
early_stop_counter = 0

# 🔸 全体の学習時間計測開始
all_start = time.time()

for epoch in range(num_epochs):
    # 🔹 各エポックの処理時間計測開始
    epoch_start = time.time()

    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        if cutmix_alpha > 0:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, cutmix_alpha)
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        if cutmix_alpha > 0:
            train_correct += (lam * preds.eq(targets_a).sum().item() + (1 - lam) * preds.eq(targets_b).sum().item())
        else:
            train_correct += preds.eq(labels).sum().item()

    avg_train_loss = train_loss / len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_accuracy = val_correct / len(val_dataset)

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"🔶 Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}")

    # 🔹 各エポックの処理時間出力
    epoch_time = time.time() - epoch_start
    print(f"⏱️ Epoch {epoch+1} 所要時間: {epoch_time:.2f} 秒（{epoch_time/60:.2f} 分）")

    if val_accuracy > best_acc:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        save_path = f"./models/{model_name}_best_{timestamp}_valacc{val_accuracy:.4f}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"🔶 モデル保存: {save_path}")

        best_acc = val_accuracy
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"🔶 EarlyStoppingカウント: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"🔶 EarlyStopping発動！（{patience}回連続で精度向上なし）")
            break

# 🔸 全体の学習時間出力
all_time = time.time() - all_start
print(f"\n⏱️ 全エポックの合計学習時間: {all_time:.2f} 秒（{all_time/60:.2f} 分）")


# ✅ 学習曲線プロット＆保存
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label="Train Acc")
plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_save_path = f"./models/{model_name}_training_curve_{timestamp}.png"
plt.savefig(plot_save_path)
print(f"🔶 学習曲線保存: {plot_save_path}")
plt.show()

# ✅ 最終テスト評価（ベストモデルで）
print("\n🔶 最終テスト評価中...")
model.load_state_dict(torch.load(save_path))
model.eval()
test_correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_correct += (outputs.argmax(1) == labels).sum().item()

test_accuracy = test_correct / len(test_dataset)
print(f"\n🔶 最終テスト精度: {test_accuracy:.4f}")


