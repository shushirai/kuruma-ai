import os
import shutil
import cv2
import time
from ultralytics import YOLO

# ⏱️ 処理時間計測開始
start_time = time.time()

# モデル選択
model = YOLO("yolov8m.pt")

# フォルダ設定
input_root = "dataset_raw"
output_root = "dataset_clean"
rejected_root = "dataset_rejected"  # ❗ 除外画像の保存先

# 検出条件
target_class = "car"
min_area_ratio = 0.05

# 統計
total_images = 0
kept_images = 0
rejected_images = []

# 各クラス（車種）フォルダ処理
for class_folder in os.listdir(input_root):
    input_dir = os.path.join(input_root, class_folder)
    output_dir = os.path.join(output_root, class_folder)
    rejected_dir = os.path.join(rejected_root, class_folder)

    if not os.path.isdir(input_dir):
        continue

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    print(f"\n🚗 処理中: {class_folder}")

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        total_images += 1
        filepath = os.path.join(input_dir, filename)

        try:
            img = cv2.imread(filepath)
            if img is None:
                raise ValueError("画像読み込み失敗")

            h, w, _ = img.shape
            img_area = h * w

            results = model(img, stream=True)
            keep = False

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    if cls_name != target_class:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_area = (x2 - x1) * (y2 - y1)
                    area_ratio = box_area / img_area
                    if area_ratio >= min_area_ratio:
                        keep = True
                        break
                if keep:
                    break

            if keep:
                kept_images += 1
                shutil.copy(filepath, os.path.join(output_dir, filename))
                #print(f"✅ {filename}")
            else:
                rejected_images.append(os.path.join(class_folder, filename))
                shutil.copy(filepath, os.path.join(rejected_dir, filename))
                print(f"🗑 {filename} → dataset_rejected に移動")

        except Exception as e:
            print(f"⚠️ スキップ: {filename} - 理由: {e}")
            rejected_images.append(os.path.join(class_folder, filename))
            shutil.copy(filepath, os.path.join(rejected_dir, filename))

# ⏱️ 統計出力
end_time = time.time()
elapsed = end_time - start_time
avg_time = elapsed / total_images if total_images else 0

print("\n📊 処理統計")
print(f"🖼️ 総画像数: {total_images}")
print(f"✅ 残した画像数: {kept_images}")
print(f"🗑 除外された画像数: {len(rejected_images)}")
print(f"⏱️ 総処理時間: {elapsed:.2f} 秒")
print(f"⚡ 平均処理時間: {avg_time:.3f} 秒/画像")

# 除外画像一覧をテキストに保存
with open("rejected_images.txt", "w", encoding="utf-8") as f:
    for path in rejected_images:
        f.write(path + "\n")

# 除外画像一覧の表示
if rejected_images:
    print("\n📂 除外された画像一覧（最大10件表示）")
    for path in rejected_images[:10]:
        print(path)
    if len(rejected_images) > 10:
        print(f"...（省略）全{len(rejected_images)}件は rejected_images.txt に保存済み")
