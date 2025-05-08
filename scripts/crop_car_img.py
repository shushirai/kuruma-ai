import os
from pathlib import Path
import cv2
from ultralytics import YOLO
import pandas as pd

# パラメータ設定
CONF_THRESHOLD = 0.5       # 信頼度の閾値
MIN_AREA_RATIO = 0.1       # 面積の最小割合（画像全体の面積に対して）
CENTER_WEIGHT = 2.0        # 中心距離の重み

# 入出力ディレクトリ
input_root = "../data/dataset_raw"
output_root = "../data/dataset_cropped"
discard_root = "../data/dataset_discarded"

os.makedirs(output_root, exist_ok=True)
os.makedirs(discard_root, exist_ok=True)

# YOLOモデル読み込み
model = YOLO("../models/yolov8l.pt")

# ログ保存用リスト
log = []

def select_best_box(boxes, img_w, img_h):
    img_area = img_w * img_h
    min_area = img_area * MIN_AREA_RATIO
    img_cx, img_cy = img_w / 2, img_h / 2

    best_score = -1
    best_box = None

    for box in boxes:
        x1, y1, x2, y2, conf, class_id = box.tolist()
        if conf < CONF_THRESHOLD or int(class_id) != 2:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5
        score = area - CENTER_WEIGHT * dist
        if score > best_score:
            best_score = score
            best_box = (int(x1), int(y1), int(x2), int(y2))
    return best_box

# サブフォルダ（車種）ごとに処理
for car_type in sorted(os.listdir(input_root)):
    input_dir = os.path.join(input_root, car_type)
    output_dir = os.path.join(output_root, car_type)
    discard_dir = os.path.join(discard_root, car_type)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(discard_dir, exist_ok=True)

    total_images = 0
    saved_images = 0

    for img_path in Path(input_dir).glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        total_images += 1
        results = model(img)[0]
        best_box = select_best_box(results.boxes.data, w, h)

        if best_box:
            x1, y1, x2, y2 = best_box
            cropped = img[y1:y2, x1:x2]
            if cropped.size > 0:
                save_path = os.path.join(output_dir, img_path.name)
                cv2.imwrite(save_path, cropped)
                saved_images += 1
        else:
            # 保存対象外 → discardフォルダに保存
            discard_path = os.path.join(discard_dir, img_path.name)
            cv2.imwrite(discard_path, img)

    log.append({
        "car_type": car_type,
        "total_images": total_images,
        "saved_crops": saved_images,
        "discarded": total_images - saved_images
    })

# ログ出力（コンソール & DataFrame表示）
df = pd.DataFrame(log)
print("\n📊 処理結果サマリー")
print(df)
