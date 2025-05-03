import os
import shutil
import random

# ✅ 一括設定
# ✅ このスクリプトファイル（例：split_dataset.py）の絶対パスを基準に
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "dataset_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "dataset_split")

def split_dataset(
    input_dir,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    random.seed(seed)

    # ✅ 出力ディレクトリを削除して初期化
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    class_count = 0  # ← カウント用変数

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        class_count += 1  # ← 車種数をカウント

        # 全画像取得＆シャッフル
        images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }

        for split, split_images in splits.items():
            split_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy(src, dst)

        print(f"✅ {class_name}: {n} 枚 → train:{n_train}, val:{n_val}, test:{n_test}")

    print(f"\n🚘 全体の車種数（クラス数）: {class_count} 種類")
    
    
# 実行
split_dataset(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR
)