import os

def count_images_in_subfolders(root_dir, extensions=(".jpg", ".jpeg", ".png")):
    result = {}
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolder_path):
            image_count = sum(
                1 for file in os.listdir(subfolder_path)
                if file.lower().endswith(extensions)
            )
            result[subfolder] = image_count
    return result

def print_counts(title, counts_dict, min_threshold=1000):
    print(f"\n📂 {title}（{min_threshold}枚未満のみ表示）")
    filtered = {k: v for k, v in counts_dict.items() if v < min_threshold}
    total = sum(filtered.values())
    if not filtered:
        print("✅ すべての車種が1000枚以上あります。")
    else:
        for folder, count in sorted(filtered.items()):
            print(f"{folder}: {count} 枚")
        print(f"⚠️ 合計（不足分のみ）: {total} 枚")

if __name__ == "__main__":
    raw_folder = "dataset_raw"
    raw_counts = count_images_in_subfolders(raw_folder)
    print_counts("【RAWデータ】", raw_counts, min_threshold=1000)
