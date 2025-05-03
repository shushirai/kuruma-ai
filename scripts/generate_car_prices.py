import json

# class_names.txt を読み込み
with open("class_names.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f]

# 空の価格情報テンプレートを作成
car_prices_template = {
    name: {"新車価格": "", "中古価格": ""} for name in class_names
}

# JSONとして保存
with open("car_prices.json", "w", encoding="utf-8") as f:
    json.dump(car_prices_template, f, ensure_ascii=False, indent=2)

print("✅ car_prices.json を作成しました")
