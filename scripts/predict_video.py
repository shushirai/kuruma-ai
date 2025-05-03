import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import os

# ✅ 車種名リスト
car_prices = {
    "Toyota Prius": {"新車価格": "275万円〜460万円", "中古価格": "3.7万円〜569万円"},
    "Honda Fit": {"新車価格": "172万円〜250.7万円", "中古価格": "4.8万円〜290万円"},
    "Nissan Note": {"新車価格": "229.9万円〜306.4万円", "中古価格": "5万円〜295万円"},
}
class_names = list(car_prices.keys())

# ✅ モデル読み込み
def load_model(model_path="./models/best_model.pth", num_classes=len(class_names)):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ✅ 前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ 動画推論＋保存
def predict_video(input_video_path, output_video_path="output_predicted.mp4", threshold=0.6):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("❌ 動画ファイルを開けませんでした")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(img_pil).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top_prob, top_idx = torch.topk(probs, k=1)

            confidence = top_prob.item()
            label = class_names[top_idx.item()]

        # 🔖 判定ラベル
        if confidence < threshold:
            text = f"判定不能 ({confidence:.2f})"
        else:
            text = f"{label} ({confidence:.2f})"

        # ✍️ フレームに書き込み
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"✅ {frame_count} フレーム処理中...")

    cap.release()
    out.release()
    print(f"🎥 保存完了: {output_video_path}")

# ✅ 使用例（あとでコマンドから呼び出してもOK）
predict_video("test1.mp4", output_video_path="your_output_video.mp4")
