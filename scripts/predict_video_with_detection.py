import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import os
from ultralytics import YOLO

# ✅ 車種情報
car_prices = {
    "Toyota Prius": {"新車価格": "275万円〜460万円", "中古価格": "3.7万円〜569万円"},
    "Honda Fit": {"新車価格": "172万円〜250.7万円", "中古価格": "4.8万円〜290万円"},
    "Nissan Note": {"新車価格": "229.9万円〜306.4万円", "中古価格": "5万円〜295万円"},
}
class_names = list(car_prices.keys())

# ✅ モデルロード関数
def load_classifier(model_path="./models/best_model.pth", num_classes=len(class_names)):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# ✅ デバイス設定 (CUDA優先)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 使用デバイス: {device}")

# ✅ モデル読み込み
classifier = load_classifier().to(device)
detector = YOLO("yolov8n.pt")  # 軽量版YOLO

# ✅ 前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ 動画推論＋分類パイプライン
def predict_video_with_detection(input_video_path, output_video_path="output_with_detection.mp4", threshold=0.6):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("❌ 動画を開けません")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.predict(source=frame, stream=True, verbose=False)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                if int(label) in [2, 5, 7]:  # car, bus, truck
                    x1, y1, x2, y2 = map(int, box)
                    cropped_car = frame[y1:y2, x1:x2]

                    if cropped_car.size == 0:
                        continue

                    img_pil = Image.fromarray(cv2.cvtColor(cropped_car, cv2.COLOR_BGR2RGB)).convert("RGB")
                    input_tensor = transform(img_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = classifier(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                        top_prob, top_idx = torch.topk(probs, k=1)

                        confidence = top_prob.item()
                        car_label = class_names[top_idx.item()]

                    # 描画テキスト
                    if confidence < threshold:
                        text = f"判定不能 ({confidence:.2f})"
                    else:
                        text = f"{car_label} ({confidence:.2f})"

                    # バウンディングボックス＋ラベル
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"✅ {frame_idx} フレーム処理中...")

    cap.release()
    out.release()
    print(f"🎯 推論完了: {output_video_path}")

# ✅ 使用例（有効にして実行）
predict_video_with_detection("test1.mp4", "output_with_detection.mp4")
