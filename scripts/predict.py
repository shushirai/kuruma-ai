import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# ✅ 🔧 モデルファイルとクラス名ファイルのパスを定義
MODEL_PATH = "./models/resnet101_best_20250502_085313_valacc0.9728.pth"
CLASS_NAMES_PATH = "class_names.txt"

# ✅ クラス名を自動読み込み
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

# ✅ 車種価格情報（登録されてないものもある想定で）
car_prices = {
    "Toyota Prius": {"新車価格": "275万円〜460万円", "中古価格": "3.7万円〜569万円"},
    "Honda Fit": {"新車価格": "172万円〜250.7万円", "中古価格": "4.8万円〜290万円"},
    "Nissan Note": {"新車価格": "229.9万円〜306.4万円", "中古価格": "5万円〜295万円"},
}

# ✅ モデル読み込み（resnet101対応）
def load_model(model_path=MODEL_PATH, num_classes=len(class_names)):
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
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

# ✅ 分類関数（カードUI）
def classify_top3_cards(image, threshold=0.6):
    img = Image.fromarray(image).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, k=3)

        html_cards = ""
        for i in range(3):
            confidence = top_probs[i].item()
            label = class_names[top_idxs[i].item()]
            price_info = car_prices.get(label, {"新車価格": "不明", "中古価格": "不明"})
            low_confidence = confidence < threshold

            card_color = "#721c24" if low_confidence else "#155724"
            text_color = "#ffffff"
            warning_text = "<p style='color:#ffcccc;'>⚠️ 信頼度が低いため参考程度</p>" if low_confidence else ""

            html_cards += f"""
            <div style='border:1px solid #ccc; border-radius:8px; margin-bottom:10px; padding:10px;
                        background-color:{card_color}; color:{text_color};'>
                <h3>🚗 候補{i+1}: {label}</h3>
                <p><strong>信頼度:</strong> {confidence:.2f}</p>
                <p><strong>新車価格:</strong> {price_info['新車価格']}</p>
                <p><strong>中古価格:</strong> {price_info['中古価格']}</p>
                {warning_text}
            </div>
            """

        return gr.HTML(value=html_cards)

# ✅ Gradio UI
interface = gr.Interface(
    fn=classify_top3_cards,
    inputs=[
        gr.Image(type="numpy", label="車の画像をアップロード"),
        gr.Slider(0, 1, value=0.6, step=0.01, label="信頼度のしきい値"),
    ],
    outputs=gr.HTML(label="上位3候補（カード形式表示）"),
    title="クルマチェッカーAI - カードUI（高コントラスト）",
    description="車種ごとの分類結果をカード形式で見やすく表示します。",
)

interface.launch(share=False)
