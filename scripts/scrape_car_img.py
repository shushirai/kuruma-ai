import os
import time
import urllib.request
import random
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ProcessPoolExecutor

# ✅ 並列処理を有効にするか（True or False）
USE_PARALLEL = True
MAX_WORKERS = min(4, os.cpu_count())  # 並列数の上限

# ✅ 保存先（スクリプト位置を基準にした絶対パスに変換）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "..", "data", "dataset_raw")
print(SAVE_DIR)

# ✅ 車種リスト
car_list = [

    # 🚗 現在の20車種（2025/05/01時点）
    {"jp_name": "トヨタ アルファード", "en_name": "toyota_alphard", "keyword": "アルファード", "target_images": 1000},
    {"jp_name": "トヨタ ルーミー", "en_name": "toyota_roomy", "keyword": "ルーミー", "target_images": 1000},
    {"jp_name": "トヨタ プリウス", "en_name": "toyota_prius", "keyword": "プリウス", "target_images": 1000},
    {"jp_name": "トヨタ アクア", "en_name": "toyota_aqua", "keyword": "アクア", "target_images": 1000},

    {"jp_name": "日産 エクストレイル", "en_name": "nissan_xtrail", "keyword": "エクストレイル", "target_images": 1000},
    {"jp_name": "日産 ノート", "en_name": "nissan_note", "keyword": "ノート", "target_images": 1000},
    {"jp_name": "日産 セレナ", "en_name": "nissan_serena", "keyword": "セレナ", "target_images": 1000},

    {"jp_name": "ホンダ N-BOX", "en_name": "honda_nbox", "keyword": "N-BOX", "target_images": 1000},
    {"jp_name": "ホンダ ヴェゼル", "en_name": "honda_vezel", "keyword": "ヴェゼル", "target_images": 1000},
    {"jp_name": "ホンダ フィット", "en_name": "honda_fit", "keyword": "フィット", "target_images": 1000},
    {"jp_name": "ホンダ フリード", "en_name": "honda_freed", "keyword": "フリード", "target_images": 1000},

    {"jp_name": "スズキ ハスラー", "en_name": "suzuki_hustler", "keyword": "ハスラー", "target_images": 1000},
    {"jp_name": "スズキ スイフト", "en_name": "suzuki_swift", "keyword": "スイフト", "target_images": 1000},

    {"jp_name": "ダイハツ タント", "en_name": "daihatsu_tanto", "keyword": "タント", "target_images": 1000},
    {"jp_name": "ダイハツ ムーヴ", "en_name": "daihatsu_move", "keyword": "ムーヴ", "target_images": 1000},

    {"jp_name": "マツダ2", "en_name": "mazda2", "keyword": "マツダ2", "target_images": 1000},
    {"jp_name": "マツダ3", "en_name": "mazda3", "keyword": "マツダ3", "target_images": 1000},
    {"jp_name": "マツダ CX-5", "en_name": "mazda_cx5", "keyword": "CX-5", "target_images": 1000},

    {"jp_name": "スバル フォレスター", "en_name": "subaru_forester", "keyword": "フォレスター", "target_images": 1000},
    {"jp_name": "スバル インプレッサ", "en_name": "subaru_impreza", "keyword": "インプレッサ", "target_images": 1000},

    # 🔼 ここまで既存20車種
    # 🔽 拡張おすすめ20車種

    {"jp_name": "トヨタ ヤリス", "en_name": "toyota_yaris", "keyword": "ヤリス", "target_images": 1000},
    {"jp_name": "トヨタ シエンタ", "en_name": "toyota_sienta", "keyword": "シエンタ", "target_images": 1000},
    {"jp_name": "トヨタ ハリアー", "en_name": "toyota_harrier", "keyword": "ハリアー", "target_images": 1000},
    {"jp_name": "トヨタ カローラ", "en_name": "toyota_corolla", "keyword": "カローラ", "target_images": 1000},
    {"jp_name": "トヨタ ライズ", "en_name": "toyota_raize", "keyword": "ライズ", "target_images": 1000},

    {"jp_name": "日産 キックス", "en_name": "nissan_kicks", "keyword": "キックス", "target_images": 1000},
    {"jp_name": "日産 デイズ", "en_name": "nissan_dayz", "keyword": "デイズ", "target_images": 1000},
    {"jp_name": "日産 マーチ", "en_name": "nissan_march", "keyword": "マーチ", "target_images": 1000},

    {"jp_name": "ホンダ シビック", "en_name": "honda_civic", "keyword": "シビック", "target_images": 1000},
    {"jp_name": "ホンダ ステップワゴン", "en_name": "honda_stepwgn", "keyword": "ステップワゴン", "target_images": 1000},

    {"jp_name": "スズキ ワゴンR", "en_name": "suzuki_wagonr", "keyword": "ワゴンR", "target_images": 1000},
    {"jp_name": "スズキ アルト", "en_name": "suzuki_alto", "keyword": "アルト", "target_images": 1000},
    {"jp_name": "スズキ ジムニー", "en_name": "suzuki_jimny", "keyword": "ジムニー", "target_images": 1000},

    {"jp_name": "ダイハツ ムーヴキャンバス", "en_name": "daihatsu_move_canvas", "keyword": "ムーヴキャンバス", "target_images": 1000},
    {"jp_name": "ダイハツ タフト", "en_name": "daihatsu_taft", "keyword": "タフト", "target_images": 1000},

    {"jp_name": "マツダ CX-3", "en_name": "mazda_cx3", "keyword": "CX-3", "target_images": 1000},
    {"jp_name": "マツダ CX-30", "en_name": "mazda_cx30", "keyword": "CX-30", "target_images": 1000},

    {"jp_name": "スバル レヴォーグ", "en_name": "subaru_levorg", "keyword": "レヴォーグ", "target_images": 1000},

    {"jp_name": "レクサス RX", "en_name": "lexus_rx", "keyword": "レクサスRX", "target_images": 1000},
    {"jp_name": "レクサス NX", "en_name": "lexus_nx", "keyword": "レクサスNX", "target_images": 1000},
    
    
    # 🔽 追加10車種（合計50に到達）
    {"jp_name": "スズキ スペーシア", "en_name": "suzuki_spacia", "keyword": "スペーシア", "target_images": 1000},
    {"jp_name": "ホンダ N-WGN", "en_name": "honda_nwgn", "keyword": "N-WGN", "target_images": 1000},
    {"jp_name": "日産 ルークス", "en_name": "nissan_roox", "keyword": "ルークス", "target_images": 1000},
    {"jp_name": "トヨタ パッソ", "en_name": "toyota_passo", "keyword": "パッソ", "target_images": 1000},
    {"jp_name": "トヨタ エスティマ", "en_name": "toyota_estima", "keyword": "エスティマ", "target_images": 1000},
    {"jp_name": "日産 リーフ", "en_name": "nissan_leaf", "keyword": "リーフ", "target_images": 1000},
    {"jp_name": "スバル XV", "en_name": "subaru_xv", "keyword": "XV", "target_images": 1000},
    {"jp_name": "レクサス UX", "en_name": "lexus_ux", "keyword": "レクサスUX", "target_images": 1000},
    {"jp_name": "三菱 デリカD:5", "en_name": "mitsubishi_delica_d5", "keyword": "デリカD:5", "target_images": 1000},
    {"jp_name": "ダイハツ アトレー", "en_name": "daihatsu_atray", "keyword": "アトレー", "target_images": 1000},

]

# ✅ 画像収集関数
def collect_images(car_info, max_pages=300, save_dir=SAVE_DIR):
    start_time = time.perf_counter()
    car_name_ja = car_info["jp_name"]
    car_name_en = car_info["en_name"]
    keyword = car_info["keyword"]
    target_images = car_info["target_images"]
    save_path = os.path.join(save_dir, car_name_en)
    os.makedirs(save_path, exist_ok=True)

    # スキップ条件
    existing_images = [f for f in os.listdir(save_path) if f.endswith(".jpg")]
    if len(existing_images) >= target_images:
        print(f"\n✅ {car_name_ja}（{car_name_en}）：{len(existing_images)} 枚保存済 → スキップ")
        return

    print(f"\n🚗 {car_name_ja}（{car_name_en}）の画像収集を開始します")
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    pattern = re.compile(r"_00\dL\.JPG$")
    seen_urls = set()
    count = len(existing_images)

    for page in range(1, max_pages + 1):
        if count >= target_images:
            break

        url = (
            f"https://www.carsensor.net/usedcar/freeword/{keyword}/index.html"
            if page == 1 else
            f"https://www.carsensor.net/usedcar/freeword/{keyword}/index{page}.html"
        )

        print(f"🌐 {car_name_en} ページ {page}: {url}")
        try:
            driver.get(url)
            time.sleep(1)
            height = driver.execute_script("return document.body.scrollHeight")
            for y in range(0, height, 1000):
                driver.execute_script(f"window.scrollTo(0, {y});")
                time.sleep(0.2)
        except Exception as e:
            print(f"❌ ページ取得失敗: {e}")
            continue

        img_tags = driver.find_elements(By.TAG_NAME, "img")
        for img in img_tags:
            if count >= target_images:
                break
            src = img.get_attribute("src")
            if not src or "ccsrpcma.carsensor.net" not in src or not pattern.search(src):
                continue
            if src.startswith("//"):
                src = "https:" + src
            if src in seen_urls:
                continue
            seen_urls.add(src)

            filename = f"{car_name_en}_{count:04}.jpg"
            filepath = os.path.join(save_path, filename)
            try:
                urllib.request.urlretrieve(src, filepath)
                if os.path.getsize(filepath) < 10 * 1024:
                    os.remove(filepath)
                    continue
                print(f"✅ 保存: {filename}")
                count += 1
            except Exception as e:
                print(f"❌ 保存失敗: {e}")

    driver.quit()
    elapsed = time.perf_counter() - start_time
    print(f"🎯 {car_name_ja} 完了：{count} 枚保存（{elapsed:.2f} 秒）")

# ✅ メイン処理（逐次 or 並列）
if __name__ == "__main__":
    start_all = time.perf_counter()

    if USE_PARALLEL:
        print(f"\n🚀 並列処理で画像収集を開始（最大 {MAX_WORKERS} 並列）")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(collect_images, car_list)
    else:
        print("\n🚶 逐次処理で画像収集を開始")
        for car in car_list:
            collect_images(car)

    end_all = time.perf_counter()
    print(f"\n🧾 合計所要時間: {end_all - start_all:.2f} 秒（{(end_all - start_all)/60:.2f} 分）")
