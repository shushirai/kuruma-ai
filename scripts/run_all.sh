#!/bin/bash

echo "✅ 画像収集"
python scrape_car_img.py

echo "✅ 車以外の画像を除外"
python filter_car_yolo.py

echo "✅ train/test/valに画像を分割（8:1:1）"
python split_dataset.py

echo "🎯 すべて完了！"
