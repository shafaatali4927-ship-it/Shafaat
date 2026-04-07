#!/bin/bash
# Models download karo (Railway pe pehli baar)
mkdir -p models

if [ ! -f "models/GFPGANv1.4.pth" ]; then
    echo "📥 GFPGAN model download ho raha hai..."
    wget -q -O models/GFPGANv1.4.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
fi

if [ ! -f "models/RealESRGAN_x4plus.pth" ]; then
    echo "📥 Real-ESRGAN model download ho raha hai..."
    wget -q -O models/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
fi

echo "✅ Models ready!"
python bot.py
