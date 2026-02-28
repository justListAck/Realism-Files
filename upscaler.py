import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
import os
import time
import cv2
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# 1. Hardware Initialization (4070 Ti Super)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4, 
    model_path='RealESRGAN_x4plus.pth', # Ensure this file is in the 'Realism' folder
    model=model, 
    tile=0, 
    device=device
)

IN_DIR = "input_queue/"
OUT_BASE = "../resourcepacks/Realism/assets/minecraft/textures/"

print(f"🚀 Realism HD Engine Active on {device.upper()}")

while True:
    for filename in os.listdir(IN_DIR):
        if filename.endswith(".png"):
            try:
                # Load the 16x texture
                img_path = os.path.join(IN_DIR, filename)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                
                # AI Upscale: 16x -> 1024x (Pixel-less)
                # We use outscale=64 because 16 * 64 = 1024
                output, _ = upsampler.enhance(img, outscale=64)
                
                # Reconstruct Minecraft folder path
                # 'block_grass.png' -> 'block/grass.png'
                sub_path = filename.replace("_", "/")
                final_out = os.path.join(OUT_BASE, sub_path)
                os.makedirs(os.path.dirname(final_out), exist_ok=True)
                
                # Save & Cleanup
                cv2.imwrite(final_out, output)
                os.remove(img_path)
                print(f"💎 Rendered HD: {sub_path}")
                
            except Exception as e:
                print(f"⚠️ Skipping {filename}: {e}")
                
    time.sleep(0.1) # Prevent CPU spiking