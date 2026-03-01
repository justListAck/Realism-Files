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
OUT_BASE = "../resourcepacks/Realism/assets/minecraft/"

print(f"🚀 Realism HD Engine Active on {device.upper()}")

import os

for f in os.listdir(IN_DIR):
    if f.endswith(".png"):
        parts = f.split('_')
        
        if len(parts) >= 3:
            sub_folders = os.path.join(*parts[:2])
            file_name = "_".join(parts[2:])
        else:
            sub_folders = parts[0]
            file_name = parts[1] if len(parts) > 1 else parts[0]

        final_out_dir = os.path.join(OUT_BASE, sub_folders)
        os.makedirs(final_out_dir, exist_ok=True)
        
        img = cv2.imread(os.path.join(IN_DIR, f), cv2.IMREAD_UNCHANGED)
        if img is None: continue
        
        output, _ = upsampler.enhance(img, outscale=64)
        cv2.imwrite(os.path.join(sub_folders, file_name), output)
        
        os.remove(os.path.join(IN_DIR, f))
        print(f"✅ Correctly saved to: {os.path.join(sub_folders, file_name)}")
