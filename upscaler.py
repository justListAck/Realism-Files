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

for f in os.listdir(IN_DIR):
    if f.endswith(".png"):
        # 1. Split 'textures_block_grass.png' into ['textures', 'block', 'grass.png']
        parts = f.split('_')
        
        # 2. Reconstruct the folder path (textures/block/)
        sub_folder = "/".join(parts[:-1]) 
        # The last part is the actual filename (grass.png)
        file_name = parts[-1]
        
        # 3. Final path in your Realism pack
        final_dest_folder = os.path.join(OUT_BASE, sub_folder)
        os.makedirs(final_dest_folder, exist_ok=True)
        
        # 4. Process
        img = cv2.imread(os.path.join(IN_DIR, f), cv2.IMREAD_UNCHANGED)
        if img is None: continue
        
        output, _ = upsampler.enhance(img, outscale=64) # 16x to 1024x
        cv2.imwrite(os.path.join(final_dest_folder, file_name), output)
        
        # 5. Remove from queue so it doesn't process twice
        os.remove(os.path.join(IN_DIR, f))
        print(f"💎 Upscaled: {sub_folder}/{file_name}")
