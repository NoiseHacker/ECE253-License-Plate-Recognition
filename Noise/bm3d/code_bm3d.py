import os
import cv2
import numpy as np
from bm3d import bm3d


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


INPUT_DIR =os.path.join(SCRIPT_DIR, "bm3d_inputs")


OUTPUT_DIR = os.path.join(SCRIPT_DIR, "bm3d_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


EXTS = (".jpg", ".jpeg", ".png", ".bmp")


sigma = 30 / 255.0


MAX_N = 15


all_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(EXTS)
]



all_files.sort()  
files = all_files[:MAX_N]

if len(files) == 0:
    raise RuntimeError(f"No images found in: {INPUT_DIR}")

print(f"Found {len(files)} images to process.")
print("Output dir:", OUTPUT_DIR)

for idx, fname in enumerate(files, start=1):
    in_path = os.path.join(INPUT_DIR, fname)

    img = cv2.imread(in_path)  # BGR uint8
    if img is None:
        print(f"[Skip] Cannot read: {in_path}") 
        continue

    img_f = img.astype(np.float32) / 255.0

    denoised_channels = []
    for ch in range(3):  # B,G,R
        denoised_ch = bm3d(img_f[:, :, ch], sigma_psd=sigma)
        denoised_channels.append(denoised_ch)

    denoised_color = np.stack(denoised_channels, axis=2)
    denoised_u8 = (denoised_color * 255.0).clip(0, 255).astype(np.uint8)

    base, ext = os.path.splitext(fname)
    out_name = f"{base}_bm3d{ext}"

    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, denoised_u8)

    print(f"[{idx:02d}] {fname} -> {out_name}")

print("Done.")
