import os
import cv2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(SCRIPT_DIR, "bilateral_inputs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "bilateral_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MAX_N = 15


d = 7               
sigmaColor = 30      
sigmaSpace = 30      


all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(EXTS)]
all_files.sort()
files = all_files[:MAX_N]

if len(files) == 0:
    raise RuntimeError(f"No images found in: {INPUT_DIR}")

print(f"Found {len(files)} images to process.")
print("Output dir:", OUTPUT_DIR)


for idx, fname in enumerate(files, start=1):
    in_path = os.path.join(INPUT_DIR, fname)

    img = cv2.imread(in_path) 
    if img is None:
        print(f"[Skip] Cannot read: {in_path}")
        continue


    denoised = cv2.bilateralFilter(
        img,
        d=d,
        sigmaColor=sigmaColor,
        sigmaSpace=sigmaSpace
    )

    base, ext = os.path.splitext(fname)
    out_name = f"{base}_bilateral{ext}"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    cv2.imwrite(out_path, denoised)
    print(f"[{idx:02d}] {fname} -> {out_name}")

print("Done.")
