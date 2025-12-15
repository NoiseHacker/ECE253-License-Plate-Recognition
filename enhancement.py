import os
import cv2
import numpy as np

def enhance_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE enhancement on the L channel of LAB image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced


def enhance_histogram_equalization(image, strength=1.0):
    """
    Adjustable-strength global Histogram Equalization.
    """

    # Convert to Y channel (luminance only)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Apply global HE on Y channel
    y_eq = cv2.equalizeHist(y)

    # Blend: y_mix = strength * HE + (1 - strength) * original
    y_mix = cv2.addWeighted(y_eq, strength, y, 1 - strength, 0)

    # Merge back
    merged = cv2.merge((y_mix, cr, cb))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return enhanced



def batch_enhance(
    input_folder="contrast/degraded",
    output_folder="contrast/enhance",
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    strength=0.9
):
    """
    For each degraded image, output:
        - CLAHE enhanced version
        - Histogram Equalization enhanced version
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
            continue

        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)
        if image is None:
            print(f"Failed to load {input_path}")
            continue

        name, ext = os.path.splitext(filename)

        # 1. CLAHE enhancement
        enhanced_clahe = enhance_clahe(image, clip_limit, tile_grid_size)
        output_path_clahe = os.path.join(output_folder, f"{name}_CLAHE{ext}")
        cv2.imwrite(output_path_clahe, enhanced_clahe)

        # 2. Global histogram equalization
        enhanced_he = enhance_histogram_equalization(image, strength)
        output_path_he = os.path.join(output_folder, f"{name}_HE{ext}")
        cv2.imwrite(output_path_he, enhanced_he)

        print(f"Saved: {output_path_clahe}")
        print(f"Saved: {output_path_he}")


if __name__ == "__main__":
    batch_enhance(
        input_folder="contrast/degraded",
        output_folder="contrast/enhance",
        clip_limit=3,            # CLAHE parameter
        tile_grid_size=(1, 1),   # CLAHE parameter
        strength=0.8         # HE blending strength  
    )
