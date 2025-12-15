import os
import cv2
import numpy as np

def degrade_contrast_linear(image, alpha=0.5, beta=0):
    """
    Linear contrast degradation:
    output = alpha * input + beta
    """
    degraded = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return degraded

def degrade_contrast_gamma(image, gamma=2.0):
    """
    Gamma correction for contrast degradation:
    gamma > 1 makes the image darker and reduces contrast
    gamma < 1 brightens and increases low contrast
    """
    gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, gamma_table)

def batch_degrade_contrast(
    input_folder,
    output_folder,
    method="linear",
    alpha=0.5,
    beta=0,
    gamma=2.0
):
    """
    Generate degraded contrast images for all images in a folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
            continue

        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)
        if image is None:
            print(f"Failed to read {input_path}")
            continue

        # choose degradation type
        if method == "linear":
            degraded = degrade_contrast_linear(image, alpha=alpha, beta=beta)
        elif method == "gamma":
            degraded = degrade_contrast_gamma(image, gamma=gamma)
        else:
            raise ValueError("Unknown method. Choose 'linear' or 'gamma'.")

        # save degraded image
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}{ext}")
        cv2.imwrite(output_path, degraded)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    batch_degrade_contrast(
        input_folder="contrast/clean",
        output_folder="contrast/degraded",
        method="gamma",   # "linear" or "gamma"
        alpha=1.2,         # contrast adjustment if method="linear"
        beta=10,           # lightness adjustment if method="linear"
        gamma=0.4       # if method="gamma"
    )
