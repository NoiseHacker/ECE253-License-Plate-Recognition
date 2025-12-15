import os
import cv2
import numpy as np
from scipy import fftpack
from skimage.restoration import richardson_lucy
from skimage import img_as_float, img_as_ubyte
from PIL import Image, ImageOps


def load_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = np.array(img)

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def resize_if_large(img, max_width=800):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        print(f"  [resize] {w}x{h} -> {max_width}x{new_h}")
        return cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return img


def slight_blur(img_rgb):
    return cv2.GaussianBlur(img_rgb, (5, 5), 0)


def motion_psf(length=25, angle=0):
    k = length if length % 2 == 1 else length + 1
    psf = np.zeros((k, k), dtype=np.float32)
    center = k // 2

    angle_rad = np.deg2rad(angle)
    dx = (length - 1) / 2 * np.cos(angle_rad)
    dy = (length - 1) / 2 * np.sin(angle_rad)

    x1 = int(round(center - dx))
    y1 = int(round(center - dy))
    x2 = int(round(center + dx))
    y2 = int(round(center + dy))

    cv2.line(psf, (x1, y1), (x2, y2), 1, thickness=1)
    psf /= psf.sum()
    return psf


def convolve_fft_channel(img_ch, kernel):
    s1 = np.array(img_ch.shape)
    s2 = np.array(kernel.shape)
    size = s1 + s2

    FI = fftpack.fft2(img_ch, shape=size)
    FH = fftpack.fft2(kernel, shape=size)
    conv = np.real(fftpack.ifft2(FI * FH))

    return conv[:img_ch.shape[0], :img_ch.shape[1]]


def add_noise(imgf, sigma=0.0):
    if sigma <= 0:
        return imgf
    noisy = imgf + np.random.normal(scale=sigma, size=imgf.shape)
    return np.clip(noisy, 0.0, 1.0)


def wiener_deconv_channel(blur, kernel, K=0.01):
    eps = 1e-8
    s1 = np.array(blur.shape)
    s2 = np.array(kernel.shape)
    size = s1 + s2

    G = fftpack.fft2(blur, shape=size)
    H = fftpack.fft2(kernel, shape=size)
    denom = (np.abs(H) ** 2) + K

    F_hat = np.conj(H) / (denom + eps) * G
    f = np.real(fftpack.ifft2(F_hat))

    f = f[:blur.shape[0], :blur.shape[1]]
    return np.clip(f, 0, 1)


def wiener_color(blur_u8, kernel, K=0.01):
    blurf = img_as_float(blur_u8)
    if blurf.ndim == 2:
        return img_as_ubyte(wiener_deconv_channel(blurf, kernel, K))

    outs = []
    for c in range(3):
        outs.append(wiener_deconv_channel(blurf[..., c], kernel, K))
    out = np.stack(outs, axis=2)
    return img_as_ubyte(np.clip(out, 0, 1))


def rl_color(blur_u8, kernel, iterations=30):
    blurf = img_as_float(blur_u8)

    def run_rl(ch):
        return richardson_lucy(ch, kernel, num_iter=iterations)

    if blurf.ndim == 2:
        return img_as_ubyte(run_rl(blurf))

    outs = []
    for c in range(3):
        outs.append(run_rl(blurf[..., c]))
    out = np.stack(outs, axis=2)
    return img_as_ubyte(np.clip(out, 0, 1))


def make_compare(orig, blur, wiener, rl):
    h = orig.shape[0]

    def fit(img):
        if img.shape[0] != h:
            scale = h / img.shape[0]
            neww = int(img.shape[1] * scale)
            return cv2.resize(img, (neww, h))
        return img

    imgs = [fit(orig), fit(blur), fit(wiener), fit(rl)]
    return cv2.hconcat(imgs)


def make_compare_grid(orig, blur, wiener, rl):
    h = orig.shape[0]
    w = orig.shape[1]

    def resize_to_same(img, target_h, target_w):
        return cv2.resize(img, (target_w, target_h))

    orig = resize_to_same(orig, h, w)
    blur = resize_to_same(blur, h, w)
    wiener = resize_to_same(wiener, h, w)
    rl = resize_to_same(rl, h, w)

    top = cv2.hconcat([orig, blur])
    bottom = cv2.hconcat([wiener, rl])
    grid = cv2.vconcat([top, bottom])
    return grid


def run(input_dir="./images/orig",
        out_dir="./images/out",
        length=25,
        angle=10,
        noise_sigma=0.01,
        K=0.01,
        rl_iters=40):

    os.makedirs(out_dir, exist_ok=True)
    for sub in ["blur", "wiener", "rl", "compare"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    kernel = motion_psf(length, angle)
    print("PSF:", kernel.shape)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        print("Processing:", fname)
        path = os.path.join(input_dir, fname)

        img_bgr = load_image(path)
        img_bgr = resize_if_large(img_bgr, max_width=800)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = slight_blur(img_rgb)

        imgf = img_as_float(img_rgb)

        blurred = np.zeros_like(imgf)
        for c in range(3):
            blurred[..., c] = convolve_fft_channel(imgf[..., c], kernel)
        blurred = add_noise(blurred, sigma=noise_sigma)
        blurred_u8 = img_as_ubyte(blurred)

        blur_bgr = cv2.cvtColor(blurred_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, "blur", fname), blur_bgr)
