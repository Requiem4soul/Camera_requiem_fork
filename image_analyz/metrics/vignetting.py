import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_vignetting(image: np.ndarray, SHOW: bool = True) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Сглаживание помогает при шуме
    gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=1.0)

    grad_radial = compute_radial_gradient(gray)

    if np.allclose(grad_radial, 0, atol=1e-6):
        print("Пустое или синтетическое изображение")
        return 10.0

    asymmetry_score = compute_gradient_asymmetry(grad_radial, SHOW=SHOW)
    print(f"asym: {asymmetry_score}")

    # Обрезаем аномальные значения
    asymmetry_score = min(asymmetry_score, 2.0)

    # Преобразование в 10-балльную шкалу (линейное)
    score = max(0.0, 10.0 - 5.0 * asymmetry_score)
    return round(score, 3)

def compute_radial_gradient(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    cx, cy = w // 2, h // 2
    y, x = np.indices((h, w))
    x = x - cx
    y = y - cy

    r = np.sqrt(x ** 2 + y ** 2)
    r[r == 0] = 1e-6
    x_norm = x / r
    y_norm = y / r

    gx = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)

    grad_radial = gx * x_norm + gy * y_norm
    return grad_radial

def compute_gradient_asymmetry(grad_radial: np.ndarray, bins: int = 100, SHOW: bool = False) -> float:
    grad_log = np.sign(grad_radial) * np.log1p(np.abs(grad_radial))
    grad_log_flat = grad_log.flatten()

    hist, bin_edges = np.histogram(grad_log_flat, bins=bins, density=True)
    mid = bins // 2
    h_pos = hist[mid:]
    h_neg = hist[:mid][::-1]

    min_len = min(len(h_pos), len(h_neg))
    h_pos = h_pos[:min_len]
    h_neg = h_neg[:min_len]

    sum_pos = np.sum(h_pos)
    sum_neg = np.sum(h_neg)

    if sum_pos < 1e-8 or sum_neg < 1e-8:
        return 0.0

    h_pos_norm = h_pos / sum_pos
    h_neg_norm = h_neg / sum_neg

    area_diff = abs(sum_pos - sum_neg)
    print(f"area_diff: {area_diff}")

    kl_div = 0.0
    for i in range(len(h_pos_norm)):
        if h_pos_norm[i] > 1e-8 and h_neg_norm[i] > 1e-8:
            kl_div += h_pos_norm[i] * np.log(h_pos_norm[i] / h_neg_norm[i])

    asymmetry = 0.5 * kl_div + 0.5 * area_diff

    if SHOW:
        x = np.linspace(-1, 1, 2 * min_len)
        plt.figure(figsize=(8, 4))
        plt.plot(x[min_len:], h_pos, label='Right side (positive)', color='green')
        plt.plot(x[:min_len], h_neg[::-1], label='Left side (reflected)', color='red')
        plt.axvline(0, color='gray', linestyle='--')
        plt.title('Symmetry of Log-Radial Gradient Histogram')
        plt.xlabel('Log-Gradient Bins (symmetrized)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return asymmetry
