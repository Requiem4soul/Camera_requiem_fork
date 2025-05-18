import numpy as np
import cv2

def calculate_vignetting(image: np.ndarray) -> float:
    grad = compute_radial_gradients(image)
    gamma = compute_asymmetry(grad)
    if gamma <= 1:
        gamma = 1 - gamma
    else:
        gamma = 0
    gamma = gamma * 10
    return gamma

def compute_radial_gradients(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    y, x = np.indices((h, w))
    x = x - w / 2
    y = y - h / 2
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    r = np.sqrt(x ** 2 + y ** 2)
    r[r == 0] = 1e-6
    grad = (gx * x + gy * y) / r
    return grad

def compute_asymmetry(grad: np.ndarray, lambda_h=0.5) -> float:
    grad = grad.flatten()
    # Логарифмический масштаб градиентов
    grad_log = np.log(1 + np.abs(grad)) * np.sign(grad)  # Сохраняем знак
    hist, bins = np.histogram(grad_log, bins=50, density=False)
    mid = np.where(bins >= 0)[0][0]  # Делим по нулю
    pos = hist[mid:]
    neg = hist[:mid][::-1]
    min_len = min(len(pos), len(neg))
    pos, neg = pos[:min_len], neg[:min_len]
    A1, A2 = np.sum(pos), np.sum(neg)
    if A1 == 0 or A2 == 0:
        return 0.0
    H_plus = pos / A1
    H_minus = neg / A2
    kl_div = np.sum(H_plus * np.log((H_plus + 1e-10) / (H_minus + 1e-10)))
    # Нормализованная разница площадей
    total = A1 + A2
    area_diff = (np.abs(A1 - A2) / total) ** 0.25 if total > 0 else 0
    return lambda_h * kl_div + (1 - lambda_h) * area_diff