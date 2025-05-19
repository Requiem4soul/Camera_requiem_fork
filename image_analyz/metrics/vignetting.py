import numpy as np
import cv2

def calculate_vignetting(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    grad_radial = compute_radial_gradient(gray)

    if np.allclose(grad_radial, 0, atol=1e-6):  # Все градиенты близки к нулю, проверка для синтетики
        print("Кто-то решил выложить искусственный белый лист")
        return 10.0

    asymmetry_score = compute_gradient_asymmetry(grad_radial)

    score = min(asymmetry_score, 10.0)
    score = round(10.0 - score, 3)
    
    return score

def compute_radial_gradient(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    cx, cy = w // 2, h // 2
    # Получаем координаты каждого пикселя
    y, x = np.indices((h, w))
    x = x - cx
    y = y - cy

    # Нормированный радиус вектора
    r = np.sqrt(x ** 2 + y ** 2)
    r[r == 0] = 1e-6  # избегаем деления на 0
    x_norm = x / r
    y_norm = y / r

    # Градиенты изображения (пока не тот, который нам нужен)
    gx = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)

    # Радиальный градиент (проекция вектора градиента на радиус-вектор)
    grad_radial = gx * x_norm + gy * y_norm
    return grad_radial

def compute_gradient_asymmetry(grad_radial: np.ndarray, bins: int = 100) -> float:
    # Приводим к логарифмическому масштабу (на нём более видны мелкие отклонения)
    grad_log = np.sign(grad_radial) * np.log1p(np.abs(grad_radial))
    grad_log_flat = grad_log.flatten()

    # Строим гистограмму
    hist, bin_edges = np.histogram(grad_log_flat, bins=bins, density=True)
    mid = bins // 2
    h_pos = hist[mid:]
    h_neg = hist[:mid][::-1]  # инвертируем левую часть (симметрия)

    # Усечение до одинаковой длины
    min_len = min(len(h_pos), len(h_neg))
    h_pos = h_pos[:min_len]
    h_neg = h_neg[:min_len]

    # Нормализация
    h_pos /= (np.sum(h_pos) + 1e-8)
    h_neg /= (np.sum(h_neg) + 1e-8)

    # Оценка симметрии через KL-дивергенцию и разницу площадей
    kl_div = np.sum(h_pos * np.log((h_pos + 1e-8) / (h_neg + 1e-8)))
    area_diff = np.abs(np.sum(h_pos) - np.sum(h_neg))

    # Смешанная метрика
    asymmetry = 0.5 * kl_div + 0.5 * area_diff
    return asymmetry
