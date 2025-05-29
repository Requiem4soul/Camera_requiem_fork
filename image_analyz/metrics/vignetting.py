import numpy as np
import cv2
import matplotlib.pyplot as plt


def calculate_vignetting(image: np.ndarray, show_plot: bool = False) -> dict:
    """
    Вычисляет оценку виньетирования изображения и возвращает детальную информацию
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    grad_radial = compute_radial_gradient(gray)

    # Проверка на однородное изображение
    if np.allclose(grad_radial, 0, atol=1e-6):
        print("Кто-то решил выложить искусственный белый лист")
        return {
            "vignetting": 10.0,
            "hist": [],
            "bin_edges": [],
            "grad_flat": []
        }

    # Вычисляем асимметрию (ИСПОЛЬЗУЕМ результат!)
    asymmetry_score = compute_gradient_asymmetry(grad_radial, bins=100, show_plot=show_plot)

    # Обрезаем аномальные значения
    asymmetry_score = min(asymmetry_score, 2.0)
    # Преобразование в 10-балльную шкалу (линейное)
    score = max(0.0, 10.0 - 5.0 * asymmetry_score)

    # Подготавливаем данные для возврата
    grad_log = np.sign(grad_radial) * np.log1p(np.abs(grad_radial))
    grad_flat = grad_log.flatten()
    hist, bin_edges = np.histogram(grad_flat, bins=100, density=True)

    return {
        "vignetting": round(score, 3),
        "hist": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


def compute_radial_gradient(image: np.ndarray) -> np.ndarray:
    """
    Вычисляет радиальный градиент изображения
    """
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


def compute_gradient_asymmetry(grad_radial: np.ndarray, bins: int = 100, show_plot: bool = False) -> float:
    """
    Вычисляет асимметрию градиента для оценки виньетирования
    """
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

    kl_div = 0.0
    for i in range(len(h_pos_norm)):
        if h_pos_norm[i] > 1e-8 and h_neg_norm[i] > 1e-8:
            kl_div += h_pos_norm[i] * np.log(h_pos_norm[i] / h_neg_norm[i])

    asymmetry = 0.5 * kl_div + 0.5 * area_diff

    # Раньше использовал для отладки
    if show_plot:
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.figure(figsize=(8, 4))
        plt.title("Гистограмма логарифмированных радиальных градиентов")
        plt.plot(centers, hist, label="Гистограмма", color="blue")
        plt.axvline(0, color='red', linestyle='--', label="Ось симметрии")
        plt.xlabel("log(градиент по радиусу)")
        plt.ylabel("Плотность")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return asymmetry