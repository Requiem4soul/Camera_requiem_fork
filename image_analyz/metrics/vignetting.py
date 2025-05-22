import numpy as np
import cv2
import matplotlib.pyplot as plt


def calculate_vignetting(image: np.ndarray, show_plot: bool = True) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    grad_radial = compute_radial_gradient(gray)

    if np.allclose(grad_radial, 0, atol=1e-6):
        print("Кто-то решил выложить искусственный белый лист")
        return 10.0

    asymmetry_score = compute_gradient_asymmetry(grad_radial, show_plot=show_plot)

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


def compute_gradient_asymmetry(grad_radial: np.ndarray, bins: int = 100, show_plot: bool = False) -> float:
    grad_log = np.sign(grad_radial) * np.log1p(np.abs(grad_radial))
    grad_log_flat = grad_log.flatten()

    hist, bin_edges = np.histogram(grad_log_flat, bins=bins, density=True)

    mid = bins // 2
    h_pos = hist[mid:]
    h_neg = hist[:mid][::-1]

    if show_plot:
        # Строим гистограмму
        hist, bin_edges = np.histogram(grad_log_flat, bins=bins, density=True)

    min_len = min(len(h_pos), len(h_neg))
    h_pos = h_pos[:min_len]
    h_neg = h_neg[:min_len]

    sum_pos = np.sum(h_pos)
    sum_neg = np.sum(h_neg)

    # area_diff = abs(sum_pos - sum_neg) / (sum_pos + sum_neg + 1e-8)

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

    # === ВИЗУАЛИЗАЦИЯ ===
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


def plot_gradient_histogram(hist, bin_edges, grad_flat):
    """
    Строит детальный график гистограммы радиальных градиентов с анализом асимметрии
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Анализ асимметрии радиальных градиентов', fontsize=16)

    # График 1: Полная гистограмма
    ax1 = axes[0, 0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax1.bar(bin_centers, hist, width=bin_centers[1] - bin_centers[0], alpha=0.7, color='skyblue')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Центр (0)')
    ax1.set_xlabel('Радиальный градиент (log scale)')
    ax1.set_ylabel('Плотность')
    ax1.set_title('Полная гистограмма радиальных градиентов')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: Разделение на положительную и отрицательную части
    ax2 = axes[0, 1]
    mid = len(hist) // 2

    # Отрицательная часть (левая)
    neg_bins = bin_centers[:mid]
    neg_hist = hist[:mid]
    ax2.bar(neg_bins, neg_hist, width=bin_centers[1] - bin_centers[0],
            alpha=0.7, color='red', label='Отрицательные градиенты')

    # Положительная часть (правая)
    pos_bins = bin_centers[mid:]
    pos_hist = hist[mid:]
    ax2.bar(pos_bins, pos_hist, width=bin_centers[1] - bin_centers[0],
            alpha=0.7, color='blue', label='Положительные градиенты')

    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.8)
    ax2.set_xlabel('Радиальный градиент (log scale)')
    ax2.set_ylabel('Плотность')
    ax2.set_title('Разделение по знаку градиента')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График 3: Анализ симметрии
    ax3 = axes[1, 0]

    # Подготовка данных для сравнения симметрии
    h_pos = hist[mid:]
    h_neg = hist[:mid][::-1]  # инвертируем для симметрии

    min_len = min(len(h_pos), len(h_neg))
    h_pos_trim = h_pos[:min_len]
    h_neg_trim = h_neg[:min_len]

    x_symmetric = np.arange(min_len)

    ax3.plot(x_symmetric, h_pos_trim, 'b-', linewidth=2, label='Положительная часть', marker='o', markersize=4)
    ax3.plot(x_symmetric, h_neg_trim, 'r-', linewidth=2, label='Отрицательная часть (отражена)', marker='s',
             markersize=4)
    ax3.fill_between(x_symmetric, h_pos_trim, h_neg_trim, alpha=0.3, color='yellow', label='Разность')

    ax3.set_xlabel('Индекс бина (от центра к краю)')
    ax3.set_ylabel('Плотность')
    ax3.set_title('Сравнение симметричных частей')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # График 4: Статистика
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Вычисляем статистики
    total_neg = np.sum(hist[:mid])
    total_pos = np.sum(hist[mid:])
    area_diff = abs(total_pos - total_neg) / (total_pos + total_neg + 1e-8)

    # KL-дивергенция
    sum_pos = np.sum(h_pos_trim)
    sum_neg = np.sum(h_neg_trim)
    if sum_pos > 1e-8 and sum_neg > 1e-8:
        h_pos_norm = h_pos_trim / sum_pos
        h_neg_norm = h_neg_trim / sum_neg
        kl_div = 0.0
        for i in range(len(h_pos_norm)):
            if h_pos_norm[i] > 1e-8 and h_neg_norm[i] > 1e-8:
                kl_div += h_pos_norm[i] * np.log(h_pos_norm[i] / h_neg_norm[i])
    else:
        kl_div = 0.0

    # Основные статистики градиентов
    mean_grad = np.mean(grad_flat)
    std_grad = np.std(grad_flat)
    neg_ratio = np.sum(grad_flat < 0) / len(grad_flat)

    stats_text = f"""
    СТАТИСТИКА РАДИАЛЬНЫХ ГРАДИЕНТОВ:

    Основные характеристики:
    • Среднее значение: {mean_grad:.4f}
    • Стандартное отклонение: {std_grad:.4f}
    • Доля отрицательных: {neg_ratio:.3f}

    Анализ асимметрии:
    • Площадь отрицательной части: {total_neg:.4f}
    • Площадь положительной части: {total_pos:.4f}
    • Разность площадей: {area_diff:.4f}
    • KL-дивергенция: {kl_div:.4f}

    Интерпретация:
    • area_diff > 0.1: заметная асимметрия
    • neg_ratio > 0.6: возможное виньетирование
    • kl_div > 0.5: значительные различия
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.show()


def create_test_images():
    """
    Создает тестовые изображения для демонстрации
    """
    size = 400

    # 1. Однородное изображение
    uniform_img = np.ones((size, size), dtype=np.uint8) * 200

    # 2. Изображение с виньетированием
    y, x = np.indices((size, size))
    cx, cy = size // 2, size // 2
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.max(distances)

    # Виньетирование: яркость падает к краям
    vignette_factor = 1.0 - 0.4 * (distances / max_dist) ** 2
    vignette_img = (200 * vignette_factor).astype(np.uint8)

    # 3. Изображение с градиентом
    gradient_img = np.linspace(50, 250, size).reshape(1, -1)
    gradient_img = np.repeat(gradient_img, size, axis=0).astype(np.uint8)

    return uniform_img, vignette_img, gradient_img

