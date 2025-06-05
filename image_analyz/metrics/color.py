import cv2
import numpy as np

# Координаты патчей (4 строки x 6 столбцов)
# В виде относительных координат (от 0 до 1)
PATCH_GRID = (4, 6)

# Патчи идут слева направо, сверху вниз
# Для каждого патча: (y0, y1, x0, x1) в относительных координатах
# Патчи равномерно распределены, с небольшим отступом от краёв
PATCH_MARGIN = 0.03  # 3% отступ от краёв


def get_patch_coords(img_shape, row, col):
    h, w = img_shape[:2]
    n_rows, n_cols = PATCH_GRID
    patch_h = (1 - 2 * PATCH_MARGIN) / n_rows
    patch_w = (1 - 2 * PATCH_MARGIN) / n_cols
    y0 = PATCH_MARGIN + row * patch_h
    y1 = y0 + patch_h
    x0 = PATCH_MARGIN + col * patch_w
    x1 = x0 + patch_w
    # В пикселях
    return (int(y0 * h), int(y1 * h), int(x0 * w), int(x1 * w))


def extract_patches(image):
    """
    Возвращает массив средних RGB-цветов для 24 патчей (4x6).
    """
    patches = []
    for row in range(PATCH_GRID[0]):
        for col in range(PATCH_GRID[1]):
            y0, y1, x0, x1 = get_patch_coords(image.shape, row, col)
            patch = image[y0:y1, x0:x1]
            mean_color = patch.mean(axis=(0, 1))  # BGR
            patches.append(mean_color[::-1])  # В RGB
    return np.array(patches)  # shape (24, 3)


def analyze_colorchecker(user_img_path, reference_img_path):
    """
    Основная функция: сравнивает пользовательское фото с эталоном по colorchecker.
    Возвращает dict с метриками: white_balance, color_gamut, contrast_ratio.
    """
    # Загружаем изображения
    user_img = cv2.imread(user_img_path)
    ref_img = cv2.imread(reference_img_path)
    if user_img is None or ref_img is None:
        raise ValueError("Не удалось загрузить изображения!")

    # Приводим к одному размеру (размеру эталона)
    user_img = cv2.resize(user_img, (ref_img.shape[1], ref_img.shape[0]))

    # Извлекаем патчи
    user_patches = extract_patches(user_img)
    ref_patches = extract_patches(ref_img)

    # --- Баланс белого ---
    # Серые патчи: последние 6 (нижний ряд)
    user_gray = user_patches[18:24]
    ref_gray = ref_patches[18:24]
    # Средний цвет по каждому каналу
    user_gray_mean = user_gray.mean(axis=0)
    ref_gray_mean = ref_gray.mean(axis=0)
    # Отклонение от эталона (1.0 = идеально)
    white_balance = np.mean(user_gray_mean / (ref_gray_mean + 1e-8))

    # --- Цветовой охват ---
    # Считаем площадь охвата патчей в RGB (можно в Convex Hull, но проще — среднее отклонение)
    color_diff = np.linalg.norm(user_patches - ref_patches, axis=1)
    # Чем меньше diff, тем лучше. Преобразуем в % охвата (условно: 100% - среднее отклонение/макс_откл*100)
    max_diff = np.linalg.norm([255, 255, 255])
    avg_diff = np.mean(color_diff)
    color_gamut = 100 - (avg_diff / max_diff) * 100

    # --- Контрастность ---
    # Яркость патчей (по формуле luminance)
    def luminance(rgb):
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    user_lums = np.array([luminance(rgb) for rgb in user_patches])
    min_lum = user_lums.min()
    max_lum = user_lums.max()
    # Контрастность как отношение максимальной к минимальной яркости
    contrast_ratio = (max_lum + 1e-8) / (min_lum + 1e-8)

    return {
        "white_balance": float(white_balance),
        "color_gamut": float(color_gamut),
        "contrast_ratio": float(contrast_ratio),
    }


def calculate_color_gamut(image_data):
    """
    Оценивает цветовой охват в % от sRGB.
    Упрощённый подход: сравниваем с "идеальным" sRGB изображением.
    """
    # Конвертируем в RGB (OpenCV использует BGR)
    rgb_img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    # Упрощённая оценка: сравниваем с "идеальным" белым (255, 255, 255)
    white_pixel = np.array([255, 255, 255])
    avg_color_diff = np.mean(np.abs(rgb_img - white_pixel)) / 255.0

    # Чем меньше отличие, тем лучше охват (условная метрика)
    coverage = 100 - (avg_color_diff * 100)
    return coverage


def calculate_white_balance(image_data):
    """
    Оценивает баланс белого по отклонению от нейтрального серого.
    Использует серые патчи для оценки баланса белого.
    Возвращает значение от 0 до 1, где:
    - 1.0 означает идеальный баланс белого
    - Значения < 1.0 указывают на отклонение от идеального баланса
    """
    # Извлекаем патчи из изображения
    patches = extract_patches(image_data)

    # Используем нижний ряд патчей (предполагается, что это серые патчи)
    gray_patches = patches[18:24]

    # Вычисляем среднее значение для каждого канала RGB
    mean_rgb = np.mean(gray_patches, axis=0)

    # Идеальный серый цвет должен иметь равные значения R, G и B
    # Вычисляем отклонение от идеального серого
    max_channel = np.max(mean_rgb)
    min_channel = np.min(mean_rgb)

    # Если все каналы равны, баланс белого идеальный (1.0)
    # Чем больше разница между каналами, тем хуже баланс белого
    if max_channel == 0:
        return 1.0

    # Нормализуем разницу между каналами
    channel_diff = (max_channel - min_channel) / max_channel

    # Преобразуем в оценку от 0 до 1
    white_balance_score = 1.0 - channel_diff

    return float(white_balance_score)


def calculate_contrast_ratio(image):
    """Рассчитывает контрастность изображения."""
    # Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Находим максимальное и минимальное значения яркости
    max_val = np.max(gray)
    min_val = np.min(gray)

    # Добавляем небольшую константу к минимальному значению, чтобы избежать деления на ноль
    min_val = max(min_val, 1)

    # Рассчитываем контрастность
    contrast_ratio = max_val / min_val

    return contrast_ratio
