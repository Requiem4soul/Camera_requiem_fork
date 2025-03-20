import numpy as np


def test_colors(image):
    """Тест цветопередачи."""
    # Преобразуем изображение в float и нормализуем его
    image_float = image.astype(np.float32) / 255.0

    # Вычисляем средний цвет по всем пикселям
    mean_color = np.mean(image_float, axis=(0, 1))

    return f"Средний цвет: {mean_color.tolist()}"
