import cv2
import numpy as np


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
    Чем ближе к 1.0, тем лучше.
    """
    gray_img = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_img) / 255.0
    return avg_brightness


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
