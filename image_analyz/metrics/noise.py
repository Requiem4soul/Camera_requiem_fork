import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_psnr(original: np.ndarray, noisy: np.ndarray) -> float:
    if original.shape != noisy.shape:
        raise ValueError("Изображения должны иметь одинаковый размер и количество каналов")

    mse = np.mean((original.astype(np.float32) - noisy.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def smooth_image(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def plot_histograms(original: np.ndarray, smoothed: np.ndarray, output_path: str):
    #Строит и сохраняет гистограммы яркости до и после сглаживания
    # Перевод в градации серого
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    smoothed_gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(10, 5))
    plt.hist(original_gray.ravel(), bins=256, range=[0, 256], alpha=0.6, label='Оригинал', color='blue')
    plt.hist(smoothed_gray.ravel(), bins=256, range=[0, 256], alpha=0.6, label='Сглажено', color='green')
    plt.title('Гистограммы яркости')
    plt.xlabel('Яркость (0-255)')
    plt.ylabel('Количество пикселей')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
