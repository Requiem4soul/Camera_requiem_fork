import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_psnr(image_data: bytes, kernel_size: int = 5) -> float:
    data = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    ideal = np.zeros((1280, 722, 3), dtype=np.uint8)
    image = cv2.resize(image, (722, 1280))
    mse = np.mean((ideal.astype(np.float32) - image.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    if psnr == float('inf'):
        return 10.0

    # Ограничим PSNR диапазоном от 10 до 50 для нормализации
    psnr = max(min(psnr, 50), 10)

    # Линейная нормализация: 10 dB → 0 баллов, 50 dB → 10 баллов
    score = (psnr - 10) / (50 - 10) * 10
    score_final = round(score, 1)
    return ideal, image, psnr, core_final
