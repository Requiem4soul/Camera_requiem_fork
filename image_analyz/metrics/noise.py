import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_psnr(image_data) -> float:
    #resized = cv2.resize(image_data, (722, 1280))
    #data = np.frombuffer(image_data, np.uint8)
    #image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    ideal = np.zeros((1280, 722, 3), dtype=np.float32)
    mse = np.mean((ideal - image_data) ** 2)
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
    return ideal, image_data, psnr, score_final

#if __name__ == "__main__":
    # Загрузка изображения
    #image_path = 'D:\photo_2025-05-29_20-38-12.jpg'  # Укажите путь к изображению
    #input_image = cv2.imread(image_path)

   # if input_image is None:
        #print("Ошибка: изображение не загружено.")
   # else:
       # ideal, image, psnr, score = calculate_psnr(input_image)

       # print(f"PSNR: {psnr:.2f} dB")
       # print(f"Качество изображения (0–10): {score}/10")
       # print(f"Размер изображения: {image.shape}")
     #   print(f"Размер эталонного (чёрного) изображения: {ideal.shape}")
