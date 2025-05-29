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
    print(psnr)
    return ideal, image, psnr
