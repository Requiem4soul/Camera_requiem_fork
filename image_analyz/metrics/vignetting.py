import cv2
import numpy as np
from scipy.optimize import curve_fit

def calculate_vignetting(image):
    """Оценивает виньетирование с помощью полярного преобразования и модели.
    Вход: изображение как np.ndarray (RGB или grayscale).
    Выход: оценка виньетирования от 0 (нет) до 10 (сильное).
    """
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image.copy()

    img = cv2.GaussianBlur(img, (9, 9), 0)  # Фильтрация шума

    h, w = img.shape
    img_4ch = np.dstack([img, np.full((h, w), 255, dtype=np.uint8)])
    rmax = np.hypot(h, w) / 2
    cx, cy = (w - 1) / 2, (h - 1) / 2
    dh, dw = 360 * 2, 1000

    warped = np.zeros((dh, dw, 2), dtype=np.uint8)
    cv2.warpPolar(dst=warped, src=img_4ch, dsize=(dw, dh), center=(cx, cy),
                  maxRadius=int(rmax), flags=cv2.INTER_LANCZOS4)

    values = warped[..., 0]
    mask = warped[..., 1]
    mvalues = np.ma.masked_array(values, mask=(mask == 0))
    mean_intensity = mvalues.mean(axis=0)

    # Модель виньетирования
    def vignette_model(r, I0, k):
        r_norm = r / r.max()
        return I0 * (1 - k * r_norm**2)

    radii = np.arange(dw)
    popt, _ = curve_fit(vignette_model, radii, mean_intensity, p0=[255, 0.1])
    I0, k = popt
    score = min(max(k * 20, 0), 10)  # k=0 -> 0, k=0.5 -> 10

    return score
