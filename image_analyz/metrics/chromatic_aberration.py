import numpy as np
import cv2
import matplotlib.pyplot as plt
import tempfile

def calculate_chromatic_aberration(image_data):
    if image_data is None or image_data.size == 0:
        return {'chromatic_aberration': 0.0, 'aberration_chart': None}

    # Преобразуем из BGR (OpenCV) в RGB
    image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    # Разделяем каналы
    r = image_rgb[:, :, 0]
    g = image_rgb[:, :, 1]
    b = image_rgb[:, :, 2]

    # Детектируем границы на каждом канале
    def get_edges(channel):
        return cv2.Canny(channel, 100, 200)

    r_edges = get_edges(r)
    g_edges = get_edges(g)
    b_edges = get_edges(b)

    # Вычисляем разницу между каналами только на границах
    diff_rg = cv2.absdiff(r_edges, g_edges)
    diff_rb = cv2.absdiff(r_edges, b_edges)
    diff_gb = cv2.absdiff(g_edges, b_edges)

    # Создаем визуализацию
    plt.figure(figsize=(12, 8))
    
    # Визуализация разницы между каналами
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(diff_rg, cmap='hot')
    plt.title('Разница R-G каналов')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(diff_rb, cmap='hot')
    plt.title('Разница R-B каналов')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(diff_gb, cmap='hot')
    plt.title('Разница G-B каналов')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    
    # Сохраняем график во временный файл
    aberration_chart_path = None
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name, bbox_inches='tight', dpi=300)
        plt.close()
        aberration_chart_path = tmp.name

    # Вычисляем метрику аберрации
    aberration_score = (np.mean(diff_rg) + np.mean(diff_rb) + np.mean(diff_gb)) / 3.0
    normalized_score = np.clip(10 - (aberration_score * 0.1), 0, 10)  # Нормализуем к шкале 0-10

    return {
        'chromatic_aberration': float(normalized_score),
        'aberration_chart': aberration_chart_path
    }