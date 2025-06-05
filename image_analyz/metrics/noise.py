import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile

def calculate_noise(image_data):
    """
    Анализирует уровень шума на изображении
    
    Возвращает:
    - noise_score: оценка шума (0-10, где 0 - нет шума, 10 - сильный шум)
    - aberration_chart: путь к файлу с визуализацией шума
    """
    if image_data is None or image_data.size == 0:
        return {
            'noise': 0.0,
            'aberration_chart': None
        }
    
    # Конвертируем в grayscale для анализа яркости
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    
    # Применяем фильтр для выделения шума (разница между оригиналом и размытой версией)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.absdiff(gray, blurred)
    
    # Вычисляем стандартное отклонение как меру шума
    noise_std = np.std(noise)
    
    # Нормализуем оценку шума (0-10)
    # Эмпирически подобранные значения для нормализации
    max_std = 25  # Значение, соответствующее сильному шуму
    noise_score = min(10, (noise_std / max_std) * 10)
    
    # Создаем визуализацию
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Оригинальное изображение
    ax1.imshow(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
    ax1.set_title('Оригинал')
    ax1.axis('off')
    
    # 2. Визуализация шума
    ax2.imshow(noise, cmap='gray', vmin=0, vmax=255)
    ax2.set_title(f'Выделенный шум\n(σ={noise_std:.1f})')
    ax2.axis('off')
    
    # 3. Гистограмма шума
    ax3.hist(noise.ravel(), bins=50, color='blue', alpha=0.7)
    ax3.set_title('Распределение шума')
    ax3.set_xlabel('Значение шума')
    ax3.set_ylabel('Частота')
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Сохраняем визуализацию во временный файл
    noise_vis_path = None
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name, bbox_inches='tight', dpi=100)
        plt.close()
        noise_vis_path = tmp.name
    
    return {
        'noise': float(noise_score),
        'aberration_chart': noise_vis_path
    }