import cv2
import matplotlib.pyplot as plt
from tests.sharpness import test_sharpness
from tests.noise import test_noise
from tests.colors import test_colors


def capture_image():
    """Захват изображения с камеры."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Не удалось захватить изображение с камеры!")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Конвертация в RGB


def display_results(image, results):
    """Отображение изображения и результатов тестов."""
    plt.figure(figsize=(10, 6))

    # Отображение изображения
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Захваченное изображение")
    plt.axis("off")

    # Отображение результатов тестов
    plt.subplot(1, 2, 2)
    plt.text(0.1, 0.6, "\n".join(results), fontsize=12, va="top")
    plt.axis("off")

    plt.show()


def main():
    # Захват изображения
    image = capture_image()

    # Запуск тестов
    results = []
    results.append(f"Резкость: {test_sharpness(image):.2f}")
    results.append(f"Уровень шума: {test_noise(image):.2f}")
    results.append(f"Цветопередача: {test_colors(image)}")

    # Отображение результатов
    display_results(image, results)


if __name__ == "__main__":
    main()
