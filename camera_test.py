import cv2

# Подключение к камере (0 — это индекс камеры по умолчанию)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Камера не подключена!")
    exit()

# Захват одного кадра
ret, frame = cap.read()

if ret:
    # Сохранение кадра в файл
    cv2.imwrite("./public/camera/test_image.jpg", frame)
    print("Изображение сохранено как test_image.jpg")
else:
    print("Ошибка: Не удалось захватить изображение!")

# Освобождение ресурсов
cap.release()
