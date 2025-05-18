# Рейтинг камер смартфонов

Telegram-бот для анализа качества камер телефонов по фото. Принимает изображения в виде документа с подписью/текстом модели телефона (пример: "Iphone 13"), анализирует их по заданным метрикам и выводит рейтинговую таблицу камер смартфонов.

## 🚀 Используемые технологии

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Aiogram](https://img.shields.io/badge/Aiogram-00ADEF?style=for-the-badge&logo=telegram&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-FF4F4F?style=for-the-badge&logo=sqlite&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## 📌 Начало работы

### 📋 Требования

- **Python 3.12+**
- [**uv**](https://docs.astral.sh/uv/#installation) (версия 0.6.9 или новее)
- **Токен телеграм-бота (.env файл)** (можно найти в группе проекта или написав в личные сообщения Хромых Илье)

### 🔧 Установка

1. **Склонируйте репозиторий:**
   ```sh
   git clone https://github.com/4uJustDev/methodologyCameraTesting.git
   cd methodologyCameraTesting
   ```

2. **Установите `uv`** (инструкции на [официальном сайте](https://docs.astral.sh/uv/#installation)).

3. **Установите зависимости:**
   ```sh
   uv sync
   ```

4. **Настройте `.env` файл** (скопируйте его в корень проекта. Не загружать в git!).

5. **Запустите бота:**
   ```sh
   uv run python main.py
   ```

---

## 📌 Использование

1. **Запустите бота командой** `uv run python main.py`.
2. **Найдите бота в Telegram:** `@Requiem4soul_bot`.
3. **Отправьте фото как документ с подписью** (модель телефона).
4. **Бот вернёт результаты анализа и рейтинговую таблицу.**

   


https://github.com/user-attachments/assets/316128d8-a0ac-447e-8986-5c06f8e528a7


---

## 📊 Как добавить свой метод оценки?

1. **Создайте новый файл в `image_analyz/metrics/`** (например, `contrast.py`).
2. **Добавьте функцию с префиксом `calculate_(название вашего метода)`:**
   ```python
   def calculate_contrast(image_data):
       # Логика анализа
       return float_value  # Оценка от 0 до 10
   ```
3. **В "data/models.py" добавьте в блоке свой метод, чтобы он был в БД**
   ```python
   class Rating(Base):
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True)
    phone_model = Column(String, nullable=False)
    sharpness = Column(Float, nullable=True)
    noise = Column(Float, nullable=True)
    glare = Column(Float, nullable=True)
    # Тут надо будет дополнять новыми метриками
    chromatic_aberration = Column(Float, nullable=True)
    vignetting = Column(Float, nullable=True)
    total_score = Column(Float, nullable=True)
   ```
3. **В "data/repository.py" в 3-ёх местах добавьте свой новый метод**
   ```python
               total_score = sum(metrics.values()) / len(metrics) if metrics else None
            rating = Rating(
                phone_model=new_phone_model,
                sharpness=metrics.get("sharpness"),
                noise=metrics.get("noise"),
                glare=metrics.get("glare"),
                # Тут добавляйте свой метод
                vignetting=metrics.get("vignetting"),
                chromatic_aberration=metrics.get("chromatic_aberration"),
                total_score=total_score
            )
   ```
   
   ```python
       def get_average_ratings(self):
        session = get_session()
        try:
            # Возвращаем все записи без усреднения, сортируя по total_score
            results = session.query(
                Rating.phone_model,
                Rating.sharpness,
                Rating.noise,
                Rating.glare,
                # Тут добавляйте свой метод
                Rating.chromatic_aberration,
                Rating.vignetting,
                Rating.total_score
            ).order_by(desc(Rating.total_score)).all()
            return [
   ```
   
   ```python
   return [
                {
                    "phone_model": r.phone_model,
                    "sharpness": r.sharpness,
                    "noise": r.noise,
                    "glare": r.glare,
                    # Тут добавляйте свой метод
                    "chromatic_aberration": r.chromatic_aberration,
                    "vignetting": r.vignetting,
                    "total_score": r.total_score
                }
   ```
5. **Перезапустите бота — метод автоматически подключится.**

### 🔹 Требования к методу:
- Принимает **NumPy-массив** (`image_data`).
- Возвращает **оценку от 0 (худшее) до 10 (лучшее)**.
- **Не используйте `scikit-image`** в реализации, но можно использовать для тестов

---

## 📦 Как добавить новую библиотеку?

1. **Добавьте библиотеку в зависимости:**
   ```sh
   uv add <нужная-библиотека>
   ```
2. **Обновите зависимости:**
   ```sh
   uv lock
   uv sync
   ```

---

4 group 244-321

Link for work board
https://github.com/4uJustDev?tab=projects
