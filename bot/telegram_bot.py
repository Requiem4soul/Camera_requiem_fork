from aiogram import Bot, Router, F
from aiogram.types import Message, FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command
from aiogram.enums import ContentType
from dotenv import load_dotenv
import os
import tempfile
import cv2
from image_analyz.analyzer import Image
from data.repository import RatingRepository

# Токен ТОЛЬКО подгружать из env! Не менять вручную!
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден в .env файле! Укажи его в .env как TELEGRAM_BOT_TOKEN=your_token")

bot = Bot(token=TOKEN)

router = Router()

repo = RatingRepository()

# Кнопки для удобства
def get_main_keyboard():
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Показать рейтинг камер", callback_data="show_ratings")],
        [InlineKeyboardButton(text="Инструкции", callback_data="show_instructions")]
    ])
    return keyboard

@router.message(Command(commands=["start"]))
async def send_welcome(message: Message):
    await message.reply("Привет! Отправь фото как документ с подписью (модель телефона), и я его проанализирую!")


@router.message(F.content_type == ContentType.DOCUMENT)
async def handle_photo(message: Message):
    """Обработка документа с фото и подписью."""
    if not (message.document and message.document.mime_type.startswith("image/")):
        await message.reply("Отправь изображение в виде документа!")
        return

    if not message.caption:
        await message.reply("Отправь фото !с подписью!, где указана модель телефона.")
        return

    phone_model = message.caption
    photo_file_id = message.document.file_id

    file_info = await bot.get_file(photo_file_id)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = temp_file.name
        await bot.download_file(file_info.file_path, file_path)

    try:
        img = Image(cv2.imread(file_path))
        img.analyze()

        # Работа с БД
        repo.add_rating(phone_model, img.metrics)

        # Формируем ответ
        response = f"Результаты анализа для модели {phone_model}: \n"
        for metric, value in img.metrics.items():
            response += f"{metric}: {value: .2f}\n"

        # Формируем ответ с результатами и таблицей
        response += "\nРезультаты сохранены!\n Используй /ratings, чтобы увидеть таблицу рейтинга камер.\n"

        await message.reply(response, reply_markup=get_main_keyboard())

    except Exception as e:
        await message.reply(f"Ошибка при анализе: {str(e)}")
    finally:
        os.remove(file_path)

@router.message(Command(commands=["ratings"]))
async def show_ratings(message: Message):
    """Вывод рейтинговой таблицы из БД."""
    try:
        table = repo.get_average_ratings()
        if not table:
            await message.answer("Рейтинговая таблица пуста. Отправь фото для анализа!", reply_markup=get_main_keyboard())
            return

        response = "Рейтинговая таблица:\n\n"
        for row in table:
            response += f"{row['phone_model']}:\n"
            # Исключаем phone_model и total_score из метрик
            metrics = {k: v for k, v in row.items() if k not in ['phone_model', 'total_score'] and v is not None}
            for metric, value in metrics.items():
                # Форматируем название метрики: убираем '_' и делаем первую букву заглавной
                metric_name = metric.replace('_', ' ').title()
                response += f"  {metric_name}: {value:.2f}\n"
            if row['total_score'] is not None:
                response += f"  Total Score: {row['total_score']:.2f}\n"
            response += "\n"

        await message.answer(response, reply_markup=get_main_keyboard())

    except Exception as e:
        await message.answer(f"Ошибка при получении рейтингов: {str(e)}", reply_markup=get_main_keyboard())


@router.callback_query(F.data == "show_ratings")
async def callback_show_ratings(callback):
    """Обработка кнопки 'Показать рейтинги'."""
    try:
        table = repo.get_average_ratings()
        if not table:
            await callback.message.answer("Рейтинговая таблица пуста. Отправь фото для анализа!", reply_markup=get_main_keyboard())
            await callback.answer()
            return

        response = "Рейтинговая таблица:\n\n"
        for row in table:
            response += f"{row['phone_model']}:\n"
            metrics = {k: v for k, v in row.items() if k not in ['phone_model', 'total_score'] and v is not None}
            for metric, value in metrics.items():
                metric_name = metric.replace('_', ' ').title()
                response += f"  {metric_name}: {value:.2f}\n"
            if row['total_score'] is not None:
                response += f"  Total Score: {row['total_score']:.2f}\n"
            response += "\n"

        await callback.message.answer(response, reply_markup=get_main_keyboard())
        await callback.answer()

    except Exception as e:
        await callback.message.answer(f"Ошибка при получении рейтингов: {str(e)}", reply_markup=get_main_keyboard())
        await callback.answer()

@router.callback_query(F.data == "show_instructions")
async def callback_show_instructions(callback):
    """Обработка кнопки 'Инструкции'."""
    await callback.message.answer(
        "Инструкции:\n"
        "1. Отправь фото как документ (Прикрепить -> Файл -> Выбрать фото).\n"
        "2. В подписи укажи модель телефона (например, 'iPhone 14').\n"
        "3. Я проанализирую фото и сохраню результаты.\n"
        "4. Используй /ratings для просмотра таблицы рейтингов.",
        reply_markup=get_main_keyboard()
    )
    await callback.answer()

@router.message()
async def handle_invalid_input(message: Message):
    """Обработка всех остальных случаев, кроме корректного файла с подписью."""
    await message.reply(
        "Пожалуйста, отправь фото как документ с подписью (модель телефона).\n"
        "Для любых устройств: Прикрепить -> Файл -> Выбрать нужное фото -> Ввести в поле текста модель телефона -> Отправить",
        reply_markup=get_main_keyboard()
    )



