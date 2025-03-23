from aiogram import Bot, Router, F
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command
from aiogram.enums import ContentType
from dotenv import load_dotenv
import os
import tempfile
import cv2
from image_analyz.analyzer import Image

# Токен ТОЛЬКО подгружать из env! Не менять вручную!
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден в .env файле! Укажи его в .env как TELEGRAM_BOT_TOKEN=your_token")

bot = Bot(token=TOKEN)

router = Router()


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

    img = Image(cv2.imread(file_path))
    img.analyze()

    # Формируем ответ
    response = f"Результаты анализа для модели {phone_model}: \n"
    for metric, value in img.metrics.items():
        response += f"{metric}: {value: .2f}\n"

    await message.reply(response)
    os.remove(file_path)


# TODO: Сохранение результатов и вывод их в таблицы (скорее всего через SQLite)
