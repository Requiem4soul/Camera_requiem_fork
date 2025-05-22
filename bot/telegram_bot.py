from aiogram import Bot, Router, F
from aiogram.types import (
    Message,
    FSInputFile,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BotCommand,
    BotCommandScopeDefault,
)
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
    raise ValueError(
        "TELEGRAM_BOT_TOKEN не найден в .env файле! Укажи его в .env как TELEGRAM_BOT_TOKEN=your_token"
    )

bot = Bot(token=TOKEN)

router = Router()

repo = RatingRepository()

ANALYSIS_METHODS = {
    "method1": "Метод 1 - chromatic_aberration",
    "method2": "Метод 2 - Виньетирование",
    "method3": "Метод 3 - Шум",
    "method4": "Метод 4 - Сверхрешётка",
}

user_methods = {}


# Кнопки для удобства
def get_main_keyboard():
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Показать рейтинг камер", callback_data="show_ratings"
                )
            ],
            [
                InlineKeyboardButton(
                    text="Выбрать метод анализа", callback_data="select_method"
                )
            ],
            [
                InlineKeyboardButton(
                    text="Инструкции", callback_data="show_instructions"
                )
            ],
        ]
    )
    return keyboard


def get_method_selection_keyboard():
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=name, callback_data=f"method_{method_id}")]
            for method_id, name in ANALYSIS_METHODS.items()
        ]
    )
    return keyboard


async def set_commands():
    commands = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="ratings", description="Показать рейтинг камер"),
        BotCommand(command="instructions", description="Инструкции"),
        BotCommand(command="select_method", description="Выбрать метод анализа"),
    ]
    await bot.set_my_commands(commands, scope=BotCommandScopeDefault())


@router.message(Command(commands=["start"]))
async def send_welcome(message: Message):
    user_id = message.from_user.id
    current_method = user_methods.get(user_id)

    welcome_text = "Привет! Отправь фото как документ с подписью (модель телефона), и я его проанализирую!\n"

    if current_method:
        welcome_text += f"Текущий метод анализа: {ANALYSIS_METHODS[current_method]}\n"
    else:
        welcome_text += (
            "Сначала выбери метод анализа с помощью команды /select_method\n"
        )

    await message.reply(welcome_text, reply_markup=get_main_keyboard())


@router.message(F.content_type == ContentType.DOCUMENT)
async def handle_photo(message: Message):
    """Обработка документа с фото и подписью."""
    user_id = message.from_user.id

    if user_id not in user_methods:
        await message.reply(
            "Сначала выбери метод анализа! Нажми кнопку 'Выбрать метод анализа'",
            reply_markup=get_main_keyboard(),
        )
        return

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
        repo.add_rating(phone_model, img.metrics, user_methods[user_id])

        # Формируем ответ
        response = f"Результаты анализа для модели {phone_model} (Метод: {ANALYSIS_METHODS[user_methods[user_id]]}):\n"
        for metric, value in img.metrics.items():
            response += f"{metric}: {value: .2f}\n"

        response += "\nРезультаты сохранены!\n Используй /ratings, чтобы увидеть таблицу рейтинга камер.\n"

        await message.reply(response, reply_markup=get_main_keyboard())

    except Exception as e:
        await message.reply(f"Ошибка при анализе: {str(e)}")
    finally:
        os.remove(file_path)


@router.message(Command(commands=["ratings"]))
async def show_ratings(message: Message):
    """Вывод рейтинговой таблицы из БД."""
    await message.answer(
        "Выберите метод по которому хотите просмотреть данные:",
        reply_markup=get_method_selection_keyboard(),
    )


@router.callback_query(F.data == "show_ratings")
async def callback_show_ratings(callback):
    """Обработка кнопки 'Показать рейтинги'."""
    await callback.message.answer(
        "Выберите метод по которому хотите просмотреть данные:",
        reply_markup=get_method_selection_keyboard(),
    )
    await callback.answer()


@router.message(Command(commands=["select_method"]))
async def select_method(message: Message):
    """Выбор метода анализа."""
    await message.answer(
        "Выбери метод анализа:", reply_markup=get_method_selection_keyboard()
    )


@router.callback_query(F.data.startswith("method_"))
async def callback_method_selected(callback):
    """Обработка выбора метода анализа."""
    method_id = callback.data.replace("method_", "")
    user_id = callback.from_user.id

    if method_id in ANALYSIS_METHODS:
        # Проверяем, пришел ли запрос от кнопки рейтингов
        if callback.message.text and "просмотреть данные" in callback.message.text:
            # Показываем рейтинги для выбранного метода
            try:
                table = repo.get_average_ratings(method_id)
                if not table:
                    await callback.message.answer(
                        "Рейтинговая таблица пуста. Отправь фото для анализа!",
                        reply_markup=get_main_keyboard(),
                    )
                    await callback.answer()
                    return

                response = (
                    f"Рейтинговая таблица (Метод: {ANALYSIS_METHODS[method_id]}):\n\n"
                )
                for row in table:
                    response += f"{row['phone_model']}:\n"
                    metrics = {
                        k: v
                        for k, v in row.items()
                        if k not in ["phone_model", "total_score"] and v is not None
                    }
                    for metric, value in metrics.items():
                        metric_name = metric.replace("_", " ").title()
                        response += f"  {metric_name}: {value:.2f}\n"
                    if row["total_score"] is not None:
                        response += f"  Total Score: {row['total_score']:.2f}\n"
                    response += "\n"

                await callback.message.answer(
                    response, reply_markup=get_main_keyboard()
                )
            except Exception as e:
                await callback.message.answer(
                    f"Ошибка при получении рейтингов: {str(e)}",
                    reply_markup=get_main_keyboard(),
                )
        else:
            # Устанавливаем метод для анализа
            user_methods[user_id] = method_id
            await callback.message.answer(
                f"Выбран метод: {ANALYSIS_METHODS[method_id]}\n"
                "Теперь можешь отправлять фото для анализа!"
            )
    else:
        await callback.message.answer("Ошибка выбора метода. Попробуй еще раз.")
    await callback.answer()


@router.callback_query(F.data == "show_instructions")
async def callback_show_instructions(callback):
    """Обработка кнопки 'Инструкции'."""
    await callback.message.answer(
        "Инструкции:\n"
        "1. Выбери метод анализа с помощью команды /select_method.\n"
        "2. Отправь фото как документ (Прикрепить -> Файл -> Выбрать фото).\n"
        "3. В подписи укажи модель телефона (например, 'iPhone 14').\n"
        "4. Я проанализирую фото и сохраню результаты.\n"
        "5. Используй /ratings для просмотра таблицы рейтингов."
    )
    await callback.answer()


@router.message()
async def handle_invalid_input(message: Message):
    """Обработка всех остальных случаев, кроме корректного файла с подписью."""
    await message.reply(
        "Пожалуйста, сначала выбери метод анализа, а затем отправь фото как документ с подписью (модель телефона).\n"
        "Для любых устройств: Прикрепить -> Файл -> Выбрать нужное фото -> Ввести в поле текста модель телефона -> Отправить",
        reply_markup=get_main_keyboard(),
    )
