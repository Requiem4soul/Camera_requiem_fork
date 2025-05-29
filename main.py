import asyncio
from aiogram import Dispatcher
from bot.telegram_bot import bot, router, set_commands


async def main():
    dp = Dispatcher()
    dp.include_router(router)

    try:
        await set_commands()
        # TODO: Добавить logger/loguru
        print("Бот запущен...")
        await dp.start_polling(bot)
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен.")
