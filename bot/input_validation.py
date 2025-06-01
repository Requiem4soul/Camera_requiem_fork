from aiogram.types import Message

# В данном файле идут проверки на корректность инпута от пользователя
# Если мы ожидаем сообщение или фото в виде документа для функции, то любой иной ввод не будет обработан
# Делается в избежании ошибок в боте и чтобы человек понял что нужно

def validate_text_only_input(message: Message) -> tuple[bool, str]:
    """
    Когда в сообщение должен быть текст и ничто больше. К примеру данная функция будет применяться к добавлению своей модели телефона
    """
    if not message.text:
        return False, "Пожалуйста, введите название модели телефона текстом (без фото, документов и других файлов)"

    if message.photo or message.document or message.video or message.audio or message.voice or message.sticker:
        return False, "Здесь нужно ввести только текст с названием модели телефона и ничего более"

    return True, ""


def validate_document_photo_only(message: Message) -> tuple[bool, str]:
    """
    Когда в сообщение должно быть только фото документом и ничего более. У нас для отправки фото для оценки методом
    """
    if message.text and message.text.strip():
        return False, "Вы ввели просто сообщение, но в данном методе нужно отправить только фото документом.\nДля этого: Прикрепить → Файл → Выбрать фото"

    if message.caption and message.caption.strip():
        return False, "Вы отправили подпись для вложения, но в данном методе нужно отправить только фото документом.\nДля этого: Прикрепить → Файл → Выбрать фото"

    if not message.document:
        if message.photo:
            return False, "Вы отправили обычное фото с сжатием, но в данном методе нужно отправить только фото документом.\nДля этого: Прикрепить → Файл → Выбрать фото"
        else:
            return False, "Отправьте фото документом.\nДля этого: Прикрепить → Файл → Выбрать фото"

    if not message.document.mime_type or not message.document.mime_type.startswith("image/"):
        return False, "Отправьте фото документом.\nДля этого: Прикрепить → Файл → Выбрать фото"

    return True, ""