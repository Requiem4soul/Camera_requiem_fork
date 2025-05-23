from .metrics import get_metrics  # Импортируем зарегистрированные метрики


class Image:
    def __init__(self, image_data):
        self.image_data = image_data
        self.metrics = {}  # Словарь для результатов

    def analyze(self):
        # Получаем все доступные метрики
        metric_functions = get_metrics()
        # Применяем каждую метрику
        for name, func in metric_functions.items():
            result = func(self.image_data)
            if name == "vignetting" and isinstance(result, dict): # Просьба в данном блоке не менять ничего, или сообщить Хромых ИА об изменениях
                for subname, subvalue in result.items():
                    self.metrics[subname] = subvalue
            else:
                self.metrics[name] = result
