from typing import Dict, Callable
import importlib
import os
import glob


def get_metrics() -> Dict[str, Callable]:
    metrics = {}
    # Получаем все .py файлы в папке metrics, кроме __init__.py
    metric_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))

    for file_path in metric_files:
        file_name = os.path.basename(file_path)
        if file_name != "__init__.py":
            module_name = f"image_analyz.metrics.{file_name[:-3]}"  # Убираем .py
            module = importlib.import_module(module_name)
            # TODO: Тут можно будет организовать перевод
            for attr_name in dir(module):
                if attr_name.startswith("calculate_"):
                    metric_name = attr_name.replace("calculate_", "")
                    metrics[metric_name] = getattr(module, attr_name)

    return metrics
