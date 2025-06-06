# [PyVideoCutter](https://github.com/NikitaShubin/PyVideoCutter)

![PyVideoCutter Demo](demo.gif)

Инструмент для выбора и экспорта фрагментов видео с поддержкой:
- Превью видео с навигацией по кадрам
- Разметки фрагментов с визуализацией
- Экспорта сегментов через FFmpeg
- Горячих клавиш для эффективной работы

## Установка

### Как Python-пакет
```bash
pip install git+https://github.com/NikitaShubin/PyVideoCutter.git
```

### Из исходников
```bash
git clone https://github.com/NikitaShubin/PyVideoCutter.git
cd PyVideoCutter
pip install .
```

## Использование

### Графический интерфейс
```bash
pyvideocutter [опции]
```

Опции:
- `--source`: Исходное видео (обязательно)
- `--preview`: Видео для превью (по умолчанию = source)
- `--fragments_dir`: Папка для экспорта

Пример:
```bash
pyvideocutter --source input.mp4 --fragments_dir ./fragments
```

### Консольный экспорт
```bash
pyvideocutter --export --source input.mp4 --fragments_dir ./fragments
```

## Горячие клавиши
| Команда | Клавиши |
|---------|---------|
| Следующий кадр | →, >, . |
| Предыдущий кадр | ←, <, , |
| Пауза/воспроизведение | Пробел |
| Скорость (2^n кадров/шаг) | 0-9 |
| Переключить направление | R |
| Переход к сегменту | J |
| Начало сегмента | ↑, [ |
| Конец сегмента | ↓, ] |
| Удалить сегмент | Del, D |
| Отмена действия | Ctrl+Z |
| Восстановление действия | Z |
| Режим подгонки изображения | F |
| Статусбар | Tab |
| Экспорт | E |
| Справка | H, F1 |
| Выход | Esc, Q |

## Сборка исполняемого файла

1. Установите зависимости:
```bash
pip install pyinstaller
```

2. Выполните сборку:
```bash
python scripts/build_exe.py
```

Готовый исполняемый файл будет находиться в папке `dist`.

## Требования
- Python 3.6+
- FFmpeg (должен быть доступен в PATH)
- OpenCV
- PyQt5
- NumPy

## Лицензия
Проект распространяется под лицензией MIT. Подробнее см. [LICENSE](https://github.com/NikitaShubin/PyVideoCutter/blob/main/LICENSE).