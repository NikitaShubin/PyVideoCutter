import os
import PyInstaller.__main__


def build_exe():
    # Пути к ресурсам
    icon_path = os.path.join('pyvideocutter', 'resources', 'icon.ico')
    main_script = os.path.join('pyvideocutter', 'main.py')

    # Параметры сборки
    args = [
        main_script,
        '--name=PyVideoCutter',
        '--onefile',
        '--windowed',
        '--icon=' + icon_path,
        '--add-data=' + os.path.join('pyvideocutter', 'resources') + os.pathsep + 'resources'
    ]

    # Сборка
    PyInstaller.__main__.run(args)


if __name__ == '__main__':
    build_exe()