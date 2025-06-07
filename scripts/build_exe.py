import os
import PyInstaller.__main__
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
print(base_dir)


def build_exe():
    # Пути к ресурсам
    icon_path = os.path.join(base_dir, 'pyvideocutter', 'resources', 'icon.ico')
    main_script = os.path.join(base_dir, 'pyvideocutter', 'main.py')

    # Параметры сборки
    args = [
        main_script,
        '--name=PyVideoCutter',
        '--onefile',
        '--console',
        '--icon=' + icon_path,
        '--add-data=' + os.path.join(base_dir, 'pyvideocutter', 'resources') + os.pathsep + 'resources',
        '--add-binary=/usr/lib/x86_64-linux-gnu/libc.so.6:.',
        '--strip',
    ]

    '''
    binaries = [
        ('/usr/lib/x86_64-linux-gnu/libGL.so.1', '.'),
        ('/usr/lib/x86_64-linux-gnu/libGLX.so.0', '.'),
        ('/usr/lib/x86_64-linux-gnu/libOpenGL.so.0', '.'),
    ]

    datas = [
        ('/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms', 'platforms'),
        ('/usr/lib/x86_64-linux-gnu/qt5/plugins/xcbglintegrations', 'xcbglintegrations'),
    ]

    for src, dst in binaries:
        args.append(f'--add-binary={src}:{dst}')
    for src, dst in datas:
        args.append(f'--add-data={src}:{dst}')
    '''

    # Компиляция:
    PyInstaller.__main__.run(args)


if __name__ == '__main__':
    build_exe()