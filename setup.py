from setuptools import setup, find_packages

VERSION = '0.1.0'

setup(
    name='pyvideocutter',
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'opencv-python',
        'numpy',
        'PyQt5'
    ],
    entry_points={
        'console_scripts': [
            'pyvideocutter = pyvideocutter.main:main'
        ]
    },
    author='Nikita Shubin',
    author_email='shubin.kit@ya.com',
    description='Инструмент для вырезания фрагментов видео',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='video editor ffmpeg',
    url='https://github.com/NikitaShubin/PyVideoCutter',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)