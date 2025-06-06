import sys
import os

import copy
import argparse
import numpy as np
import cv2
import base64

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout,
    QMessageBox, QFileDialog, QStatusBar, QProgressDialog
)
from PyQt5.QtGui import QImage, QPixmap, QKeyEvent, QIcon
from PyQt5.QtCore import Qt, QTimer

# Загружаем иконку приложения в base64 из другого файла:
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(base_dir))
from pyvideocutter.icon import ICO_BASE64


class QtDialogHelper:
    """Класс-помощник для диалоговых окон Qt:"""

    @staticmethod
    def msgbox(msg, title='Внимание', parent=None):
        msg_box = QMessageBox(parent)
        msg_box.setWindowTitle(title)
        msg_box.setText(msg)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()

    @staticmethod
    def errorbox(msg, title='Ошибка', parent=None):
        msg_box = QMessageBox(parent)
        msg_box.setWindowTitle(title)
        msg_box.setText(msg)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.exec_()

    @staticmethod
    def ask_open_file(title=None, parent=None):
        return QFileDialog.getOpenFileName(
            parent,
            title or "Выберите файл",
            "",
            "All Files (*);;Video Files (*.mp4 *.avi *.mov)"
        )[0]

    @staticmethod
    def ask_dir(title=None, parent=None):
        return QFileDialog.getExistingDirectory(
            parent,
            title or "Выберите папку"
        )

    @staticmethod
    def ask_ok_cancel(question, title=None, parent=None):
        msg_box = QMessageBox(parent)
        msg_box.setWindowTitle(title or "Подтверждение")
        msg_box.setText(question)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        return msg_box.exec_() == QMessageBox.Ok


class CircleInd:
    """
    Целое число с замкнутым инкриментом/декриментом.
    Полезно для круговой адресации к элементам массива:
    """

    def __init__(self, circle, ind=0):
        assert 0 <= ind < circle
        self.circle = circle
        self.ind = ind

    def inc(self):
        self.ind += 1
        if self.ind == self.circle:
            self.ind = 0
        return self.ind

    def dec(self):
        self.ind -= 1
        if self.ind == -1:
            self.ind = self.circle - 1
        return self.ind

    def __int__(self):
        return self.ind

    __call__ = __int__

    def __eq__(self, other):
        return self.ind == int(other)

    def __ne__(self, other):
        return self.ind != int(other)


class VideoReader:
    """
    Читатель видеофайла с буфером кадров:
    """

    def __init__(self, vi_file, buf_size=512):
        # Открываем видеофайл и фиксируем некоторые его параметры:
        self.cap = cv2.VideoCapture(vi_file)
        if not self.cap.isOpened():
            raise ValueError('Ошибка открытия файла "%s"' % vi_file)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Заполняем буфер кадрами:
        self.buf_size = min(buf_size, self.total_frames)
        self.buffer = np.zeros((self.buf_size, self.h, self.w, 3), np.uint8)
        for frame_ind in range(self.buf_size):
            self.buffer[frame_ind, ...] = self.cap.read()[1]

        # Расставляем переменные адресации кадров в буфере:
        self.start = CircleInd(self.buf_size)
        self.cur = CircleInd(self.buf_size)
        self.end = CircleInd(self.buf_size, self.buf_size - 1)

        # Номер последнего кадра, хранящегося в буфере:
        self.frame_ind = self.buf_size - 1

        # Номер текущего кадра в видеопоследовательности:
        self.position = 0

    # Возвращает текущий кадр:
    def current(self):
        return self.buffer[self.cur(), ...]

    # Возвращает следующий кадр:
    def next(self, get_frame=True):
        # Если следующий кадр уже есть в буфере:
        if self.cur != self.end:
            self.cur.inc()
            self.position += 1
            return self.current() if get_frame else None

        # Если текущий кадр является последним, возвращаем пустоту:
        elif self.frame_ind == self.total_frames - 1:
            self.close()  # Закрываем видеофайл
            return

        # Если есть возможность взять следующий кадр, то берём его:
        else:
            self.frame_ind += 1
            self.start.inc()
            self.cur.inc()
            self.end.inc()
            new_frame = self.cap.read()[1]
            self.buffer[self.cur(), ...] = new_frame
            self.position += 1
            return new_frame if get_frame else None

    def prev(self, get_frame=True):
        # Если курсор уже на самом раннем кадре буфера или мы на первом кадре
        # исходной видеопоследовательности, возвращаем пустоту:
        if self.cur == self.start or not self.position:
            return

        # Если предыдущий кадр есть в буфере, берём его:
        else:
            self.cur.dec()
            self.position -= 1
            return self.current()

    # Закрываем открытый видеофайл:
    def close(self):
        self.cap.release()


class Backend:
    """
    База данных списка фрагментов видеопоследовательности:
    """

    def __init__(self,
                 total_frames,
                 source_video_file,
                 out_dir):
        self.fragments = []
        self.total_frames = total_frames
        self.source_video_file = source_video_file

        # Раскладываем путь к исходному видеофайлу на составляющие:
        source_dir, source_basename = os.path.split(source_video_file)
        source_name, source_ext = os.path.splitext(source_basename)

        # Файл списка фрагментов размещаем в папке с фрагментами:
        self.fragments_file = self.target_prefix = \
            os.path.join(out_dir, source_name + '_fragments.txt')

        # Определяем префикс и суффикс имён фрагментов:
        self.target_prefix = os.path.join(out_dir, source_name)
        self.target_suffix = source_ext

        # Инициируем список фрагментов и историю его изменений:
        self.load()
        self.hist_pos = 0
        self.init_hist()

    # Инициализация истории операций и синхронизации с файлом состояния:
    def init_hist(self):
        # Если файл с сегментами существует, то читаем его:
        if os.path.isfile(self.fragments_file):
            self.load()
        # Если файла нет, то инициируем новый список сегментов:
        else:
            self.fragments = []
            self.save()

        # Переменные для историй операций:
        self.history = [copy.deepcopy(self.fragments)]
        self.hist_pos = 0

    # Внесение изменений в историю операций:
    def update_hist(self, sync=True):
        self.history = self.history[: self.hist_pos + 1] + \
                       [copy.deepcopy(self.fragments)]
        self.hist_pos = len(self.history) - 1
        if sync:
            self.save()

    # Отмена последней операции:
    def undo(self, sync=True):
        if self.hist_pos:
            self.hist_pos -= 1
            self.fragments = copy.deepcopy(self.history[self.hist_pos])
            if sync:
                self.save()
            return True
        else:
            return False

    # Возвращение отменённой операции:
    def redo(self, sync=True):
        next_hist_pos = self.hist_pos + 1
        if next_hist_pos == len(self.history):
            return False
        else:
            self.hist_pos = next_hist_pos
            self.fragments = copy.deepcopy(self.history[self.hist_pos])
            if sync:
                self.save()
            return True

    # Сохранение текущего состояния в файл:
    def save(self):
        with open(self.fragments_file, 'w') as f:
            for start, end in self.fragments:
                f.write(f'{start} {end}\n')

    # Загрузка состояния из файла:
    def load(self):
        self.fragments = []
        if os.path.isfile(self.fragments_file):
            with open(self.fragments_file, 'r') as f:
                for line in f.readlines():
                    fragment = list(map(int, line.split(' ')))
                    self.fragments.append(fragment)

    # Проверка корректности новой позиции:
    def check(self, position):
        return 0 <= position < self.total_frames

    def extract_fragments(self, parent=None):

        if parent:
            progress = QProgressDialog("Обработка фрагментов...", "Отмена",
                                       0, len(self.fragments), parent)
            progress.setWindowTitle("Экспорт фрагментов")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
        else:
            print(f"Экспорт {len(self.fragments)} фрагментов:")

        for fragment_ind, (start, end) in enumerate(self.fragments, 1):

            if parent:
                progress.setValue(fragment_ind)
                if progress.wasCanceled():
                    break
            else:
                print(f'[{fragment_ind}/{len(self.fragments)}] '
                      f'Кадры {start}-{end}.')

            # Формируем путь до сохраняемого фрагмента:
            target_file = self.target_prefix + '_fragment_' + \
                          str(fragment_ind) + self.target_suffix

            # Формируем путь до лог-файла:
            log_file = self.target_prefix + '_fragment_' + \
                       str(fragment_ind) + '.log'

            # Выполняем вырезание:
            os.system(f'ffmpeg -i "{self.source_video_file}" ' +
                      '-y -avoid_negative_ts make_zero ' +
                      '-vf mpdecimate,setpts=N/FRAME_RATE/TB,' +
                      fr'select="between(n\,{start}\,{end + 1}),' +
                      'setpts=PTS-STARTPTS" -an -c:v libx264 ' +
                      '-preset slow -crf 15 -tune animation ' +
                      f'"{target_file}"')
            # ' >{log_file} 2>&1')

    def start_frame2sb_axis(self, width, frame):
        return np.fix((width - 1) * frame / self.total_frames).astype(int)

    def end_frame2sb_axis(self, width, frame):
        return self.start_frame2sb_axis(width, frame) + 1

    def draw_statusbar(self, width, height, position=0, select_range=None):
        sb = np.zeros((height, width, 3), np.uint8)

        # Рисуем фрагменты на линии времени:
        sb[:, :, 1] = 255
        for start_frame, end_frame in self.fragments:
            start = self.start_frame2sb_axis(width, start_frame)
            end = self.end_frame2sb_axis(width, end_frame)
            sb[:, start:end, 2] = 255
            sb[:, start:end, 1] = 0

        # Прошедшие кадры отмечаем зелёным:
        current_shift = int(width * position / self.total_frames)
        sb[:, current_shift:, :] //= 2

        # Рисуем выбранный диапазон, если надо:
        if select_range is not None:
            start = self.start_frame2sb_axis(width, select_range[0])
            end = self.end_frame2sb_axis(width, select_range[1])
            sb[:, start:end, 0] = 255

        return sb

    # Вносим новый фрагмент, если нет наложения с уже имеющимими:
    def add(self, fragment):
        new_start, new_end = fragment
        new_ind = 0
        for ind, (start, end) in enumerate(self.fragments):
            if new_start > end:
                new_ind = ind + 1
            elif new_end < start:
                new_ind = ind
                break
            else:
                return False

        # Вставляем новый фрагмент в нужное место:
        self.fragments = self.fragments[:new_ind] + [fragment] + \
                         self.fragments[new_ind:]
        self.update_hist()

        return True

    # Изменяем начало следующего фрагмента:
    def new_start(self, position):
        for ind, (start, end) in enumerate(self.fragments):
            if end >= position:
                self.fragments[ind] = [position, end]
                break
        else:
            self.fragments.append([position, self.total_frames - 1])
        self.update_hist()
        return True

    # Изменяем конец предыдущего фрагмента:
    def new_end(self, position):
        for ind, (start, end) in reversed(list(enumerate(self.fragments))):
            if start <= position:
                self.fragments[ind] = [start, position]
                break
        else:
            self.fragments.append([0, position])
        self.update_hist()
        return True

    def delete(self, position):
        # Если удалять нечего:
        if len(self.fragments) == 0:
            return False

        # Если в этом месте есть однокадровый фрагмент, то удаляем именно его:
        point = [position, position]
        if point in self.fragments:
            self.fragments.remove(point)
            self.update_hist()
            return True

        # Если фрагмент для удаления придётся искать:
        else:
            # Ищем номера всех фрагментов, один из КРАЁВ которых попадает на
            # текущий кадр:
            fragments2del = [_ for _ in self.fragments if position in _]

            # Если найден лишь один такой сегмент, то его и удаляем:
            if len(fragments2del) == 1:
                self.fragments.remove(fragments2del[0])
                self.update_hist()
                return True

            # Если таковых два, то просим уточнить:
            elif len(fragments2del) == 2:
                QtDialogHelper.errorbox(
                    'Текущий кадр находится на стыке двух фрагментов. ' +
                    'Переместите курсор и повторите действие.'
                )
                return False

            # Если ни одного краевого фрагмента не найдено,
            # то будем искать уже промежутки:
            elif len(fragments2del) == 0:

                # Если текущая позиция стоит раньше самого первого фрагмента,
                # то ставим начало этого фрагмента в начало
                # видеопоследовательности:
                if position < self.fragments[0][0]:
                    self.fragments[0][0] = 0
                    self.update_hist()
                    return True

                # Если текущая позиция стоит после самого последнего
                # фрагмента, то ставим конец этого фрагмента в конец
                # видеопоследовательности:
                elif position > self.fragments[-1][1]:
                    self.fragments[-1][1] = self.total_frames - 1
                    self.update_hist()
                    return True

                # Удаляем фрагмент (или пропуск между фрагментами), в центре
                # которого находится текущая позиция:
                fragments = np.array(self.fragments).flatten()
                for ind in range(len(fragments) - 1):
                    if fragments[ind] < position < fragments[ind + 1]:
                        fragments = np.delete(fragments, [ind, ind + 1])
                        fragments = fragments.reshape(-1, 2)
                        self.fragments = []
                        for fragment in fragments:
                            self.fragments.append(list(map(int, fragment)))
                        self.update_hist()
                        return True
                else:
                    raise ValueError('Ошибка логики программы!')

            else:
                raise ValueError('Ошибка в списках фрагментов!')

    def selected_frames(self):
        return sum([end - start for start, end, *_ in self.fragments])

    def get_all_key_frames(self):
        """Возвращает все ключевые кадры (начала и концы фрагментов)"""
        key_frames = set()
        for start, end in self.fragments:
            key_frames.add(start)
            key_frames.add(end)
        return sorted(key_frames)


class MainWindow(QMainWindow):
    """Главное окно приложения с использованием PyQt5:"""

    # Горячие клавиши:
    next_frame_keys = [Qt.Key_Right, Qt.Key_Period, Qt.Key_Greater]
    prev_frame_keys = [Qt.Key_Left, Qt.Key_Comma, Qt.Key_Less]
    pause_key = Qt.Key_Space
    jump_keys = [Qt.Key_J]
    start_fragment_keys = [Qt.Key_Up, Qt.Key_BracketLeft]
    end_fragment_keys = [Qt.Key_Down, Qt.Key_BracketRight]
    delete_keys = [Qt.Key_Delete, Qt.Key_D]
    insert_fragment_edge_keys = [Qt.Key_K, Qt.Key_Insert, Qt.Key_F12]
    undo_redo_keys = [Qt.Key_Z]
    drop_fragment_edge_keys = [Qt.Key_Escape, Qt.Key_Q]
    fit_to_window_keys = [Qt.Key_F]
    reverse_play_keys = [Qt.Key_R]
    show_statusbar_key = Qt.Key_Tab
    export_fragments_keys = [Qt.Key_E, Qt.Key_P]
    show_hotkeys_info_key = [Qt.Key_H, Qt.Key_F1]
    speed_keys = list(range(Qt.Key_0, Qt.Key_9 + 1))

    def __init__(self, source_file, preview_file, fragments_dir=None):
        super().__init__()

        # Устанавливаем иконку приложения:
        self.setWindowIcon(self.create_icon_from_base64(ICO_BASE64))

        # Инициализация переменных состояния:
        self.show_statusbar = True
        self.show_hotkeys_info = False
        self.speed = 1
        self.jump_to = None
        self.key_pose = None
        self.is_playing = False
        self.fit_to_window = True
        self.play_direction = 1  # 1 - вперед, -1 - назад

        # Создаем VideoReader и Backend:
        self.video_reader = VideoReader(preview_file)
        self.backend = Backend(
            self.video_reader.total_frames,
            source_file,
            out_dir=fragments_dir
        )

        # Настройка UI:
        self.setWindowTitle("Fragmentator")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1, 1)  # Разрешаем уменьшение размера
        self.layout.addWidget(self.image_label)

        # Статус бар:
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Таймер для воспроизведения видео:
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Обновление каждые 30 мс

        # Загрузка первого кадра:
        self.update_frame()

        # Устанавливаем размер окна:
        self.resize(800, 600)

    def create_icon_from_base64(self, base64_str):
        """Создает иконку из base64 строки:"""
        icon_data = base64.b64decode(ICO_BASE64)

        # Создаем QPixmap из бинарных данных:
        pixmap = QPixmap()
        pixmap.loadFromData(icon_data)

        # Создаем QIcon из QPixmap:
        return QIcon(pixmap)

    def update_title(self):
        """Обновляет заголовок окна:"""
        source_video_file = os.path.basename(self.backend.source_video_file)
        title = f'"{source_video_file}"'

        # Добавляем информацию о кадрах:
        title += f"\tframe: {self.video_reader.position} / {self.video_reader.total_frames} "
        title += f"({100 * self.video_reader.position / self.video_reader.total_frames:.2f} %)"

        # Добавляем информацию о выбранных кадрах:
        selected = self.backend.selected_frames()
        total = self.video_reader.total_frames
        title += f"\tselected frames: {selected} / {total} "
        title += f"({100 * selected / total:.2f} %)"

        # Добавляем информацию о направлении воспроизведения:
        if self.play_direction == -1:
            title += " [REV]"

        self.setWindowTitle(title)

    def _draw_frame(self, frame=None):
        """Отрисовывает изображение в окне:"""

        # Получаем текущий кадр, если он явно не задан:
        if frame is None:
            frame = self.video_reader.current()

        # Конвертируем кадр OpenCV в QImage:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width

        # Если нужно добавить статусбар:
        if self.show_statusbar:
            # Создаем изображение с дополнительным местом для статусбара:
            status_height = int(height / 30)
            combined = np.zeros((height + status_height, width, 3),
                                dtype=np.uint8)
            combined[:height, :, :] = frame

            # Отрисовываем статусбар:
            position = self.video_reader.position
            select_fragment = sorted([self.key_pose, position]) \
                if self.key_pose else None
            statusbar = self.backend.draw_statusbar(width, status_height,
                                                    position, select_fragment)

            # Добавляем статусбар к изображению:
            combined[height:, :, :] = statusbar

            # Конвертируем комбинированное изображение:
            height, width, channel = combined.shape
            bytes_per_line = 3 * width
            q_img = QImage(combined.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Если достаточно самого изображения:
        else:
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Отображаем изображение:
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.IgnoreAspectRatio if self.fit_to_window else Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def update_frame(self):
        """Обновляет отображаемый кадр:"""
        if not self.is_playing:
            return

        # Получаем новый кадр:
        frame = None
        for _ in range(self.speed):

            # Читаем очередной кадр в зависимости от направления
            # воспроизведения:
            if self.play_direction == 1:
                frame = self.video_reader.next()
            else:
                frame = self.video_reader.prev()

            if frame is None or self.video_reader.position == self.jump_to:
                self.jump_to = None
                self.is_playing = False
                break

        # Отрисовываем изображение в окне:
        self._draw_frame(frame)

        # Обновляем заголовок:
        self.update_title()

    def keyPressEvent(self, event: QKeyEvent):
        """Обработка нажатий клавиш:"""
        key = event.key()
        modifiers = event.modifiers()

        # Переключение режима подгонки изображения:
        if key in self.fit_to_window_keys:
            self.fit_to_window = not self.fit_to_window
            self.update_frame_directly()

        # Переключение направления воспроизведения:
        if key in self.reverse_play_keys:
            self.play_direction *= -1
            self.jump_to = None
            direction = "назад" if self.play_direction == -1 else "вперед"
            self.status_bar.showMessage(f"Направление воспроизведения: {direction}", 2000)
            self.update_title()

        # Установка скорости воспроизведения:
        if key in self.speed_keys:
            n = key - Qt.Key_0
            self.speed = 2 ** n
            self.status_bar.showMessage(f"Скорость: {self.speed} кадров/шаг", 2000)

        # Переход на следующий кадр:
        if key in self.next_frame_keys:
            self.jump_to = None
            self.is_playing = False
            frame = self.video_reader.next()
            if frame is not None:
                self.update_frame_directly(frame)

        # Возврат на предыдущий кадр:
        elif key in self.prev_frame_keys:
            self.jump_to = None
            self.is_playing = False
            frame = self.video_reader.prev()
            if frame is not None:
                self.update_frame_directly(frame)

        # Пауза/воспроизведение:
        elif key == self.pause_key:
            self.is_playing = not self.is_playing

        # Переход к ближайшему сегменту (в любом направлении):
        elif key in self.jump_keys:
            position = self.video_reader.position
            key_frames = self.backend.get_all_key_frames()

            # Добавляем начало и конец всего видео:
            key_frames.insert(0, 0)
            key_frames.append(self.video_reader.total_frames - 1)

            # Находим ближайший ключевой кадр в текущем направлении:
            if self.play_direction == 1:  # Вперед
                # Ищем следующий ключевой кадр:
                for frame in key_frames:
                    if frame > position:
                        self.jump_to = frame
                        break
                else:
                    # Если ничего не найдено, переходим к последнему:
                    self.jump_to = key_frames[-1]
            else:  # Назад
                # Ищем предыдущий ключевой кадр:
                for frame in reversed(key_frames):
                    if frame < position:
                        self.jump_to = frame
                        break
                else:
                    # Если ничего не найдено, переходим к первому:
                    self.jump_to = key_frames[0]
            self.is_playing = True

        # Начало сегмента:
        elif key in self.start_fragment_keys:
            position = self.video_reader.position
            self.backend.new_start(position)
            self.update_frame_directly()

        # Конец сегмента:
        elif key in self.end_fragment_keys:
            position = self.video_reader.position
            self.backend.new_end(position)
            self.update_frame_directly()

        # Удаление сегмента:
        elif key in self.delete_keys:
            position = self.video_reader.position
            if self.backend.delete(position):
                self.update_frame_directly()

        # Отмена действия (Ctrl+Z) или его повтор (Z):
        elif key in self.undo_redo_keys:
            if modifiers & Qt.ControlModifier:  # нажат ли Ctrl?
                if self.backend.undo():
                    self.update_frame_directly()
            else:
                if self.backend.redo():
                    self.update_frame_directly()

        # Включение/выключение статусбара:
        elif key == self.show_statusbar_key:
            self.show_statusbar = not self.show_statusbar
            self.update_frame_directly()

        # Экспорт фрагментов:
        elif key in self.export_fragments_keys:
            self.backend.extract_fragments(self)

        # Показать/скрыть справку:
        elif key in self.show_hotkeys_info_key:
            self.show_hotkeys_info = not self.show_hotkeys_info
            QtDialogHelper.msgbox(
                "Список горячих клавиш:\n"
                "→ / > - следующий кадр\n"
                "← / < - предыдущий кадр\n"
                "Пробел - пауза/воспроизведение\n"
                "0-9 - скорость воспроизведения (2^n кадров/шаг)\n"
                "R - переключить направление воспроизведения (вперед/назад)\n"
                "J - переход к ближайшему сегменту (начало/конец)\n"
                "↑ / [ - начало сегмента\n"
                "↓ / ] - конец сегмента\n"
                "Del / D - удалить сегмент\n"
                "Ctrl+Z - отмена действия\n"
                "Z - восстановление отменённого действия\n"
                "F - переключение режима подгонки изображения\n"
                "Tab - вкл/выкл статусбар\n"
                "E - экспорт фрагментов\n"
                "H - справка по горячим клавишам\n"
                "Esc / Q - выход",
                "Справка по горячим клавишам",
                self
            )

        # Выход:
        elif key in self.drop_fragment_edge_keys:
            if QtDialogHelper.ask_ok_cancel(
                    "Вы уверены, что хотите выйти из программы?",
                    "Заверение работы",
                    self
            ):
                self.close()

        # Разрезание сегмента:
        elif key in self.insert_fragment_edge_keys:
            position = self.video_reader.position
            if self.key_pose is None:
                for start, end in self.backend.fragments:
                    if start <= position <= end:
                        QtDialogHelper.errorbox(
                            "Невозможно создать фрагмент внутри другого!",
                            "Ошибка",
                            self
                        )
                        break
                else:
                    self.key_pose = position
            else:
                fragment = sorted([self.key_pose, position])
                if self.backend.add(fragment):
                    self.key_pose = None
                    self.update_frame_directly()

    def update_frame_directly(self, frame=None):
        """Непосредственное обновление кадра без воспроизведения:"""
        if frame is None:
            frame = self.video_reader.current()

        # Отрисовываем изображение в окне:
        self._draw_frame(frame)

        # Обновляем заголовок:
        self.update_title()

    def resizeEvent(self, event):
        """Обработка изменения размера окна:"""
        super().resizeEvent(event)
        if hasattr(self, 'video_reader') and hasattr(self.video_reader, 'current'):
            self.update_frame_directly()

    def closeEvent(self, event):
        """Обработка закрытия окна:"""
        self.video_reader.close()
        event.accept()


description = 'Простой видеоредактор, позволяющий просматривать одно видео ' \
              '(preview), а вырезать кадры из другого (source). ' \
              'Сопоставление двух видео идёт по номерам кадров. Например, ' \
              'если имеется превью результата обработки видео нейросетью, ' \
              'то по нему пользователь может определить диапазоны кадров с ' \
              'некорректными результатами и экспортировать их в отдельные ' \
              'видео, чтобы использовать для дообучения модели:'


def get_args():
    parser = argparse.ArgumentParser(description=description)

    # Параметр исходного видео:
    kwargs = {'type': str,
              'default': '',
              'help': 'Исходное видео.'}
    parser.add_argument('--source', '-s', dest='source', **kwargs)
    parser.add_argument('source', nargs='?', **kwargs)

    kwargs = {'type': str,
              'default': '',
              'help': 'Обработанное видео (превью). По-умолчанию берётся '
                      'исходное видео.'}
    parser.add_argument('--preview', '-p', dest='preview', **kwargs)
    parser.add_argument('preview', nargs='?', **kwargs)

    kwargs = {'type': str,
              'default': '',
              'help': 'Директория для сохранения фрагментов. По-умолчанию '
                      'фрагменты располагаются рядом с исходным видео.'}
    parser.add_argument('--fragments_dir', '-t',
                        dest='fragments_dir', **kwargs)
    parser.add_argument('fragments_dir', nargs='?', **kwargs)

    parser.add_argument(
        '-e',
        '--export',
        action='store_true',
        default=False,
        help='Режим экспорта. Позволяет выполнять экспортирование из '
             'консоли (без запуска графического интерфейса). В этом режиме '
             'параметр source является обязательным.'
    )

    args = parser.parse_args()

    if args.export:
        source_file = args.source
        if source_file == '':
            raise ValueError('В режиме экспорта параметр source является обязательным!')
        preview_file = args.preview or source_file
        fragments_dir = args.fragments_dir or os.path.dirname(source_file)

    else:
        # Используем QtDialogHelper для диалогов:
        source_file = args.source or QtDialogHelper.ask_open_file('Укажите исходный файл')
        preview_file = args.preview or QtDialogHelper.ask_open_file('Укажите файл-превью') or source_file
        fragments_dir = args.fragments_dir or QtDialogHelper.ask_dir('Укажите папку для фрагментов') or os.path.dirname(
            source_file)

    return source_file, preview_file, fragments_dir, 'export' if args.export else 'edit'


def main_app():
    app = QApplication(sys.argv)

    # Получаем список параметров:
    source_file, preview_file, fragments_dir, mode = get_args()
    if source_file == '':
        QtDialogHelper.errorbox('Требуется исходное видео!', 'Ошибка')
        sys.exit(1)

    # В режиме редактирования запускаем интерфейс:
    if mode == 'edit':
        window = MainWindow(source_file, preview_file, fragments_dir)
        window.show()
        sys.exit(app.exec_())

    # В режиме экспорта запускаем генерацию уже отмеченных фрагментов:
    elif mode == 'export':
        backend = Backend(
            float('inf'),
            source_file,
            out_dir=fragments_dir
        )
        backend.extract_fragments()
        return 0


if __name__ == '__main__':
    main_app()


def main():
    main_app()