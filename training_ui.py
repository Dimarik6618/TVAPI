"""Compact annotation/training popup; all video viewing stays in an external player."""
from __future__ import annotations

import subprocess
import tkinter as tk
import zipfile
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from learning import LearningStore, download_references, report_text, train
from hardware import Runtime


class TrainingWindow:
    def __init__(self, app):
        self.app = app
        self.folder = app.learning_folder
        self.store = LearningStore(self.folder)
        self.window = app.popup('Обучение отбора')
        self.window.geometry('780x655')
        self.window.minsize(740, 610)
        self.window.resizable(True, True)
        self.body = app.popup_body(self.window, 18, 14)
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(4, weight=1)
        self.buttons = []
        self.summary = tk.StringVar()
        self.model_status = tk.StringVar()
        self.enabled = tk.BooleanVar(value=self.store.enabled())
        app.label(self.body, 'МЕТКИ → ПРИЗНАКИ ВИДЕО → ПРОВЕРКА НА ДРУГИХ ЗАПИСЯХ', app.PINK).grid(row=0, column=0, sticky='w')
        tk.Label(self.body, textvariable=self.summary, bg=app.CARD, fg=app.PALE,
                 font=(app.font_family, 10), anchor='w').grid(row=1, column=0, sticky='ew', pady=(6, 3))
        tk.Label(self.body, text='Папка модели: ' + str(self.folder), bg=app.CARD, fg=app.MUTED,
                 wraplength=705, justify='left', anchor='w').grid(row=2, column=0, sticky='ew', pady=(0, 8))
        controls = tk.Frame(self.body, bg=app.CARD)
        controls.grid(row=3, column=0, sticky='ew', pady=(0, 8))
        for text, command in [('ГОРОДСКИЕ ПРИМЕРЫ', self.references), ('ИМПОРТ МЕТОК', self.import_file),
                              ('ПАПКА MP4', self.bind_folder), ('СВОЯ МЕТКА', self.annotation)]:
            self.add_button(controls, text, command).pack(side='left', padx=(0, 5))
        area = tk.Frame(self.body, bg=app.CARD)
        area.grid(row=4, column=0, sticky='nsew')
        area.columnconfigure(0, weight=1)
        area.rowconfigure(0, weight=1)
        self.table = ttk.Treeview(area, columns=('file', 'start', 'end', 'type', 'label'),
                                 show='headings', style='Neon.Treeview', height=7)
        for key, title, width in [('file', 'MP4', 110), ('start', 'НАЧАЛО', 94), ('end', 'КОНЕЦ', 94),
                                  ('type', 'МЕТКА', 108), ('label', 'СОБЫТИЕ / ИСТОЧНИК МЕТКИ', 255)]:
            self.table.heading(key, text=title)
            self.table.column(key, width=width, stretch=key in ('file', 'label'),
                              anchor='w' if key in ('file', 'label') else 'center')
        self.table.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(area, orient='vertical', command=self.table.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.bind('<Double-1>', self.open_video)
        actions = tk.Frame(self.body, bg=app.CARD)
        actions.grid(row=5, column=0, sticky='ew', pady=8)
        for title, command in [('ОТКРЫТЬ ОТРЕЗОК', self.open_video), ('ИЗМЕНИТЬ МЕТКУ', lambda: self.annotation(True)),
                                ('ПРИВЯЗАТЬ MP4', self.bind_selected), ('УДАЛИТЬ МЕТКУ', self.remove)]:
            self.add_button(actions, title, command).pack(side='left', padx=(0, 5))
        tk.Label(self.body, text='Части одной поездки и копии видео должны иметь одну группу исходной записи.\n'
                                'Городские примеры размечены помощником; их можно исправить перед обучением.',
                 bg=app.CARD, fg=app.MUTED, justify='left', font=(app.font_family, 9)).grid(row=6, column=0, sticky='w')
        bottom = tk.Frame(self.body, bg=app.CARD)
        bottom.grid(row=7, column=0, sticky='ew', pady=(12, 4))
        self.add_button(bottom, 'ОБУЧИТЬ И ПРОВЕРИТЬ', self.start_training, app.PINK, True).pack(side='left', padx=(0, 8))
        self.report_button = self.add_button(bottom, 'РЕЗУЛЬТАТ ПРОВЕРКИ', self.show_report)
        self.report_button.pack(side='left')
        self.stop_button = app.button(bottom, '■ СТОП', app.stop_work, app.PINK)
        self.stop_button.pack(side='right')
        self.check = tk.Checkbutton(self.body, text='Использовать обученную модель при следующем поиске',
                                    variable=self.enabled, command=self.toggle,
                                    bg=app.CARD, fg=app.CYAN, selectcolor='#173047',
                                    activebackground=app.CARD, activeforeground=app.CYAN)
        self.check.grid(row=8, column=0, sticky='w')
        tk.Label(self.body, textvariable=self.model_status, bg=app.CARD, fg=app.MUTED,
                 wraplength=705, justify='left', anchor='w', font=(app.font_family, 9)).grid(row=9, column=0, sticky='ew', pady=(5, 0))
        self.refresh()

    def alive(self):
        return bool(self.window.winfo_exists())

    def add_button(self, parent, title, command, accent=None, filled=False):
        button = self.app.button(parent, title, command, accent, filled)
        self.buttons.append(button)
        return button

    def refresh(self):
        if not self.alive():
            return
        self.store = LearningStore(self.folder)
        self.summary.set(self.store.summary())
        self.table.delete(*self.table.get_children())
        from moto_shorts import timestamp
        sources = {s['id']: s for s in self.store.data['sources']}
        for e in self.store.data['examples']:
            s = sources[e['source_id']]
            origin = 'ваша метка' if e.get('reviewed_by_user') else 'импорт'
            self.table.insert('', 'end', iid=e['id'], values=(s['file'], timestamp(e['start']), timestamp(e['end']),
                                'Событие' if e['accepted'] else 'Обычный', (e.get('label') or e.get('category')) + ' · ' + origin))
        try:
            model = self.store.model()
            self.enabled.set(self.store.enabled() and model is not None)
            if model:
                data, _ = model
                evaluation = data['evaluation']
                changed = data['dataset_signature'] != self.store.signature()
                text = ('Пилотная модель' if evaluation['pilot'] else 'Модель сохранена')
                text += f" · {evaluation['examples_count']} примеров · " + ('включена' if self.enabled.get() else 'выключена')
                if changed:
                    text += '. Метки изменились: для их учёта запустите обучение снова.'
                elif not evaluation['improved']:
                    text += '. Проверка пока не показала уверенного улучшения.'
                self.model_status.set(text)
            else:
                self.model_status.set('Для первого обучения: минимум 5 событий, 5 обычных отрезков и 3 независимые записи.\n'
                                      'Установите Установить_обучение.bat. Первый запуск скачает модель X-CLIP.')
        except (ValueError, OSError, KeyError, TypeError) as exc:
            model = None
            self.model_status.set(str(exc))
        self.check.config(state='normal' if model and not self.app.busy else 'disabled')
        self.report_button.config(state='normal' if model and not self.app.busy else 'disabled')
        self.stop_button.config(state='normal' if self.app.busy else 'disabled')

    def set_busy(self, busy):
        if not self.alive():
            return
        for button in self.buttons:
            button.config(state='disabled' if busy else 'normal')
        self.check.config(state='disabled' if busy else 'normal')
        self.stop_button.config(state='normal' if busy else 'disabled')
        if not busy:
            self.refresh()

    def task(self, function):
        def work():
            message = function(lambda f, text: self.app.events.put(('progress', (f, text))))
            return ('training', message)
        self.app.update_progress(0., 'Подготовка обучения')
        self.app.background(work)

    def references(self):
        if self.app.busy:
            return
        self.store.import_file(Path(__file__).parent / 'training_examples' / 'urban_annotations.json')
        self.store.install_reference_model(Path(__file__).parent / 'training_examples' / 'urban_pilot_model.json')
        self.refresh()
        store = self.store
        def work(progress):
            n, errors = download_references(store, progress)
            return f'Доступно городских MP4: {n}.\n' + '\n'.join(errors[:4])
        self.task(work)

    def import_file(self):
        name = filedialog.askopenfilename(parent=self.window, title='Разметка или архив городского теста',
                                         filetypes=[('Разметка', '*.json *.zip')])
        if name:
            try:
                self.store.import_file(name)
                self.refresh()
            except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
                messagebox.showerror('Разметка', str(exc), parent=self.window)

    def bind_folder(self):
        name = filedialog.askdirectory(parent=self.window, title='Папка с MP4 из разметки')
        if name:
            store = self.store
            def work(progress):
                progress(0., 'Проверка соответствия MP4 разметке')
                n, errors = store.bind_folder(name)
                return f'Подключено исходников: {n}.\n' + '\n'.join(errors[:4])
            self.task(work)

    def selected(self):
        values = self.table.selection()
        if not values:
            return None, None
        example = next(e for e in self.store.data['examples'] if e['id'] == values[0])
        source = next(s for s in self.store.data['sources'] if s['id'] == example['source_id'])
        return example, source

    def bind_selected(self):
        example, source = self.selected()
        if source:
            name = filedialog.askopenfilename(parent=self.window, title='MP4 для ' + source['file'], filetypes=[('Видео', '*.mp4')])
            if name:
                store = self.store
                def work(progress):
                    progress(0., 'Проверка исходника')
                    store.bind_source(source['id'], name)
                    return 'MP4 подключён.'
                self.task(work)

    def annotation(self, edit=False):
        if self.app.busy:
            return
        from moto_shorts import parse_timecode, timestamp, video_info, ffmpeg_exe
        example, source = self.selected() if edit else (None, None)
        if edit and not example:
            return
        missing_media = bool(edit and (not source.get('path') or not Path(source['path']).is_file()))
        index = self.app.selected_index()
        initial = source.get('path', '') if source else (str(self.app.sources[index]) if index is not None else '')
        if not initial and self.app.files and not edit:
            initial = str(self.app.files[0])
        clip = self.app.clips[index] if index is not None and not edit else None
        dialog = self.app.popup('Изменить метку' if edit else 'Своя метка события')
        def close_dialog():
            dialog.destroy()
            if self.alive():self.window.grab_set()
        dialog.protocol('WM_DELETE_WINDOW', close_dialog)
        body = self.app.popup_body(dialog, 18, 16)
        filename = tk.StringVar(value=initial)
        start = tk.StringVar(value=timestamp(example['start'] if example else clip.start if clip else 0.))
        end = tk.StringVar(value=timestamp(example['end'] if example else clip.end if clip else 6.))
        group = tk.StringVar(value=source['group'] if source else initial)
        label = tk.StringVar(value=example.get('label', '') if example else '')
        kind = tk.StringVar(value='Событие' if not example or example['accepted'] else 'Обычный отрезок')
        for title, var in [('MP4', filename), ('Начало (таймкод)', start), ('Конец (таймкод)', end),
                           ('Исходная поездка / группа', group), ('Что происходит', label)]:
            self.app.label(body, title).pack(anchor='w', pady=(5, 2))
            ttk.Entry(body, textvariable=var, width=62, style='Neon.TEntry',
                      state='readonly' if title == 'MP4' or (missing_media and title in ('Начало (таймкод)','Конец (таймкод)')) else 'normal').pack(fill='x')
            if title == 'MP4' and not edit:
                def choose():
                    name = filedialog.askopenfilename(parent=dialog, filetypes=[('Видео', '*.mp4')])
                    if name:
                        filename.set(name)
                        group.set(str(Path(name).resolve()))
                self.app.button(body, 'ВЫБРАТЬ MP4', choose).pack(anchor='w', pady=4)
        ttk.Combobox(body, textvariable=kind, values=['Событие', 'Обычный отрезок'], state='readonly',
                     style='Neon.TCombobox').pack(fill='x', pady=10)
        self.app.label(body, 'У частей одной поездки укажите одинаковую группу.').pack(anchor='w')
        def save():
            try:
                a, b = parse_timecode(start.get()), parse_timecode(end.get())
                # The display rounds timecodes. Changing only a label must
                # retain the exact interval of its frozen features.
                if example:
                    if start.get() == timestamp(example['start']):a=example['start']
                    if end.get() == timestamp(example['end']):b=example['end']
                offset = source.get('offset', 0.) if source else 0.
                if missing_media:
                    if (a,b)!=(example['start'],example['end']):
                        raise ValueError('Для новых границ нужны кадры исходного MP4.')
                    duration=b-offset
                else:
                    duration, _ = video_info(Path(filename.get()), ffmpeg_exe())
                if not 0 <= a-offset < b-offset <= duration + .06 or not group.get().strip():
                    raise ValueError('Границы должны быть внутри MP4; укажите исходную поездку.')
                if example:
                    example.update(start=a, end=b, accepted=kind.get() == 'Событие', label=label.get(),
                                   origin='user', reviewed_by_user=True, confidence='user')
                    source['group'] = group.get().strip()
                    self.store.save()
                    if source.get('archive_id'):
                        from event_archive import EventArchive
                        EventArchive(self.folder).review(source['archive_id'],a-offset,b-offset,example['accepted'])
                else:
                    self.store.add_user(Path(filename.get()), a, b, kind.get() == 'Событие', label.get(), group.get().strip())
                close_dialog()
                self.refresh()
            except (ValueError, OSError) as exc:
                messagebox.showerror('Метка', str(exc), parent=dialog)
        self.app.button(body, 'СОХРАНИТЬ МЕТКУ', save, self.app.PINK, True).pack(fill='x', pady=(10, 0))

    def remove(self):
        example, _ = self.selected()
        if example:
            self.store.data['examples'] = [e for e in self.store.data['examples'] if e['id'] != example['id']]
            self.store.save()
            self.refresh()

    def open_video(self, event=None):
        if self.app.busy:
            return
        if event is not None:
            row = self.table.identify_row(event.y)
            if row:
                self.table.selection_set(row)
        example, source = self.selected()
        if not source:
            return
        if not source.get('path') or not Path(source['path']).is_file():
            path=Path(source.get('path') or source['file'])
            drive=getattr(self.app,'drive_store',None)
            if drive and drive.entry(path):
                def ready(_):
                    if self.alive():
                        self.table.selection_set(example['id']);self.open_video()
                def work():
                    drive.ensure(path,lambda f:self.app.events.put(('progress',(f,'Загрузка исходника с Google Drive'))))
                    return ('callback',(ready,None))
                self.app.background(work)
            else:
                messagebox.showinfo('MP4','Для просмотра нужен исходник. Метку и группу можно менять без MP4; обучение использует сохранённые признаки.',parent=self.window)
            return
        from moto_shorts import external_player_command, timestamp
        offset = source.get('offset', 0.)
        command = external_player_command(Path(source['path']), example['start']-offset, example['end']-offset)
        try:
            if command:
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self.app.system_open(Path(source['path']))
                messagebox.showinfo('Таймкод', 'В системном плеере перейдите к ' + timestamp(example['start']-offset) +
                                    '. Для автоматического перехода установите VLC или mpv.', parent=self.window)
        except OSError as exc:
            messagebox.showerror('Просмотр', str(exc), parent=self.window)

    def start_training(self):
        store, profile = self.store, self.app.profile_key()
        def work(progress):
            from moto_shorts import ffmpeg_exe
            output = train(store, ffmpeg_exe(), Runtime.detect(profile), progress)
            return report_text(output)
        self.task(work)

    def toggle(self):
        try:
            self.store.set_enabled(self.enabled.get())
            self.refresh()
        except (OSError, ValueError) as exc:
            messagebox.showerror('Модель', str(exc), parent=self.window)
            self.enabled.set(False)

    def show_report(self):
        model = self.store.model()
        if model:
            messagebox.showinfo('Проверка обучения', report_text(model[0]), parent=self.window)
