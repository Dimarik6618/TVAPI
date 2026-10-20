"""Compact boundary popup with two on-demand still images."""
from __future__ import annotations

import base64
import queue
import tkinter as tk
from tkinter import messagebox, ttk

from boundary_frames import BoundaryPreviewWorker
from neon_widgets import RoundedCard


class BoundaryEditor:
    def __init__(self, host, source, duration, start, end, on_save, *,
                 ffmpeg, parse_timecode, timestamp):
        self.host, self.source, self.duration = host, source, duration
        self.on_save, self.parse, self.format = on_save, parse_timecode, timestamp
        self.window = host.popup('Границы рилса')
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.window.bind('<Destroy>', self._destroyed, add='+')
        self.closed = False
        self.worker = BoundaryPreviewWorker(ffmpeg, source, duration)
        self.step = getattr(host, 'boundary_step', 1)
        self.fields = [tk.StringVar(value=timestamp(start)), tk.StringVar(value=timestamp(end))]
        self.summary = tk.StringVar()
        self.captions = [tk.StringVar(), tk.StringVar()]
        self.revisions = [0, 0]
        self.pending_after = [None, None]
        self.images = [None, None]
        self.canvases, self.scales, self.move_buttons, self.step_buttons = [], [], [], {}
        self._scale_values = [None, None]
        self._trace_ids = []
        self.poll_after = None
        self._build()
        for side, variable in enumerate(self.fields):
            token = variable.trace_add('write', lambda *_, s=side: self._changed(s))
            self._trace_ids.append(token)
            self._changed(side)
        self.poll_after = self.window.after(60, self._poll)
        self.window.update_idletasks()
        # Centre on the parent, keeping this compact fixed-size popup on-screen.
        width, height = self.window.winfo_reqwidth(), self.window.winfo_reqheight()
        x = max(0, min(host.root.winfo_x()+(host.root.winfo_width()-width)//2,
                       self.window.winfo_screenwidth()-width))
        y = max(0, min(host.root.winfo_y()+(host.root.winfo_height()-height)//2,
                       self.window.winfo_screenheight()-height))
        self.window.geometry(f'+{x}+{y}')

    def _build(self):
        h = self.host
        body = h.popup_body(self.window, 16, 12)
        filename = h.label(body, self.source.name, h.PALE)
        filename.configure(wraplength=770, justify='left')
        filename.pack(anchor='w', pady=(0, 4))
        h.label(body, f'Исходное видео: {self.format(self.duration)}', h.MUTED).pack(anchor='w')
        step_row = tk.Frame(body, bg=h.CARD)
        step_row.pack(fill='x', pady=(8, 10))
        h.label(step_row, 'ШАГ СДВИГА', h.MUTED).pack(side='left', padx=(0, 12))
        for step in (1, 5):
            button = h.button(step_row, f'{step} с', lambda s=step: self.set_step(s))
            button.pack(side='left', padx=(0, 7))
            self.step_buttons[step] = button
        h.label(step_row, 'Точный таймкод можно ввести вручную', h.MUTED).pack(side='right')
        panels = tk.Frame(body, bg=h.CARD)
        panels.pack(fill='x')
        for side, title in enumerate(('НАЧАЛО', 'КОНЕЦ')):
            card = RoundedCard(panels, color='#10243A', edge=h.EDGE, padx=10, pady=10)
            card.grid(row=0, column=side, sticky='nsew', padx=(0, 6) if side == 0 else (6, 0))
            panel = card.body
            h.label(panel, title, h.CYAN if side == 0 else h.PINK, 11).pack(anchor='w', pady=(0, 6))
            canvas = tk.Canvas(panel, width=360, height=202, bg=h.INK, highlightthickness=0)
            canvas.pack()
            self.canvases.append(canvas)
            tk.Label(panel, textvariable=self.captions[side], bg=panel.cget('bg'),
                     fg=h.MUTED, font=(h.font_family, 9)).pack(anchor='w', pady=(5, 6))
            row = tk.Frame(panel, bg=panel.cget('bg'))
            row.pack(fill='x')
            entry = ttk.Entry(row, textvariable=self.fields[side], width=17, style='Neon.TEntry')
            entry.pack(side='left', fill='x', expand=True, padx=(0, 8))
            buttons = []
            for direction in (-1, 1):
                button = h.button(row, '−5 с' if direction == -1 else '+5 с',
                                  lambda s=side, d=direction: self.move(s, d*self.step))
                button.pack(side='left', padx=(4, 0))
                buttons.append(button)
            self.move_buttons.append(buttons)
            scale = tk.Scale(panel, from_=0, to=self.duration, resolution=.01,
                             orient='horizontal', showvalue=False, length=360,
                             bg=panel.cget('bg'), fg=h.PALE, troughcolor='#173047',
                             highlightthickness=0, activebackground=h.CYAN,
                             command=lambda value, s=side: self._scale_changed(s, value))
            self.scales.append(scale)
            scale.pack(fill='x', pady=(4, 0))
            for key, direction in (('<Left>', -1), ('<Right>', 1)):
                scale.bind(key, lambda _, s=side, d=direction: self._key_move(s, d*self.step))
            for key, direction in (('<Shift-Left>', -1), ('<Shift-Right>', 1)):
                scale.bind(key, lambda _, s=side, d=direction: self._key_move(s, d*5))
        tk.Label(body, textvariable=self.summary, bg=h.CARD, fg=h.CYAN,
                 font=(h.font_family, 10)).pack(anchor='w', pady=(10, 4))
        h.label(body, 'Кадры обновляются после сдвига. Конец — последний кадр перед границей.',
                h.MUTED).pack(anchor='w', pady=(0, 10))
        actions = tk.Frame(body, bg=h.CARD)
        actions.pack(fill='x')
        self.save_button = h.button(actions, 'СОХРАНИТЬ', self.save, h.PINK)
        self.save_button.pack(side='left')
        h.button(actions, 'СОХРАНИТЬ И ОТКРЫТЬ', lambda: self.save(True)).pack(side='left', padx=7)
        h.button(actions, 'ОТМЕНА', self.close).pack(side='right')
        self.set_step(self.step)

    def set_step(self, step):
        self.step = step
        self.host.boundary_step = step
        for value, button in self.step_buttons.items():
            button.filled = value == step
            button.draw()
        for left, right in self.move_buttons:
            left.configure(text=f'−{step} с')
            right.configure(text=f'+{step} с')

    def move(self, side, delta):
        try:
            value = min(self.duration, max(0., self.parse(self.fields[side].get())+delta))
            self.fields[side].set(self.format(value))
        except ValueError:
            messagebox.showerror('Таймкод', 'Введите корректный таймкод.', parent=self.window)

    def _key_move(self, side, delta):
        self.move(side, delta)
        return 'break'

    def _scale_changed(self, side, text):
        value = float(text)
        # Scale.set schedules a callback too; don't replace precise typed values.
        if self._scale_values[side] is not None and abs(value-self._scale_values[side]) < .001:
            return
        self._scale_values[side] = value
        self.fields[side].set(self.format(value))

    def _changed(self, side):
        if self.closed:
            return
        self.revisions[side] += 1
        revision = self.revisions[side]
        if self.pending_after[side] is not None:
            self.window.after_cancel(self.pending_after[side])
            self.pending_after[side] = None
        self.worker.invalidate(side)
        self.images[side] = None
        self.canvases[side].delete('all')
        try:
            position = self.parse(self.fields[side].get())
            if not 0 <= position <= self.duration or (side == 1 and position == 0):
                raise ValueError('Граница за пределами видео')
            self.scales[side].set(position)
            self._scale_values[side] = self.scales[side].get()
            self.captions[side].set('Загрузка кадра…')
            self.canvases[side].create_text(180, 101, text='Загрузка кадра…',
                                            fill=self.host.MUTED, font=(self.host.font_family, 10))
            self.pending_after[side] = self.window.after(
                250, lambda: self._request(side, revision, position))
        except ValueError:
            self.captions[side].set('Проверьте таймкод границы')
            self.canvases[side].create_text(180, 101, text='Введите корректный таймкод',
                                            fill=self.host.MUTED, font=(self.host.font_family, 10))
        self._update_summary()

    def _update_summary(self):
        try:
            start, end = (self.parse(v.get()) for v in self.fields)
            if not 0 <= start < end <= self.duration:
                raise ValueError('Начало должно быть раньше конца и внутри исходного видео.')
            self.summary.set(f'Длительность рилса: {end-start:.2f} с')
        except ValueError as exc:
            self.summary.set(str(exc) or 'Проверьте таймкоды.')

    def _request(self, side, revision, position):
        self.pending_after[side] = None
        if not self.closed and revision == self.revisions[side]:
            self.worker.request(side, revision, position)

    def _poll(self):
        if self.closed:
            return
        try:
            while True:
                side, revision, frame, error = self.worker.results.get_nowait()
                if revision != self.revisions[side]:
                    continue
                canvas = self.canvases[side]
                canvas.delete('all')
                if error:
                    self.captions[side].set('Кадр недоступен')
                    canvas.create_text(180, 101, text=error, width=330,
                                       fill=self.host.MUTED, font=(self.host.font_family, 9))
                    continue
                try:
                    photo = tk.PhotoImage(master=self.window,
                                          data=base64.b64encode(frame.png), format='png')
                    self.images[side] = photo
                    canvas.create_image(180, 101, image=photo)
                    prefix = 'Первый кадр' if side == 0 else 'Последний кадр'
                    self.captions[side].set(f'{prefix} · {self.format(frame.timestamp)}')
                except tk.TclError:
                    self.captions[side].set('Кадр недоступен')
                    canvas.create_text(180, 101, text='Не удалось отобразить кадр',
                                       fill=self.host.MUTED, font=(self.host.font_family, 9))
        except queue.Empty:
            pass
        self.poll_after = self.window.after(60, self._poll)

    def save(self, open_video=False):
        try:
            start, end = (self.parse(v.get()) for v in self.fields)
            if not 0 <= start < end <= self.duration:
                raise ValueError('Начало должно быть раньше конца, а границы — внутри исходного видео.')
        except ValueError as exc:
            messagebox.showerror('Границы', str(exc), parent=self.window)
            return
        if self.on_save(start, end, open_video):
            self.close()

    def _destroyed(self, event):
        if event.widget == self.window:
            self._cleanup()

    def _cleanup(self):
        if self.closed:
            return
        self.closed = True
        self.worker.close()
        for token in [*self.pending_after, self.poll_after]:
            if token is not None:
                try:
                    self.window.after_cancel(token)
                except tk.TclError:
                    pass
        for variable, token in zip(self.fields, self._trace_ids):
            variable.trace_remove('write', token)
        self.images = [None, None]
        # Tk variables must be destroyed here on the UI thread. Leaving them
        # in closed-widget cycles lets a later decoder/analysis allocation
        # trigger Python's GC and their Tk destructors on a worker thread.
        self.fields.clear()
        self.captions.clear()
        self.summary = None
        self.canvases.clear()
        self.scales.clear()
        self.move_buttons.clear()
        self.step_buttons.clear()
        self.save_button = None
        if getattr(self.host, 'boundary_editor', None) is self:
            self.host.boundary_editor = None

    def close(self):
        self._cleanup()
        if self.window.winfo_exists():
            self.window.destroy()
