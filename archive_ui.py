"""Inspect and rescore stored evidence without reconstructing or storing video."""
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import fields
from pathlib import Path

from event_archive import EventArchive
from learning import LearningStore


def evidence_text(clip, source=None):
    from moto_shorts import timestamp
    lines = []
    if source:
        lines.append('Видео: '+source.get('file', ''))
    lines += [f'Отрезок: {timestamp(clip.start)} — {timestamp(clip.end)}',
              f'Баллы: {clip.score:.3f}', 'Причина: '+clip.reason]
    if clip.decision_reason:
        lines.append('Ваше решение: '+clip.decision_reason)
    lines.append('\nСигналы:')
    labels = {'baseline':'Быстрый анализ','objects':'Объекты','sound':'Звуковые события','speech':'Речь',
              'meaning':'Сюжет','learned_event':'Обученная модель','quality':'Качество','gps':'GPS','camera':'Движение камеры'}
    lines += [f'{labels.get(k,k)}: {v:.3f}' for k,v in clip.signals.items()]
    lines.append('\nОбъекты и развитие события:')
    lines += [f"{timestamp(e['start'])}–{timestamp(e['end'])}: {e['label']}" for e in clip.details.get('object_events',[])] or ['Нет распознанных событий.']
    lines.append('\nЗвуки:')
    lines += [f"{timestamp(e['start'])}–{timestamp(e['end'])}: {e['label']} ({e['score']:.2f})" for e in clip.details.get('sound_events',[])] or ['Нет распознанных событий.']
    lines.append('\nРечь:')
    lines += [f"{timestamp(s['start'])}–{timestamp(s['end'])}: {s['text']}" for s in clip.details.get('speech',[])] or ['Речь не распознана или анализ речи выключен.']
    lines.append('\nПамять: '+('Признаки сохранены; можно переобучать текущую модель без MP4.' if clip.embedding else
                              'Сохранён анализ. Для получения визуальных признаков нужен исходник и Установить_обучение.bat.'))
    return '\n'.join(lines)


def show_evidence(host, clip, source=None):
    window = host.popup('Анализ события')
    window.geometry('690x570')
    body = host.popup_body(window,18,14)
    body.pack(fill='both',expand=True)
    text = tk.Text(body, bg=host.CARD, fg=host.PALE, font=(host.font_family,10), wrap='word',
                   borderwidth=0, padx=8, pady=8)
    text.pack(fill='both',expand=True)
    text.insert('1.0',evidence_text(clip,source))
    text.config(state='disabled')


class ArchiveWindow:
    def __init__(self, host):
        self.host, self.archive = host, EventArchive(host.learning_folder)
        self.window = host.popup('Память событий')
        self.window.geometry('820x580')
        body = host.popup_body(self.window,18,14)
        body.pack(fill='both',expand=True)
        host.label(body,'Сведения и признаки сохраняются после удаления MP4. Показаны последние 1000 событий.').pack(anchor='w',pady=(0,8))
        self.table = ttk.Treeview(body,columns=('video','from','score','status','features'),show='headings',style='Neon.Treeview')
        for key,title,width in [('video','ВИДЕО',250),('from','НАЧАЛО',95),('score','БАЛЛ',75),('status','РЕШЕНИЕ',140),('features','ПРИЗНАКИ',105)]:
            self.table.heading(key,text=title)
            self.table.column(key,width=width,stretch=key=='video')
        self.table.pack(fill='both',expand=True)
        self.table.bind('<Double-1>',lambda _:self.inspect())
        buttons = tk.Frame(body,bg=host.CARD)
        buttons.pack(fill='x',pady=(10,0))
        host.button(buttons,'АНАЛИЗ',self.inspect).pack(side='left')
        host.button(buttons,'ПЕРЕОЦЕНИТЬ',self.rescore).pack(side='left',padx=7)
        host.button(buttons,'ОБНОВИТЬ',self.refresh).pack(side='left')
        host.button(buttons,'■ СТОП',host.stop_work,host.PINK).pack(side='right')
        self.refresh()

    def refresh(self):
        from moto_shorts import timestamp
        self.items = self.archive.list_events()
        self.table.delete(*self.table.get_children())
        for i,e in enumerate(self.items):
            status = ('Принято' if e.get('selected') else 'Исключено') if e.get('reviewed') else 'Не проверено'
            self.table.insert('','end',iid=str(i),values=(e['source']['file'],timestamp(e['start']),
                f"{e['score']:.3f}",status,'Есть' if e['feature_ready'] else 'Нет'))

    def inspect(self):
        from moto_shorts import Clip
        rows = self.table.selection()
        if not rows:return
        e = self.items[int(rows[0])]
        allowed = {f.name for f in fields(Clip)}
        clip = Clip(**{k:v for k,v in e.items() if k in allowed})
        feature = self.archive.feature(e['source']['id'],clip.start,clip.end)
        if feature:clip.embedding=feature[0].tolist()
        show_evidence(self.host,clip,e['source'])

    def rescore(self):
        if self.host.busy:return
        from moto_shorts import Clip
        value = LearningStore(self.host.learning_folder).model()
        if value is None:
            messagebox.showinfo('Память событий','Сначала обучите модель.',parent=self.window)
            return
        _, model = value
        items = list(self.items)
        allowed = {f.name for f in fields(Clip)}
        def work():
            import task_control as tasks
            count=0
            for e in items:
                tasks.check()
                feature = self.archive.feature(e['source']['id'],e['start'],e['end'])
                if not feature:continue
                clip=Clip(**{k:v for k,v in e.items() if k in allowed})
                clip.embedding=feature[0].tolist()
                score=float(model.predict(feature[0]))
                clip.signals['learned_event']=score
                clip.score=round(.45*clip.signals.get('baseline',clip.score)+.55*score,3)
                self.archive.save_clip(e['source']['id'],clip)
                count+=1
            return ('callback',(self.rescored,count))
        self.host.background(work)

    def rescored(self,count):
        if self.window.winfo_exists():
            self.refresh()
            messagebox.showinfo('Память событий',f'Переоценено без MP4: {count}. Результаты сохранены в памяти событий.',parent=self.window)
