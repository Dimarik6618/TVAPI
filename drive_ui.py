"""Small Google Drive picker. All network and OAuth work runs outside Tk."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from event_archive import EventArchive


class DriveWindow:
    def __init__(self, host):
        self.host = host
        self.window = host.popup('Google Drive')
        self.window.geometry('760x580')
        self.window.minsize(690, 500)
        body = host.popup_body(self.window, 18, 14)
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(4, weight=1)
        self.folder = tk.StringVar()
        self.recursive = tk.BooleanVar(value=True)
        self.cleanup = tk.BooleanVar(value=host.drive_store.auto_cleanup)
        host.label(body, 'Папка Google Drive: ссылка или ID; пусто — все MP4').grid(row=0,column=0,sticky='w')
        ttk.Entry(body, textvariable=self.folder, style='Neon.TEntry').grid(row=1,column=0,sticky='ew',pady=6)
        options = tk.Frame(body, bg=host.CARD)
        options.grid(row=2,column=0,sticky='ew')
        for title, variable in [('Включая подпапки', self.recursive),
                                ('Удалять временный MP4 после сохранения признаков', self.cleanup)]:
            tk.Checkbutton(options, text=title, variable=variable, bg=host.CARD, fg=host.PALE,
                           selectcolor='#173047', activebackground=host.CARD).pack(anchor='w')
        controls = tk.Frame(body, bg=host.CARD)
        controls.grid(row=3,column=0,sticky='ew',pady=8)
        host.button(controls, 'ПОДКЛЮЧИТЬ GOOGLE', self.connect).pack(side='left')
        host.button(controls, 'ПОКАЗАТЬ MP4', self.refresh).pack(side='left', padx=7)
        host.button(controls, '■ СТОП', host.stop_work, host.PINK).pack(side='left')
        area = tk.Frame(body, bg=host.CARD)
        area.grid(row=4,column=0,sticky='nsew')
        area.columnconfigure(0,weight=1)
        area.rowconfigure(0,weight=1)
        self.table = ttk.Treeview(area, columns=('name','size'), show='headings', style='Neon.Treeview',
                                  selectmode='extended', height=5)
        self.table.heading('name', text='ВИДЕО')
        self.table.heading('size', text='РАЗМЕР')
        self.table.column('name', width=480)
        self.table.column('size', width=100, anchor='center', stretch=False)
        self.table.grid(row=0,column=0,sticky='nsew')
        scrollbar=ttk.Scrollbar(area,orient='vertical',command=self.table.yview)
        scrollbar.grid(row=0,column=1,sticky='ns')
        self.table.configure(yscrollcommand=scrollbar.set)
        self.items = []
        foot = tk.Frame(body, bg=host.CARD)
        foot.grid(row=5,column=0,sticky='ew',pady=(10,0))
        host.button(foot, 'ВЫБРАТЬ ВСЕ', lambda:self.table.selection_set(self.table.get_children())).pack(side='left')
        host.button(foot, 'В ОЧЕРЕДЬ', self.add, host.CYAN, True).pack(side='left', padx=7)
        host.button(foot, 'ОЧИСТИТЬ ЗАГРУЗКИ', self.clean, host.PINK).pack(side='left')
        caption=host.label(body, 'Drive хранит оригинал; программа удаляет только свои временные загрузки.')
        caption.config(wraplength=600,justify='left')
        caption.grid(row=6,column=0,sticky='w',pady=(8,0))

    def alive(self):
        return bool(self.window.winfo_exists())

    def connect(self):
        if self.host.busy:return
        name = filedialog.askopenfilename(parent=self.window,title='Desktop OAuth JSON из Google Cloud',
                                          filetypes=[('OAuth JSON','*.json')])
        if not name:return
        def work():
            self.host.drive_store.connect(name)
            return ('callback',(self.connected,None))
        self.host.update_progress(0., 'Завершите вход Google в браузере; ОСТАНОВИТЬ отменяет вход.')
        self.host.background(work)

    def connected(self, _):
        if self.alive():
            messagebox.showinfo('Google Drive','Аккаунт подключён с доступом на чтение. Нажмите ПОКАЗАТЬ MP4.',parent=self.window)

    def refresh(self):
        if self.host.busy:return
        folder, recursive = self.folder.get(), self.recursive.get()
        self.host.background(lambda:('callback',(self.loaded,self.host.drive_store.list_videos(folder,recursive))))

    def loaded(self, items):
        if not self.alive():return
        self.items = items
        self.table.delete(*self.table.get_children())
        for i, item in enumerate(items):
            size = f"{int(item.get('size',0))/1024**3:.2f} ГБ"
            self.table.insert('', 'end', iid=str(i), values=(item['name'],size))
        self.host.status.set(f'Google Drive: {len(items)} MP4.')

    def add(self):
        if self.host.busy:return
        selected = self.table.selection()
        if not selected:return
        paths = [self.host.drive_store.add(self.items[int(i)]) for i in selected]
        self.host.drive_store.set_auto_cleanup(self.cleanup.get())
        self.host.set_files(paths)
        from pathlib import Path
        self.host.folder.set(str(Path(self.host.learning_folder).parent/'shorts'))
        self.window.destroy()

    def clean(self):
        if self.host.busy:return
        archive = EventArchive(self.host.learning_folder)
        entries = list(self.host.drive_store.entries)
        def work():
            removed = sum(self.host.drive_store.release(path,archive) for path in entries)
            return ('callback',(self.cleaned,removed))
        self.host.background(work)

    def cleaned(self, count):
        if self.alive():
            messagebox.showinfo('Временные загрузки',f'Удалено MP4: {count}. Незавершённые записи и файлы без признаков сохранены.',parent=self.window)
