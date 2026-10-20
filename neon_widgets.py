"""Rounded Tk widgets with no extra runtime dependencies."""
import tkinter as tk
from tkinter import font as tkfont


def ui_font(parent, semibold=False):
    names={name.lower():name for name in tkfont.families(parent)}
    choices=(['Segoe UI Semibold','Segoe UI','DejaVu Sans','Arial'] if semibold else
             ['Segoe UI','DejaVu Sans','Arial'])
    return next((names[name.lower()] for name in choices if name.lower() in names),
                tkfont.nametofont('TkDefaultFont').actual('family'))


def rounded_rect(canvas, x1, y1, x2, y2, radius=18, **options):
    radius = min(radius, (x2-x1)/2, (y2-y1)/2)
    points = [x1+radius,y1, x2-radius,y1, x2,y1, x2,y1+radius,
              x2,y2-radius, x2,y2, x2-radius,y2, x1+radius,y2,
              x1,y2, x1,y2-radius, x1,y1+radius, x1,y1]
    return canvas.create_polygon(points, smooth=True, splinesteps=32, **options)


def blend(first, second, amount):
    a = tuple(int(first[i:i+2],16) for i in (1,3,5))
    b = tuple(int(second[i:i+2],16) for i in (1,3,5))
    return '#{:02x}{:02x}{:02x}'.format(*(round(x+(y-x)*amount) for x,y in zip(a,b)))


class RoundedCard(tk.Frame):
    def __init__(self, parent, *, color='#0C1B2E', edge='#2A455F', padx=16, pady=14):
        super().__init__(parent, bg=parent.cget('bg'), borderwidth=0)
        self.surface = tk.Canvas(self, bg=self.cget('bg'), highlightthickness=0)
        self.surface.place(x=0, y=0, relwidth=1, relheight=1)
        self.body = tk.Frame(self, bg=color)
        self.body.pack(fill='both', expand=True, padx=padx, pady=pady)
        def draw(event):
            self.surface.delete('all')
            rounded_rect(self.surface, 1, 1, event.width-1, event.height-1,
                         24, fill=color, outline=edge, width=1)
        self.bind('<Configure>', draw)


class RoundedButton(tk.Canvas):
    def __init__(self, parent, *, text, command, accent='#27E5E8', filled=False):
        self.text, self.command, self.accent, self.filled = text, command, accent, filled
        self.state, self.hover, self.focused, self.pressed = 'normal', False, False, False
        self.typeface = tkfont.Font(family=ui_font(parent), size=10, weight='bold' if filled else 'normal')
        super().__init__(parent, width=self.typeface.measure(text)+32,
                         height=46 if filled else 38, bg=parent.cget('bg'),
                         highlightthickness=0, borderwidth=0, takefocus=1,
                         cursor='hand2')
        self.bind('<Configure>', lambda _: self.draw())
        self.bind('<Enter>', lambda _: self.set_hover(True))
        self.bind('<Leave>', lambda _: self.set_hover(False))
        self.bind('<Button-1>', self.press)
        self.bind('<ButtonRelease-1>', self.release)
        self.bind('<space>', self.activate)
        self.bind('<Return>', self.activate)
        self.bind('<FocusIn>', lambda _: self.set_focus(True))
        self.bind('<FocusOut>', lambda _: self.set_focus(False))

    def set_hover(self, value):
        self.hover = value
        self.draw()

    def set_focus(self, value):
        self.focused = value
        self.draw()

    def draw(self):
        self.delete('all')
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return
        disabled = self.state == 'disabled'
        fill = ('#14283D' if disabled else self.accent) if self.filled else (
            '#1C3D55' if self.hover and not disabled else '#132B43')
        if self.filled and self.hover and not disabled:
            fill = blend(fill, '#FFFFFF', .15)
        outline = self.accent if (self.hover or self.focused) and not disabled else '#34536E'
        rounded_rect(self, 1, 1, w-1, h-1, h/2, fill=fill, outline=outline, width=1)
        fg = '#70869A' if disabled else ('#06111F' if self.filled else '#F4F8FF')
        self.create_text(w/2, h/2, text=self.text, font=self.typeface, fill=fg)

    def config(self, cnf=None, **kwargs):
        if isinstance(cnf, dict):
            kwargs = {**cnf, **kwargs}
        for key in ('state', 'text', 'command'):
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
        if kwargs:
            super().configure(**kwargs)
        super().configure(cursor='arrow' if self.state == 'disabled' else 'hand2',
                          takefocus=0 if self.state == 'disabled' else 1)
        self.draw()

    configure = config

    def cget(self, key):
        if key in ('state', 'text', 'command'):
            return getattr(self, key)
        return super().cget(key)

    def invoke(self):
        if self.state != 'disabled' and self.command:
            return self.command()

    def activate(self, event):
        self.invoke()
        return 'break'

    def release(self, event):
        pressed, self.pressed = self.pressed, False
        if pressed and 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height():
            self.invoke()

    def press(self, event):
        self.pressed = self.state != 'disabled'
        if self.pressed:
            self.focus_set()
