"""
Графічний інтерфейс наукового дослідження алгоритмів оптимізації.

Математичне ядро не реалізується в цьому файлі.
Використовуються наявні модулі:
datasets.py
experiments.py
optimizers.py
visualization.py
"""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Any

import pandas as pd

# Pillow використовується для якісного масштабування мініатюр графіків.
# Якщо бібліотека відсутня, програма має резервний варіант на Tkinter.
try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - резервний режим
    Image = None
    ImageTk = None

from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

from datasets import DATASET_NAMES
from experiments import run_experiments, validate_config
from optimizers import OPTIMIZER_NAMES


# ============================================================
# КОНСТАНТИ
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

BG = "#08090D"
SIDEBAR = "#0D0F14"

PANEL = "#141720"
PANEL_ALT = "#1C202B"

BORDER = "#292E3A"

TEXT = "#F3F5FA"
MUTED = "#929AA9"

PINK = "#FF4FA3"
PINK_HOVER = "#FF75B9"

BLUE = "#56C8FF"
YELLOW = "#FFC857"
CYAN = "#50E3C2"
ROSE = "#FF5DB1"

GREEN = "#2EEA8B"
RED = "#FF526D"
ORANGE = "#FFCA57"

OPTIMIZER_COLORS = {
    "SGD": BLUE,
    "Momentum": YELLOW,
    "RMSProp": CYAN,
    "Adam": ROSE,
}

OPTIMIZER_ROW_COLORS = {
    "SGD": ("#142738", "#E6F4FF"),
    "Momentum": ("#302817", "#FFF0C2"),
    "RMSProp": ("#142F29", "#DCFFF4"),
    "Adam": ("#35182C", "#FFE0F0"),
}


PROJECT_TITLE = (
    "Порівняльний аналіз алгоритмів оптимізації для "
    "мінімізації функції втрат нейронної мережі "
    "на нелінійно роздільних наборах даних"
)


# ============================================================
# ГОЛОВНИЙ КЛАС
# ============================================================

class OptiLabApp:

    def __init__(self, root: tk.Tk):

        self.root = root

        self.root.title(PROJECT_TITLE)
        self.root.geometry("1510x960")
        self.root.minsize(1150, 760)
        self.root.configure(bg=BG)

        self.events = queue.Queue()

        self.worker = None
        self.cancel_event = None
        self.is_running = False

        self.current_dir = None
        self.current_config = {}
        self.current_runs = pd.DataFrame()

        self.history_paths = {}
        self.run_lookup = {}

        self.sort_states = {}

        self.current_page = "overview"

        self.plot_figure = None
        self.plot_canvas = None

        self._setup_styles()
        self._create_variables()
        self._build_interface()

        self._refresh_history()

        self.root.after(100, self._poll_events)

    # ========================================================
    # ТЕМА
    # ========================================================

    def _setup_styles(self):

        style = ttk.Style(self.root)

        style.theme_use("clam")

        style.configure(
            ".",
            background=BG,
            foreground=TEXT,
            font=("Segoe UI", 10)
        )

        style.configure(
            "TFrame",
            background=BG
        )

        style.configure(
            "Panel.TFrame",
            background=PANEL
        )

        style.configure(
            "TLabel",
            background=BG,
            foreground=TEXT
        )

        style.configure(
            "Panel.TLabel",
            background=PANEL,
            foreground=TEXT
        )

        style.configure(
            "Muted.TLabel",
            background=BG,
            foreground=MUTED
        )

        style.configure(
            "PanelMuted.TLabel",
            background=PANEL,
            foreground=MUTED
        )

        style.configure(
            "Title.TLabel",
            background=BG,
            foreground=TEXT,
            font=("Segoe UI Semibold", 17)
        )

        style.configure(
            "PageTitle.TLabel",
            background=BG,
            foreground=TEXT,
            font=("Segoe UI Semibold", 21)
        )

        style.configure(
            "Section.TLabel",
            background=PANEL,
            foreground=TEXT,
            font=("Segoe UI Semibold", 11)
        )

        style.configure(
            "TNotebook",
            background=PANEL,
            borderwidth=0
        )

        style.configure(
            "TNotebook.Tab",
            background=PANEL_ALT,
            foreground=MUTED,
            padding=(15, 10),
            borderwidth=0
        )

        style.map(
            "TNotebook.Tab",
            background=[
                ("selected", "#343A48"),
                ("active", "#292E3A")
            ],
            foreground=[
                ("selected", TEXT),
                ("active", TEXT)
            ]
        )

        style.configure(
            "Treeview",
            background=PANEL,
            fieldbackground=PANEL,
            foreground=TEXT,
            borderwidth=0,
            rowheight=34,
            font=("Segoe UI", 9)
        )

        style.map(
            "Treeview",
            background=[
                ("selected", "#414959")
            ],
            foreground=[
                ("selected", TEXT)
            ]
        )

        style.configure(
            "Treeview.Heading",
            background=PANEL_ALT,
            foreground=TEXT,
            relief="flat",
            padding=9,
            font=("Segoe UI Semibold", 9)
        )

        style.map(
            "Treeview.Heading",
            background=[
                ("active", "#343A48"),
                ("pressed", "#343A48")
            ],
            foreground=[
                ("active", TEXT),
                ("pressed", TEXT)
            ]
        )

        style.configure(
            "TCombobox",
            fieldbackground=PANEL_ALT,
            background=PANEL_ALT,
            foreground=TEXT,
            arrowcolor=TEXT,
            padding=7,
            bordercolor=BORDER
        )

        style.map(
            "TCombobox",
            fieldbackground=[
                ("readonly", PANEL_ALT)
            ],
            foreground=[
                ("readonly", TEXT)
            ]
        )

        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=PANEL_ALT,
            background=PINK,
            borderwidth=0
        )

        style.configure(
            "Running.Horizontal.TProgressbar",
            troughcolor=PANEL_ALT,
            background=YELLOW
        )

        style.configure(
            "Success.Horizontal.TProgressbar",
            troughcolor=PANEL_ALT,
            background=GREEN
        )

        self.style = style

    # ========================================================
    # ЗМІННІ ІНТЕРФЕЙСУ
    # ========================================================

    def _create_variables(self):

        self.dataset_var = tk.StringVar(
            value=DATASET_NAMES["moons"]
        )

        self.all_datasets_var = tk.BooleanVar(value=False)

        self.optimizer_vars = {
            name: tk.BooleanVar(value=True)
            for name in OPTIMIZER_NAMES
        }

        self.learning_rate_var = tk.StringVar(value="0.01")
        self.epochs_var = tk.StringVar(value="200")
        self.batch_var = tk.StringVar(value="32")
        self.repeats_var = tk.StringVar(value="3")

        self.target_loss_var = tk.StringVar(value="0.05")
        self.patience_var = tk.StringVar(value="25")
        self.min_delta_var = tk.StringVar(value="0.0001")

        self.samples_var = tk.StringVar(value="600")
        self.seed_var = tk.StringVar(value="42")

        self.sweep_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(
            value="Готово до запуску"
        )

        self.progress_var = tk.DoubleVar(value=0)

        self.progress_text_var = tk.StringVar(
            value="0 / 0 запусків"
        )

        self.chart_dataset_var = tk.StringVar()

        self.run_selector_var = tk.StringVar()

        self.kpi_experiments = tk.StringVar(value="0")
        self.kpi_runs = tk.StringVar(value="0")
        self.kpi_accuracy = tk.StringVar(value="—")
        self.kpi_loss = tk.StringVar(value="—")

    # ========================================================
    # ДОПОМІЖНІ ВІЗУАЛЬНІ КОМПОНЕНТИ
    # ========================================================

    def _card(self, parent, padding=17):

        return tk.Frame(
            parent,
            bg=PANEL,
            padx=padding,
            pady=padding,
            highlightthickness=1,
            highlightbackground=BORDER
        )

    def _label(
        self,
        parent,
        text,
        size=10,
        color=TEXT,
        bold=False
    ):

        return tk.Label(
            parent,
            text=text,
            bg=parent.cget("bg"),
            fg=color,
            font=(
                "Segoe UI",
                size,
                "bold" if bold else "normal"
            ),
            anchor="w",
            justify="left",
            wraplength=900
        )

    def _button(
        self,
        parent,
        text,
        command,
        color=PANEL_ALT,
        foreground=TEXT
    ):

        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg=foreground,
            activebackground=PINK_HOVER,
            activeforeground="#210817",
            relief="flat",
            borderwidth=0,
            padx=16,
            pady=10,
            cursor="hand2",
            font=("Segoe UI Semibold", 9)
        )

        return button

    def _entry(self, parent, variable, width=16):

        return tk.Entry(
            parent,
            textvariable=variable,
            width=width,
            bg=PANEL_ALT,
            fg=TEXT,
            insertbackground=TEXT,
            selectbackground="#414959",
            selectforeground=TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=PINK,
            font=("Consolas", 10)
        )

    def _page_heading(self, parent, title, subtitle):

        frame = tk.Frame(parent, bg=BG)

        self._label(
            frame,
            title,
            21,
            TEXT,
            True
        ).pack(anchor="w")

        self._label(
            frame,
            subtitle,
            9,
            MUTED
        ).pack(anchor="w", pady=(5, 0))

        return frame

    # ========================================================
    # ГОЛОВНИЙ КАРКАС
    # ========================================================

    def _build_interface(self):

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._build_header()

        body = tk.Frame(self.root, bg=BG)

        body.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=17,
            pady=(0, 12)
        )

        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_sidebar(body)

        self.page_container = tk.Frame(
            body,
            bg=BG
        )

        self.page_container.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(17, 0)
        )

        self.page_container.columnconfigure(0, weight=1)
        self.page_container.rowconfigure(0, weight=1)

        self.pages = {}

        self._build_overview_page()
        self._build_experiment_page()
        self._build_results_page()
        self._build_model_page()
        self._build_datasets_page()
        self._build_help_page()

        self._build_footer()

        self.show_page("overview")

    # ========================================================
    # ВЕРХНЯ ПАНЕЛЬ
    # ========================================================

    def _build_header(self):

        header = tk.Frame(
            self.root,
            bg=BG,
            padx=25,
            pady=18
        )

        header.grid(
            row=0,
            column=0,
            sticky="ew"
        )

        header.columnconfigure(0, weight=1)

        self._label(
            header,
            PROJECT_TITLE,
            13,
            TEXT,
            True
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        self._label(
            header,
            "Python · NumPy · власний backpropagation · 4 оптимізатори",
            9,
            MUTED
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(5, 0)
        )

        self.status_badge = tk.Label(
            header,
            text="● ГОТОВО",
            bg="#303642",
            fg=TEXT,
            padx=15,
            pady=9,
            font=("Segoe UI Semibold", 9)
        )

        self.status_badge.grid(
            row=0,
            column=1,
            rowspan=2,
            sticky="e",
            padx=(15, 0)
        )

    # ========================================================
    # БІЧНА ПАНЕЛЬ
    # ========================================================

    def _build_sidebar(self, parent):

        self.sidebar = tk.Frame(
            parent,
            bg=SIDEBAR,
            width=230,
            highlightthickness=1,
            highlightbackground=BORDER
        )

        self.sidebar.grid(
            row=0,
            column=0,
            sticky="ns"
        )

        self.sidebar.grid_propagate(False)

        tk.Label(
            self.sidebar,
            text="НАВІГАЦІЯ",
            bg=SIDEBAR,
            fg=MUTED,
            font=("Segoe UI Semibold", 9)
        ).pack(
            anchor="w",
            padx=19,
            pady=(23, 15)
        )

        self.nav_buttons = {}

        navigation = [
            ("overview", "⌂   Огляд"),
            ("experiment", "＋   Експерименти"),
            ("results", "▤   Результати"),
            ("model", "◇   Архітектура"),
            ("datasets", "◉   Набори даних"),
            ("help", "?   Довідка")
        ]

        for key, title in navigation:

            button = tk.Button(
                self.sidebar,
                text=title,
                anchor="w",
                command=lambda k=key: self.show_page(k),
                bg=SIDEBAR,
                fg=MUTED,
                activebackground="#292E3A",
                activeforeground=TEXT,
                relief="flat",
                borderwidth=0,
                padx=20,
                pady=13,
                font=("Segoe UI Semibold", 10),
                cursor="hand2"
            )

            button.pack(
                fill="x",
                padx=8,
                pady=2
            )

            self.nav_buttons[key] = button

        spacer = tk.Frame(
            self.sidebar,
            bg=SIDEBAR
        )

        spacer.pack(
            fill="both",
            expand=True
        )

        tk.Frame(
            self.sidebar,
            bg=BORDER,
            height=1
        ).pack(
            fill="x",
            padx=15
        )

        tk.Label(
            self.sidebar,
            text="ЛОКАЛЬНЕ ДОСЛІДЖЕННЯ\n\n"
                 "Виконали:\n"
                 "Бровінська В.С / Лущик А.С\n\n"
                 "1 курс спец. F3 Комп'ютерні науки, група ІН-102\n\n"
                 "© 2026, КНЕУ ім. Вадима Гетьмана",
            bg=SIDEBAR,
            fg=MUTED,
            justify="left",
            font=("Segoe UI", 9)
        ).pack(
            anchor="w",
            padx=19,
            pady=18
        )

    # ========================================================
    # ПЕРЕМИКАННЯ СТОРІНОК
    # ========================================================

    def _create_page(self, key):

        page = tk.Frame(
            self.page_container,
            bg=BG
        )

        page.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        self.pages[key] = page

        return page

    def show_page(self, key):

        self.current_page = key

        self.pages[key].tkraise()

        for name, button in self.nav_buttons.items():

            if name == key:

                button.configure(
                    bg="#303746",
                    fg=PINK
                )

            else:

                button.configure(
                    bg=SIDEBAR,
                    fg=MUTED
                )

    # ========================================================
    # ОГЛЯД
    # ========================================================

    def _build_overview_page(self):

        page = self._create_page("overview")

        top = self._page_heading(
            page,
            "Огляд дослідження",
            "Загальний стан експериментів та основні результати"
        )

        top.pack(
            anchor="w",
            fill="x",
            pady=(8, 20)
        )

        cards = tk.Frame(page, bg=BG)

        cards.pack(
            fill="x",
            pady=(0, 18)
        )

        for col in range(4):

            cards.columnconfigure(col, weight=1)

        self._metric_card(
            cards, 0,
            "СЕРІЇ ЕКСПЕРИМЕНТІВ",
            self.kpi_experiments,
            "Збережені запуски",
            PINK
        )

        self._metric_card(
            cards, 1,
            "УСПІШНІ ЗАПУСКИ",
            self.kpi_runs,
            "Фактичні обчислення",
            GREEN
        )

        self._metric_card(
            cards, 2,
            "СЕРЕДНЯ ТОЧНІСТЬ",
            self.kpi_accuracy,
            "Тестова вибірка",
            BLUE
        )

        self._metric_card(
            cards, 3,
            "НАЙМЕНША BCE",
            self.kpi_loss,
            "Тестова функція втрат",
            ROSE
        )

        lower = tk.Frame(page, bg=BG)

        lower.pack(
            fill="both",
            expand=True
        )

        lower.columnconfigure(0, weight=3)
        lower.columnconfigure(1, weight=2)
        lower.rowconfigure(0, weight=1)

        history_card = self._card(lower)

        history_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 9)
        )

        self._label(
            history_card,
            "ІСТОРІЯ ЕКСПЕРИМЕНТІВ",
            11,
            TEXT,
            True
        ).pack(
            anchor="w",
            pady=(0, 13)
        )

        columns = (
            "date",
            "datasets",
            "runs",
            "status"
        )

        self.history_tree = ttk.Treeview(
            history_card,
            columns=columns,
            show="headings",
            height=11
        )

        headings = {
            "date": ("Дата запуску", 145),
            "datasets": ("Набори даних", 190),
            "runs": ("Запуски", 80),
            "status": ("Статус", 110)
        }

        for col in columns:

            title, width = headings[col]

            self.history_tree.heading(
                col,
                text=title
            )

            self.history_tree.column(
                col,
                width=width,
                anchor="center"
            )

        self.history_tree.pack(
            fill="both",
            expand=True
        )

        self.history_tree.bind(
            "<Double-1>",
            self._open_history_selection
        )

        right_card = self._card(lower)

        right_card.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(9, 0)
        )

        self._label(
            right_card,
            "ІНФОРМАЦІЯ ПРО ПРОЄКТ",
            11,
            TEXT,
            True
        ).pack(
            anchor="w",
            pady=(0, 15)
        )

        information = [
            ("Математична модель", "2 → 16 → 1"),
            ("Прихований шар", "ReLU"),
            ("Вихідний шар", "Sigmoid"),
            ("Функція втрат", "Binary Cross-Entropy"),
            ("Метод градієнтів", "Backpropagation"),
            ("Оптимізатори", "SGD · Momentum · RMSProp · Adam"),
            ("Набори даних", "Moons · Circles · XOR")
        ]

        for name, value in information:

            block = tk.Frame(
                right_card,
                bg=PANEL
            )

            block.pack(
                fill="x",
                pady=5
            )

            self._label(
                block,
                name,
                9,
                MUTED
            ).pack(
                anchor="w"
            )

            self._label(
                block,
                value,
                10,
                TEXT,
                True
            ).pack(
                anchor="w",
                pady=(2, 0)
            )

        self._button(
            right_card,
            "＋  Новий експеримент",
            lambda: self.show_page("experiment"),
            PINK,
            "#210817"
        ).pack(
            fill="x",
            pady=(20, 0)
        )

    def _metric_card(
        self,
        parent,
        col,
        title,
        variable,
        subtitle,
        color
    ):

        card = self._card(parent, 14)

        card.grid(
            row=0,
            column=col,
            sticky="ew",
            padx=5
        )

        tk.Frame(
            card,
            bg=color,
            height=3
        ).pack(
            fill="x",
            pady=(0, 12)
        )

        self._label(
            card,
            title,
            8,
            MUTED,
            True
        ).pack(
            anchor="w"
        )

        tk.Label(
            card,
            textvariable=variable,
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 23)
        ).pack(
            anchor="w",
            pady=7
        )

        self._label(
            card,
            subtitle,
            8,
            MUTED
        ).pack(
            anchor="w"
        )

    # ========================================================
    # ЕКСПЕРИМЕНТИ
    # ========================================================

    def _build_experiment_page(self):

        page = self._create_page("experiment")

        header = self._page_heading(
            page,
            "Новий експеримент",
            "Налаштування контрольованого порівняння алгоритмів"
        )

        header.pack(
            anchor="w",
            pady=(8, 20)
        )

        content = tk.Frame(page, bg=BG)

        content.pack(
            fill="both",
            expand=True
        )

        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)

        # Ліва панель налаштувань

        settings = self._card(content)

        settings.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 9)
        )

        # Внутрішня прокручувана область: усі параметри, включно із Seed,
        # залишаються доступними навіть на екранах із невеликою висотою.
        settings_canvas = tk.Canvas(
            settings,
            bg=PANEL,
            highlightthickness=0,
            borderwidth=0
        )

        settings_scrollbar = ttk.Scrollbar(
            settings,
            orient="vertical",
            command=settings_canvas.yview
        )

        settings_canvas.configure(
            yscrollcommand=settings_scrollbar.set
        )

        settings_scrollbar.pack(
            side="right",
            fill="y"
        )

        settings_canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        settings_body = tk.Frame(
            settings_canvas,
            bg=PANEL
        )

        settings_window = settings_canvas.create_window(
            (0, 0),
            window=settings_body,
            anchor="nw"
        )

        def _update_settings_scrollregion(_event=None):
            settings_canvas.configure(
                scrollregion=settings_canvas.bbox("all")
            )

        def _resize_settings_body(event):
            settings_canvas.itemconfigure(
                settings_window,
                width=event.width
            )

        settings_body.bind(
            "<Configure>",
            _update_settings_scrollregion
        )

        settings_canvas.bind(
            "<Configure>",
            _resize_settings_body
        )

        def _scroll_settings(event):
            if getattr(event, "delta", 0):
                settings_canvas.yview_scroll(
                    int(-event.delta / 120),
                    "units"
                )

        def _bind_settings_wheel(_event):
            settings_canvas.bind_all(
                "<MouseWheel>",
                _scroll_settings
            )

        def _unbind_settings_wheel(_event):
            settings_canvas.unbind_all("<MouseWheel>")

        settings_canvas.bind("<Enter>", _bind_settings_wheel)
        settings_canvas.bind("<Leave>", _unbind_settings_wheel)

        self._label(
            settings_body,
            "ПАРАМЕТРИ ДОСЛІДЖЕННЯ",
            11,
            TEXT,
            True
        ).pack(
            anchor="w",
            pady=(0, 16)
        )

        self._label(
            settings_body,
            "Набір даних",
            9,
            MUTED
        ).pack(
            anchor="w"
        )

        self.dataset_combo = ttk.Combobox(
            settings_body,
            textvariable=self.dataset_var,
            values=list(DATASET_NAMES.values()),
            state="readonly"
        )

        self.dataset_combo.pack(
            fill="x",
            pady=(5, 7)
        )

        self._check(
            settings_body,
            "Порівняти всі три геометрії",
            self.all_datasets_var
        )

        self._label(
            settings_body,
            "Оптимізатори",
            9,
            MUTED
        ).pack(
            anchor="w",
            pady=(15, 7)
        )

        optimizer_frame = tk.Frame(
            settings_body,
            bg=PANEL
        )

        optimizer_frame.pack(
            fill="x"
        )

        for i, name in enumerate(OPTIMIZER_NAMES):

            tk.Checkbutton(
                optimizer_frame,
                text=name,
                variable=self.optimizer_vars[name],
                bg=PANEL,
                fg=OPTIMIZER_COLORS[name],
                activebackground=PANEL,
                activeforeground=OPTIMIZER_COLORS[name],
                selectcolor=PANEL_ALT,
                font=("Segoe UI Semibold", 10)
            ).grid(
                row=i // 2,
                column=i % 2,
                sticky="w",
                padx=(0, 25),
                pady=4
            )

        self._label(
            settings_body,
            "Гіперпараметри",
            9,
            MUTED
        ).pack(
            anchor="w",
            pady=(18, 8)
        )

        fields = [
            ("Learning rate", self.learning_rate_var),
            ("Максимум епох", self.epochs_var),
            ("Розмір мініпакета", self.batch_var),
            ("Кількість повторів", self.repeats_var),
            ("Цільова BCE", self.target_loss_var),
            ("Терпіння", self.patience_var),
            ("Min delta", self.min_delta_var),
            ("Кількість зразків", self.samples_var),
            ("Seed", self.seed_var)
        ]

        self.settings_entries = {}

        grid = tk.Frame(
            settings_body,
            bg=PANEL
        )

        grid.pack(
            fill="x"
        )

        for i, (label, variable) in enumerate(fields):

            row = i // 2
            col = i % 2

            cell = tk.Frame(
                grid,
                bg=PANEL
            )

            cell.grid(
                row=row,
                column=col,
                sticky="ew",
                padx=(0, 12),
                pady=7
            )

            grid.columnconfigure(
                col,
                weight=1
            )

            self._label(
                cell,
                label,
                9,
                MUTED
            ).pack(
                anchor="w",
                pady=(0, 5)
            )

            entry = self._entry(
                cell,
                variable,
                15
            )

            entry.pack(
                fill="x",
                ipady=5
            )

            self.settings_entries[label] = entry

        self._check(
            settings_body,
            "Дослідити різні швидкості навчання (×0.1, ×1, ×10)",
            self.sweep_var
        )

        # Права панель запуску

        launch = self._card(content)

        launch.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(9, 0)
        )

        self._label(
            launch,
            "ЗАПУСК ОБЧИСЛЕНЬ",
            11,
            TEXT,
            True
        ).pack(
            anchor="w",
            pady=(0, 17)
        )

        self._label(
            launch,
            "Експерименти виконуються локально у фоновому потоці. Вікно залишатиметься доступним під час навчання.",
            10,
            MUTED
        ).pack(
            anchor="w",
            pady=(0, 20)
        )

        self.run_button = self._button(
            launch,
            "▶  ПОЧАТИ ЕКСПЕРИМЕНТ",
            self._start_experiment,
            PINK,
            "#210817"
        )

        self.run_button.pack(
            fill="x",
            pady=(0, 9)
        )

        self.cancel_button = self._button(
            launch,
            "Скасувати запуск",
            self._cancel_experiment,
            PANEL_ALT
        )

        self.cancel_button.pack(
            fill="x"
        )

        self.cancel_button.configure(
            state="disabled"
        )

        tk.Frame(
            launch,
            bg=BORDER,
            height=1
        ).pack(
            fill="x",
            pady=22
        )

        self._label(
            launch,
            "ЩО БУДЕ ВИКОНАНО",
            9,
            MUTED,
            True
        ).pack(
            anchor="w",
            pady=(0, 10)
        )

        notes = [
            "Навчання нейронної мережі",
            "Порівняння чотирьох оптимізаторів",
            "Обчислення тестових метрик",
            "Аналіз критерію збіжності",
            "Формування графіків",
            "Збереження CSV та JSON",
            "Створення текстового звіту"
        ]

        for item in notes:

            self._label(
                launch,
                "✓  " + item,
                9,
                GREEN
            ).pack(
                anchor="w",
                pady=4
            )

        tk.Frame(
            launch,
            bg=BORDER,
            height=1
        ).pack(
            fill="x",
            pady=22
        )

        self._label(
            launch,
            "Результати попередніх запусків не видаляються.",
            9,
            MUTED
        ).pack(
            anchor="w"
        )

    def _check(self, parent, text, variable):

        tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            bg=PANEL,
            fg=TEXT,
            activebackground=PANEL,
            activeforeground=TEXT,
            selectcolor=PANEL_ALT,
            font=("Segoe UI", 9),
            anchor="w",
            justify="left",
            wraplength=500
        ).pack(
            anchor="w",
            pady=5
        )

    # ========================================================
    # РЕЗУЛЬТАТИ
    # ========================================================

    def _build_results_page(self):

        page = self._create_page("results")

        header = self._page_heading(
            page,
            "Результати дослідження",
            "Таблиці, метрики та візуальний аналіз фактичних експериментів"
        )

        header.pack(
            anchor="w",
            pady=(8, 16)
        )

        selector = self._card(page, 11)

        selector.pack(
            fill="x",
            pady=(0, 12)
        )

        self._label(
            selector,
            "Вибраний експеримент",
            9,
            MUTED
        ).pack(
            side="left",
            padx=(0, 12)
        )

        self.run_combo = ttk.Combobox(
            selector,
            textvariable=self.run_selector_var,
            state="readonly",
            width=42
        )

        self.run_combo.pack(
            side="left",
            fill="x",
            expand=True
        )

        self.run_combo.bind(
            "<<ComboboxSelected>>",
            self._on_run_selected
        )

        self.results_notebook = ttk.Notebook(page)

        self.results_notebook.pack(
            fill="both",
            expand=True
        )

        table_page = tk.Frame(
            self.results_notebook,
            bg=PANEL,
            padx=10,
            pady=10
        )

        plots_page = tk.Frame(
            self.results_notebook,
            bg=PANEL,
            padx=10,
            pady=10
        )

        self.results_notebook.add(
            table_page,
            text="Таблиця метрик"
        )

        self.results_notebook.add(
            plots_page,
            text="Аналітичні графіки"
        )

        self._build_results_table(table_page)
        self._build_plot_grid(plots_page)

    def _open_graph_window(self, image_path: str | Path, title: str) -> None:
        """Відкриває вибраний графік в окремому великому вікні."""

        path = Path(image_path)

        if not path.exists():
            messagebox.showerror(
                "Графік не знайдено",
                f"Не вдалося знайти файл:\n{path}",
                parent=self.root,
            )
            return

        try:
            image = imread(str(path))
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                "Помилка відкриття графіка",
                str(exc),
                parent=self.root,
            )
            return

        window = tk.Toplevel(self.root)
        window.title(f"{title} — повний перегляд")
        window.configure(bg=BG)
        window.geometry("1300x850")
        window.minsize(850, 600)

        try:
            window.state("zoomed")
        except tk.TclError:
            pass

        holder = tk.Frame(window, bg=BG)
        holder.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10,
        )

        figure = Figure(
            figsize=(14, 9),
            dpi=110,
            facecolor=BG,
        )
        axes = figure.add_subplot(111)
        axes.imshow(image, interpolation="lanczos")
        axes.set_axis_off()

        figure.subplots_adjust(
            left=0,
            right=1,
            top=1,
            bottom=0,
        )

        canvas = FigureCanvasTkAgg(figure, master=holder)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(
            canvas,
            holder,
            pack_toolbar=False,
        )
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        canvas.get_tk_widget().pack(
            side="top",
            fill="both",
            expand=True,
        )

        # Зберігаємо посилання, щоб Tkinter/Matplotlib не видалили об'єкти.
        window.graph_figure = figure
        window.graph_canvas = canvas
        window.graph_toolbar = toolbar

    def _build_results_table(self, parent):

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        columns = (
            "dataset",
            "optimizer",
            "lr",
            "train",
            "minimum",
            "test",
            "accuracy",
            "f1",
            "epochs",
            "time",
            "status"
        )

        self.result_tree = ttk.Treeview(
            parent,
            columns=columns,
            show="headings",
            height=17
        )

        definitions = {
            "dataset": ("Набір даних", 125),
            "optimizer": ("Оптимізатор", 105),
            "lr": ("LR", 75),
            "train": ("Train BCE", 95),
            "minimum": ("Min BCE", 95),
            "test": ("Test BCE", 95),
            "accuracy": ("Accuracy", 90),
            "f1": ("F1", 75),
            "epochs": ("Епохи до цілі", 120),
            "time": ("Час, с", 85),
            "status": ("Статус", 115)
        }

        for col in columns:

            title, width = definitions[col]

            self.result_tree.heading(
                col,
                text=title,
                command=lambda c=col: self._sort_table(c)
            )

            self.result_tree.column(
                col,
                width=width,
                minwidth=65,
                anchor="center",
                stretch=False
            )

        self.result_tree.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        scroll_y = ttk.Scrollbar(
            parent,
            orient="vertical",
            command=self.result_tree.yview
        )

        scroll_x = ttk.Scrollbar(
            parent,
            orient="horizontal",
            command=self.result_tree.xview
        )

        self.result_tree.configure(
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set
        )

        scroll_y.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        scroll_x.grid(
            row=1,
            column=0,
            sticky="ew"
        )

        for name, colors in OPTIMIZER_ROW_COLORS.items():

            self.result_tree.tag_configure(
                name,
                background=colors[0],
                foreground=colors[1]
            )

        self.result_tree.tag_configure(
            "bad",
            background="#41131F",
            foreground="#FFE0E8"
        )

        self.result_tree.tag_configure(
            "neutral",
            background=PANEL_ALT,
            foreground=MUTED
        )

    # ========================================================
    # ГРАФІКИ
    # ========================================================

    def _build_plot_grid(self, parent):

        top = tk.Frame(
            parent,
            bg=PANEL
        )

        top.pack(
            fill="x",
            pady=(0, 10)
        )

        self._label(
            top,
            "Геометрія набору",
            9,
            MUTED
        ).pack(
            side="left",
            padx=(0, 10)
        )

        self.chart_combo = ttk.Combobox(
            top,
            textvariable=self.chart_dataset_var,
            state="readonly",
            width=28
        )

        self.chart_combo.pack(
            side="left"
        )

        self.chart_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._render_charts()
        )

        self.chart_frame = tk.Frame(
            parent,
            bg=PANEL
        )

        self.chart_frame.pack(
            fill="both",
            expand=True
        )

        self.chart_placeholder = tk.Label(
            self.chart_frame,
            text="Після проведення експерименту тут з'являться чотири графіки.",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 12)
        )

        self.chart_placeholder.pack(
            expand=True
        )

    def _render_charts(self):
        """Показує чотири великі графіки у сітці 2×2.

        Зображення масштабується до доступної області картки, зокрема
        збільшується, якщо вихідний PNG має менший розмір. Для детального
        перегляду кожен графік можна відкрити в окремому вікні.
        """

        if self.current_dir is None or not self.current_dir.exists():
            return

        dataset_title = self.chart_dataset_var.get()
        reverse = {
            title: key
            for key, title in DATASET_NAMES.items()
        }
        dataset = reverse.get(dataset_title)

        if dataset is None:
            return

        plots_dir = self.current_dir / "plots"
        definitions = [
            ("loss", "Зміна функції втрат"),
            ("accuracy", "Точність класифікації"),
            ("convergence", "Критерій збіжності"),
            ("decision_boundary", "Межа рішень"),
        ]

        # Видаляємо картки попереднього набору даних.
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        self.chart_photo_images = []
        self.chart_frame.update_idletasks()

        frame_width = max(900, self.chart_frame.winfo_width())
        frame_height = max(440, self.chart_frame.winfo_height())

        # Доступна площа для одного графіка. Висота враховує заголовок
        # картки та кнопку, щоб усі чотири графіки залишалися в сітці 2×2.
        preview_width = max(
            420,
            min(1100, frame_width // 2 - 34),
        )
        preview_height = max(
            170,
            min(235, frame_height // 2 - 42),
        )

        self.chart_frame.grid_columnconfigure(0, weight=1, uniform="chart_columns")
        self.chart_frame.grid_columnconfigure(1, weight=1, uniform="chart_columns")
        self.chart_frame.grid_rowconfigure(0, weight=1, uniform="chart_rows")
        self.chart_frame.grid_rowconfigure(1, weight=1, uniform="chart_rows")

        for index, (prefix, chart_title) in enumerate(definitions):
            plot_path = plots_dir / f"{prefix}_{dataset}.png"

            chart_card = tk.Frame(
                self.chart_frame,
                bg=PANEL,
                highlightbackground=BORDER,
                highlightthickness=1,
            )
            chart_card.grid(
                row=index // 2,
                column=index % 2,
                sticky="nsew",
                padx=7,
                pady=7,
            )
            chart_card.grid_columnconfigure(0, weight=1)
            chart_card.grid_rowconfigure(1, weight=1)

            # Заголовок і кнопка знаходяться в одному рядку, тому графік
            # отримує більше вертикального простору.
            card_header = tk.Frame(chart_card, bg=PANEL)
            card_header.grid(
                row=0,
                column=0,
                sticky="ew",
                padx=10,
                pady=(7, 3),
            )
            card_header.grid_columnconfigure(0, weight=1)

            tk.Label(
                card_header,
                text=chart_title,
                bg=PANEL,
                fg=TEXT,
                font=("Segoe UI", 11, "bold"),
                anchor="w",
            ).grid(
                row=0,
                column=0,
                sticky="w",
            )

            open_button = tk.Button(
                card_header,
                text="↗ Відкрити окремо",
                command=lambda p=plot_path, t=chart_title:
                    self._open_graph_window(p, t),
                bg=PANEL_ALT,
                fg=TEXT,
                activebackground=PINK_HOVER,
                activeforeground=TEXT,
                relief="flat",
                cursor="hand2",
                font=("Segoe UI", 9),
                padx=10,
                pady=4,
                state="normal" if plot_path.exists() else "disabled",
            )
            open_button.grid(
                row=0,
                column=1,
                sticky="e",
                padx=(8, 0),
            )

            image_holder = tk.Frame(
                chart_card,
                bg=BG,
            )
            image_holder.grid(
                row=1,
                column=0,
                sticky="nsew",
                padx=9,
                pady=(2, 9),
            )
            image_holder.grid_columnconfigure(0, weight=1)
            image_holder.grid_rowconfigure(0, weight=1)

            if plot_path.exists():
                try:
                    if Image is not None and ImageTk is not None:
                        with Image.open(plot_path) as source_image:
                            source_image = source_image.convert("RGBA")
                            source_width, source_height = source_image.size

                            # На відміну від thumbnail(), resize() також
                            # збільшує невеликі PNG до розміру картки.
                            scale = min(
                                preview_width / max(1, source_width),
                                preview_height / max(1, source_height),
                            )
                            target_size = (
                                max(1, round(source_width * scale)),
                                max(1, round(source_height * scale)),
                            )
                            resampling = getattr(Image, "Resampling", Image)
                            source_image = source_image.resize(
                                target_size,
                                resampling.LANCZOS,
                            )
                            preview = ImageTk.PhotoImage(
                                source_image,
                                master=self.root,
                            )
                    else:
                        # Резервний режим без Pillow. Масштабування Tkinter
                        # менш плавне, але не потребує додаткових бібліотек.
                        original = tk.PhotoImage(
                            master=self.root,
                            file=str(plot_path),
                        )
                        scale = min(
                            max(1, preview_width // max(1, original.width())),
                            max(1, preview_height // max(1, original.height())),
                        )
                        preview = (
                            original.zoom(scale, scale)
                            if scale > 1
                            else original
                        )

                        downscale = max(
                            1,
                            math.ceil(preview.width() / preview_width),
                            math.ceil(preview.height() / preview_height),
                        )
                        if downscale > 1:
                            preview = preview.subsample(downscale, downscale)

                    image_label = tk.Label(
                        image_holder,
                        image=preview,
                        bg=BG,
                        bd=0,
                        anchor="center",
                    )
                    image_label.grid(
                        row=0,
                        column=0,
                        sticky="nsew",
                    )

                    # Зберігаємо посилання на PhotoImage, інакше Tkinter
                    # може прибрати зображення після завершення методу.
                    image_label.image = preview
                    self.chart_photo_images.append(preview)

                except (OSError, ValueError, tk.TclError) as exc:
                    tk.Label(
                        image_holder,
                        text=f"Не вдалося показати графік:\n{exc}",
                        bg=BG,
                        fg=RED,
                        font=("Segoe UI", 9),
                        justify="left",
                        wraplength=preview_width,
                    ).grid(
                        row=0,
                        column=0,
                        sticky="nsew",
                        padx=12,
                        pady=12,
                    )
            else:
                tk.Label(
                    image_holder,
                    text="Графік недоступний для цього експерименту.",
                    bg=BG,
                    fg=MUTED,
                    font=("Segoe UI", 10),
                    wraplength=preview_width,
                ).grid(
                    row=0,
                    column=0,
                    sticky="nsew",
                    padx=12,
                    pady=12,
                )

    # ========================================================
    # АРХІТЕКТУРА МОДЕЛІ
    # ========================================================

    def _build_model_page(self):

        page = self._create_page("model")

        heading = self._page_heading(
            page,
            "Архітектура нейронної мережі",
            "Повнозв'язна мережа для бінарної класифікації"
        )

        heading.pack(
            anchor="w",
            pady=(8, 18)
        )

        card = self._card(page)

        card.pack(
            fill="both",
            expand=True
        )

        self._label(
            card,
            "МАТЕМАТИЧНА СТРУКТУРА",
            11,
            TEXT,
            True
        ).pack(
            anchor="w",
            pady=(0, 13)
        )

        canvas = tk.Canvas(
            card,
            bg=PANEL,
            height=540,
            highlightthickness=0
        )

        canvas.pack(
            fill="both",
            expand=True
        )

        input_x = 100
        hidden_x = 430
        output_x = 760

        hidden_y = [
            65 + i * 27
            for i in range(16)
        ]

        input_y = [230, 330]
        output_y = 280

        # Зв'язки вхідного шару з прихованим

        for y1 in input_y:

            for y2 in hidden_y:

                canvas.create_line(
                    input_x + 13,
                    y1,
                    hidden_x - 10,
                    y2,
                    fill="#344052",
                    width=1
                )

        # Зв'язки прихованого шару з вихідним

        for y in hidden_y:

            canvas.create_line(
                hidden_x + 10,
                y,
                output_x - 14,
                output_y,
                fill="#344052",
                width=1
            )

        def draw_node(x, y, color, radius=11):

            canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=color,
                outline=color,
                width=1
            )

        for y in input_y:
            draw_node(input_x, y, BLUE, 16)

        for y in hidden_y:
            draw_node(hidden_x, y, PINK, 9)

        draw_node(output_x, output_y, GREEN, 17)

        canvas.create_text(
            input_x,
            395,
            text="ВХІДНИЙ ШАР\n2 ознаки",
            fill=TEXT,
            font=("Segoe UI Semibold", 11),
            justify="center"
        )

        canvas.create_text(
            hidden_x,
            525,
            text="ПРИХОВАНИЙ ШАР\n16 нейронів · ReLU",
            fill=TEXT,
            font=("Segoe UI Semibold", 11),
            justify="center"
        )

        canvas.create_text(
            output_x,
            335,
            text="ВИХІДНИЙ ШАР\n1 нейрон · Sigmoid",
            fill=TEXT,
            font=("Segoe UI Semibold", 11),
            justify="center"
        )

        self._label(
            card,
            "Архітектура: 2 → 16 → 1",
            12,
            PINK,
            True
        ).pack(
            anchor="w",
            pady=(15, 5)
        )

        self._label(
            card,
            "Під час прямого поширення сигнал проходить через "
            "прихований шар ReLU, після чого вихідний "
            "нейрон Sigmoid формує ймовірність класу 1.",
            9,
            MUTED
        ).pack(
            anchor="w"
        )

    # ========================================================
    # НАБОРИ ДАНИХ
    # ========================================================

    def _build_datasets_page(self):

        page = self._create_page("datasets")

        heading = self._page_heading(
            page,
            "Набори даних",
            "Нелінійно роздільні синтетичні задачі класифікації"
        )

        heading.pack(
            anchor="w",
            pady=(8, 20)
        )

        definitions = [
            (
                "MOONS",
                "Два півмісяці",
                "Два вигнуті класи. Геометрія потребує нелінійної межі рішень.",
                BLUE
            ),
            (
                "CIRCLES",
                "Концентричні кола",
                "Внутрішнє та зовнішнє кільця. Лінійного розділення недостатньо.",
                CYAN
            ),
            (
                "XOR",
                "XOR-кластери",
                "Чотири просторові області з XOR-розміткою класів.",
                ROSE
            )
        ]

        for code, title, description, color in definitions:

            card = self._card(page)

            card.pack(
                fill="x",
                pady=8
            )

            tk.Label(
                card,
                text=code,
                bg=PANEL,
                fg=color,
                font=("Segoe UI Semibold", 12)
            ).pack(
                anchor="w"
            )

            self._label(
                card,
                title,
                15,
                TEXT,
                True
            ).pack(
                anchor="w",
                pady=(7, 4)
            )

            self._label(
                card,
                description,
                10,
                MUTED
            ).pack(
                anchor="w"
            )

        note = self._card(page)

        note.pack(
            fill="x",
            pady=12
        )

        self._label(
            note,
            "Підготовка даних",
            11,
            TEXT,
            True
        ).pack(
            anchor="w"
        )

        self._label(
            note,
            "Для навчальної вибірки параметри стандартизації "
            "обчислюються окремо. Тестова вибірка не "
            "використовується для оновлення ваг мережі.",
            9,
            MUTED
        ).pack(
            anchor="w",
            pady=(7, 0)
        )

    # ========================================================
    # ДОВІДКА
    # ========================================================

    def _build_help_page(self):

        page = self._create_page("help")

        heading = self._page_heading(
            page,
            "Довідка",
            "Основні поняття та правила інтерпретації результатів"
        )

        heading.pack(
            anchor="w",
            pady=(8, 18)
        )

        sections = [
            (
                "ФУНКЦІЯ ВТРАТ BCE",
                "Бінарна перехресна ентропія вимірює "
                "відмінність між передбаченими ймовірностями "
                "та правильними мітками класів. Менше "
                "значення означає кращу якість імовірнісного прогнозу."
            ),
            (
                "SGD",
                "Оновлює параметри безпосередньо за градієнтом "
                "функції втрат."
            ),
            (
                "MOMENTUM",
                "Враховує накопичений напрям попередніх "
                "оновлень параметрів."
            ),
            (
                "RMSPROP",
                "Використовує рухоме середнє квадратів "
                "градієнтів для адаптації кроку."
            ),
            (
                "ADAM",
                "Поєднує оцінки першого та другого моментів "
                "градієнта, а також поправки початкового зміщення."
            ),
            (
                "ЗБІЖНІСТЬ",
                "Досягнення заданого порога BCE відрізняється "
                "від зупинки через відсутність покращення. "
                "Якщо поріг не виконано, програма явно це позначає."
            ),
            (
                "ПОВТОРЮВАНІСТЬ",
                "Фіксовані початкові значення генератора "
                "випадкових чисел дозволяють відтворювати "
                "експерименти за однакових параметрів."
            )
        ]

        for title, description in sections:

            card = self._card(page, 13)

            card.pack(
                fill="x",
                pady=5
            )

            self._label(
                card,
                title,
                10,
                PINK,
                True
            ).pack(
                anchor="w"
            )

            self._label(
                card,
                description,
                9,
                TEXT
            ).pack(
                anchor="w",
                pady=(6, 0)
            )

    # ========================================================
    # НИЖНЯ ПАНЕЛЬ
    # ========================================================

    def _build_footer(self):

        footer = tk.Frame(
            self.root,
            bg=BG,
            padx=22,
            pady=10
        )

        footer.grid(
            row=2,
            column=0,
            sticky="ew"
        )

        footer.columnconfigure(0, weight=1)

        self.status_label = tk.Label(
            footer,
            textvariable=self.status_var,
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9)
        )

        self.status_label.grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 7)
        )

        self.progress = ttk.Progressbar(
            footer,
            variable=self.progress_var,
            maximum=100,
            style="Horizontal.TProgressbar"
        )

        self.progress.grid(
            row=1,
            column=0,
            sticky="ew"
        )

        tk.Label(
            footer,
            textvariable=self.progress_text_var,
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9)
        ).grid(
            row=1,
            column=1,
            padx=(12, 0)
        )

    # ========================================================
    # НАЛАШТУВАННЯ КОНФІГУРАЦІЇ
    # ========================================================

    def _read_config(self):

        reverse = {
            value: key
            for key, value in DATASET_NAMES.items()
        }

        if self.all_datasets_var.get():

            datasets = list(DATASET_NAMES)

        else:

            selected = self.dataset_var.get()

            if selected not in reverse:
                raise ValueError("Виберіть набір даних.")

            datasets = [reverse[selected]]

        optimizers = [
            name
            for name, variable in self.optimizer_vars.items()
            if variable.get()
        ]

        try:

            config = {
                "datasets": datasets,
                "optimizers": optimizers,

                "learning_rate": float(
                    self.learning_rate_var.get()
                ),

                "learning_rate_sweep": self.sweep_var.get(),

                "max_epochs": int(
                    self.epochs_var.get()
                ),

                "batch_size": int(
                    self.batch_var.get()
                ),

                "repeats": int(
                    self.repeats_var.get()
                ),

                "target_loss": float(
                    self.target_loss_var.get()
                ),

                "patience": int(
                    self.patience_var.get()
                ),

                "min_delta": float(
                    self.min_delta_var.get()
                ),

                "seed": int(
                    self.seed_var.get()
                ),

                "n_samples": int(
                    self.samples_var.get()
                )
            }

        except ValueError as exc:

            raise ValueError(
                "Перевірте числові параметри."
            ) from exc

        return validate_config(config)

    # ========================================================
    # ЗАПУСК ЕКСПЕРИМЕНТУ
    # ========================================================

    def _start_experiment(self):

        if self.is_running:
            return

        try:

            config = self._read_config()

        except ValueError as exc:

            messagebox.showerror(
                "Помилка параметрів",
                str(exc),
                parent=self.root
            )

            return

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.cancel_event = threading.Event()

        self.is_running = True

        self.run_button.configure(
            state="disabled",
            text="ОБЧИСЛЕННЯ ТРИВАЮТЬ…",
            bg=YELLOW,
            fg="#111111"
        )

        self.cancel_button.configure(
            state="normal"
        )

        self.status_badge.configure(
            text="● ВИКОНУЄТЬСЯ",
            bg=YELLOW,
            fg="#111111"
        )

        self.progress_var.set(0)

        self.progress.configure(
            style="Running.Horizontal.TProgressbar"
        )

        self.progress_text_var.set(
            "0 / підготовка"
        )

        self.status_var.set(
            "Підготовка даних та нейронної мережі..."
        )

        cancel_event = self.cancel_event

        def worker_function():

            try:

                result = run_experiments(
                    config,
                    output_root=RESULTS_DIR,
                    progress_callback=lambda done, total, msg: (
                        self.events.put(
                            ("progress", done, total, msg)
                        )
                    ),
                    cancel_event=cancel_event
                )

                self.events.put(
                    ("done", result)
                )

            except Exception as exc:

                self.events.put(
                    (
                        "error",
                        type(exc).__name__,
                        str(exc)
                    )
                )

        self.worker = threading.Thread(
            target=worker_function,
            daemon=True,
            name="ExperimentWorker"
        )

        self.worker.start()

    # ========================================================
    # СКАСУВАННЯ
    # ========================================================

    def _cancel_experiment(self):

        if self.cancel_event is not None:

            if self.is_running:

                self.cancel_event.set()

                self.cancel_button.configure(
                    state="disabled"
                )

                self.status_var.set(
                    "Надіслано запит на скасування..."
                )

    # ========================================================
    # ОБРОБКА ПОДІЙ ФОНОВОГО ПОТОКУ
    # ========================================================

    def _poll_events(self):

        try:

            while True:

                event = self.events.get_nowait()

                if event[0] == "progress":

                    _, done, total, message = event

                    percentage = (
                        100 * done / max(total, 1)
                    )

                    self.progress_var.set(
                        percentage
                    )

                    self.progress_text_var.set(
                        f"{done} / {total} запусків"
                    )

                    self.status_var.set(
                        message
                    )

                elif event[0] == "done":

                    self._finish_experiment(
                        event[1]
                    )

                elif event[0] == "error":

                    self._handle_error(
                        event[1],
                        event[2]
                    )

        except queue.Empty:

            pass

        self.root.after(
            120,
            self._poll_events
        )

    # ========================================================
    # ЗАВЕРШЕННЯ ЕКСПЕРИМЕНТУ
    # ========================================================

    def _finish_experiment(self, result):

        self.is_running = False

        self.run_button.configure(
            state="normal",
            text="▶  ПОЧАТИ ЕКСПЕРИМЕНТ",
            bg=PINK,
            fg="#210817"
        )

        self.cancel_button.configure(
            state="disabled"
        )

        self.current_dir = Path(
            result["output_dir"]
        )

        cancelled = result["cancelled"]

        if cancelled:

            self.status_badge.configure(
                text="● СКАСОВАНО",
                bg=ORANGE,
                fg="#171006"
            )

            self.status_var.set(
                "Експеримент скасовано. Збережені результати залишилися на диску."
            )

        else:

            self.status_badge.configure(
                text="● ЗАВЕРШЕНО",
                bg=GREEN,
                fg="#062010"
            )

            self.status_var.set(
                "Експерименти завершено. Результати збережено."
            )

            self.progress_var.set(100)

            self.progress.configure(
                style="Success.Horizontal.TProgressbar"
            )

        self.progress_text_var.set(
            f"{result['completed_runs']} / {result['planned_runs']} запусків"
        )

        self._refresh_history()

        self._load_experiment(
            self.current_dir
        )

        self.show_page("results")

        if cancelled:

            messagebox.showinfo(
                "Експеримент скасовано",
                "Усі завершені запуски збережені."
            )

    # ========================================================
    # ПОМИЛКИ
    # ========================================================

    def _handle_error(self, error_type, message):

        self.is_running = False

        self.run_button.configure(
            state="normal",
            text="▶  ПОЧАТИ ЕКСПЕРИМЕНТ",
            bg=PINK,
            fg="#210817"
        )

        self.cancel_button.configure(
            state="disabled"
        )

        self.status_badge.configure(
            text="● ПОМИЛКА",
            bg=RED,
            fg="#21040A"
        )

        self.status_var.set(
            f"Помилка {error_type}: {message}"
        )

        messagebox.showerror(
            "Помилка експерименту",
            f"{error_type}: {message}\n\n"
            "Попередні завершені результати залишаються збереженими.",
            parent=self.root
        )

    # ========================================================
    # ІСТОРІЯ
    # ========================================================

    def _refresh_history(self):
        """Оновлює історію та KPI, читаючи кожен runs.csv лише один раз."""

        if not RESULTS_DIR.exists():
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        directories = sorted(
            (
                path for path in RESULTS_DIR.iterdir()
                if path.is_dir() and (path / "runs.csv").exists()
            ),
            key=lambda path: path.name,
            reverse=True,
        )

        self.history_paths.clear()

        if hasattr(self, "history_tree"):
            self.history_tree.delete(*self.history_tree.get_children())

        total = 0
        accuracies = []
        losses = []

        for directory in directories:
            label = directory.name.replace("run_", "")
            if len(label) >= 15:
                label = label[:8] + " " + label[9:15]

            try:
                frame = pd.read_csv(
                    directory / "runs.csv",
                    encoding="utf-8-sig",
                )

                with open(directory / "config.json", encoding="utf-8") as file:
                    config = json.load(file)

                completed_mask = frame["status"] == "completed"
                completed = int(completed_mask.sum())
                datasets = ", ".join(config.get("datasets", []))
                status = (
                    "Завершено"
                    if completed == len(frame) and len(frame) > 0
                    else "Частково"
                )

                total += len(frame)
                valid = frame.loc[completed_mask]

                if not valid.empty:
                    if "test_accuracy" in valid.columns:
                        accuracies.extend(
                            valid["test_accuracy"].dropna().tolist()
                        )
                    if "final_test_loss" in valid.columns:
                        losses.extend(
                            valid["final_test_loss"].dropna().tolist()
                        )

            except (
                OSError,
                ValueError,
                KeyError,
                pd.errors.ParserError,
            ):
                continue

            row_id = str(len(self.history_paths))
            self.history_paths[row_id] = directory

            if hasattr(self, "history_tree"):
                self.history_tree.insert(
                    "",
                    "end",
                    iid=row_id,
                    values=(label, datasets, len(frame), status),
                )

        self.run_lookup = {
            path.name: path for path in directories
        }

        if hasattr(self, "run_combo"):
            self.run_combo.configure(values=list(self.run_lookup.keys()))

        self.kpi_experiments.set(str(len(directories)))
        self.kpi_runs.set(str(total))

        if accuracies:
            self.kpi_accuracy.set(
                f"{sum(accuracies) / len(accuracies) * 100:.1f}%"
            )
        else:
            self.kpi_accuracy.set("—")

        if losses:
            self.kpi_loss.set(f"{min(losses):.4f}")
        else:
            self.kpi_loss.set("—")

    def _open_history_selection(self, _event=None):

        selected = self.history_tree.selection()

        if not selected:
            return

        path = self.history_paths.get(
            selected[0]
        )

        if path:

            self._load_experiment(path)

            self.show_page("results")

    # ========================================================
    # ЗАВАНТАЖЕННЯ РЕЗУЛЬТАТІВ
    # ========================================================

    def _on_run_selected(self, _event=None):

        path = self.run_lookup.get(
            self.run_selector_var.get()
        )

        if path:

            self._load_experiment(path)

    def _load_experiment(self, directory: Path):

        if not directory.exists():
            return

        runs_file = directory / "runs.csv"
        config_file = directory / "config.json"

        if not runs_file.exists():
            return

        try:

            frame = pd.read_csv(
                runs_file,
                encoding="utf-8-sig"
            )

            with open(
                config_file,
                encoding="utf-8"
            ) as file:

                config = json.load(file)

        except (
            OSError,
            ValueError,
            pd.errors.ParserError
        ) as exc:

            messagebox.showerror(
                "Помилка читання результатів",
                str(exc),
                parent=self.root
            )

            return

        self.current_dir = directory
        self.current_runs = frame
        self.current_config = config

        if hasattr(self, "result_tree"):

            self._populate_results(frame)

        datasets = config.get(
            "datasets",
            []
        )

        titles = [
            DATASET_NAMES.get(name, name)
            for name in datasets
        ]

        if hasattr(self, "chart_combo"):

            self.chart_combo.configure(
                values=titles
            )

            if titles:

                self.chart_dataset_var.set(
                    titles[0]
                )

                self._render_charts()

        if hasattr(self, "run_combo"):

            self.run_selector_var.set(
                directory.name
            )

    # ========================================================
    # ЗАПОВНЕННЯ ТАБЛИЦІ
    # ========================================================

    @staticmethod
    def _number(value, digits=4):

        try:

            number = float(value)

            if math.isfinite(number):

                return f"{number:.{digits}f}"

        except (ValueError, TypeError):

            pass

        return "—"

    def _populate_results(self, frame):

        self.result_tree.delete(
            *self.result_tree.get_children()
        )

        for _, row in frame.iterrows():

            optimizer = str(
                row.get("optimizer", "")
            )

            status = str(
                row.get("status", "")
            )

            if status == "completed":

                tag = optimizer
                status_text = "Завершено"

            elif status == "diverged":

                tag = "bad"
                status_text = "Числова помилка"

            else:

                tag = "neutral"
                status_text = "Перервано"

            target_reached = bool(
                row.get("target_reached", False)
            )

            if target_reached:

                epochs = self._number(
                    row.get("epochs_to_target"),
                    0
                )

            elif status == "cancelled":

                epochs = "—"

            else:

                epochs = "Не досягнуто"

            accuracy = self._number(
                row.get("test_accuracy", float("nan")) * 100,
                1
            )

            if accuracy != "—":
                accuracy += "%"

            values = (
                row.get("dataset_title", row.get("dataset", "")),
                optimizer,
                self._number(row.get("learning_rate"), 5),
                self._number(row.get("final_train_loss")),
                self._number(row.get("minimum_train_loss")),
                self._number(row.get("final_test_loss")),
                accuracy,
                self._number(row.get("test_f1")),
                epochs,
                self._number(row.get("elapsed_seconds"), 3),
                status_text
            )

            self.result_tree.insert(
                "",
                "end",
                values=values,
                tags=(tag,)
            )

    def _sort_table(self, column):

        rows = [
            (
                self.result_tree.set(item, column),
                item
            )
            for item in self.result_tree.get_children("")
        ]

        reverse = self.sort_states.get(
            column,
            False
        )

        def sort_key(item):

            value = item[0]

            try:

                number = float(
                    value.replace("%", "")
                )

                return (0, number)

            except ValueError:

                return (1, value.casefold())

        rows.sort(
            key=sort_key,
            reverse=reverse
        )

        for index, (_, item) in enumerate(rows):

            self.result_tree.move(
                item,
                "",
                index
            )

        self.sort_states[column] = not reverse

    # ========================================================
    # ВІДКРИТТЯ ПАПКИ РЕЗУЛЬТАТІВ
    # ========================================================

    def open_results_folder(self):

        target = (
            self.current_dir
            if self.current_dir and self.current_dir.exists()
            else RESULTS_DIR
        )

        target.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            if sys.platform.startswith("win"):

                os.startfile(str(target))

            elif sys.platform == "darwin":

                subprocess.Popen(["open", str(target)])

            else:

                subprocess.Popen(["xdg-open", str(target)])

        except OSError as exc:

            messagebox.showerror(
                "Не вдалося відкрити папку",
                str(exc),
                parent=self.root
            )


# ============================================================
# ЗАПУСК
# ============================================================

def main():

    root = tk.Tk()

    app = OptiLabApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()