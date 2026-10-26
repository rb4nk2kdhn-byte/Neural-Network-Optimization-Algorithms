"""Графічний інтерфейс порівняльного дослідження алгоритмів оптимізації."""
from __future__ import annotations

import math
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.image import imread

from datasets import DATASET_NAMES
from experiments import run_experiments, validate_config
from optimizers import OPTIMIZER_NAMES

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# Чорна основа, білий текст і сірі нейтральні елементи; яскраві кольори
# залишаємо для станів виконання та кольорових серій на графіках.
BG = "#000000"
PANEL = "#0A0A0A"
PANEL_ALT = "#191919"
BORDER = "#424242"
TEXT = "#FFFFFF"
MUTED = "#BDBDBD"
ACCENT = "#666666"
ACCENT_HOVER = "#858585"
SELECTED_BG = "#454545"
HEADING_BG = "#242424"
STATUS_BG = "#303030"
STATUS_READY_BG = "#303030"
STATUS_RUNNING_BG = "#FFD43B"  # яскраво-жовтий — обчислення тривають
STATUS_SUCCESS_BG = "#2EEA8B"  # зелений — завершено успішно
STATUS_CANCEL_BG = "#FFCA57"   # золотистий — скасовано
STATUS_ERROR_BG = "#FF526D"    # рожево-червоний — сталася помилка
RUNNING_BUTTON_BG = "#FFD43B"
RUNNING_BUTTON_FG = "#111111"
START_BUTTON_BG = "#FF4FA3"       # яскраво-рожева кнопка запуску
START_BUTTON_HOVER = "#FF79BC"
START_BUTTON_PRESSED = "#E92F91"
START_BUTTON_FG = "#210817"

# Дуже темні кольорові відтінки допомагають швидко відрізнити рядки оптимізаторів,
# але не відволікають від числових результатів.
OPTIMIZER_ROW_COLORS = {
    "SGD": ("#102235", "#E6F4FF"),
    "Momentum": ("#302712", "#FFF0C2"),
    "RMSProp": ("#102D26", "#DCFFF4"),
    "Adam": ("#32152A", "#FFE0F0"),
}


class OptimizerLabApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Порівняльний аналіз алгоритмів оптимізації для мінімізації функції втрат нейронної мережі на нелінійно роздільних наборах даних")
        self.root.geometry("1390x900")
        self.root.minsize(1080, 740)
        self.root.configure(bg=BG)

        self.event_queue: queue.Queue = queue.Queue()
        self.cancel_event: threading.Event | None = None
        self.worker: threading.Thread | None = None
        self.is_running = False
        self.current_output_dir: Path | None = None
        self._sort_state: dict[str, bool] = {}
        self._plot_canvas: FigureCanvasTkAgg | None = None
        self._plot_figure: Figure | None = None

        self._setup_styles()
        self._build_variables()
        self._build_layout()
        self.root.after(100, self._poll_events)

    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        # Основні поверхні та текст.
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Alt.TFrame", background=PANEL_ALT)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 14))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI Semibold", 10))

        # У всіх станах вкладки залишаються темними; наведення не повинно давати білу пляму.
        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure(
            "TNotebook.Tab", background=PANEL_ALT, foreground=MUTED,
            padding=(15, 9), font=("Segoe UI Semibold", 9), borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", ACCENT), ("active", SELECTED_BG), ("!selected", PANEL_ALT)],
            foreground=[("selected", TEXT), ("active", TEXT), ("!selected", MUTED)],
        )

        # Явно задаємо active/pressed стани заголовків Treeview: саме вони могли ставати білими.
        style.configure(
            "Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
            rowheight=35, borderwidth=0, relief="flat", font=("Segoe UI", 9),
        )
        style.map(
            "Treeview",
            background=[("selected", SELECTED_BG), ("active", PANEL)],
            foreground=[("selected", TEXT), ("active", TEXT)],
        )
        style.configure(
            "Treeview.Heading", background=HEADING_BG, foreground=TEXT,
            relief="flat", borderwidth=0, padding=(8, 9),
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "Treeview.Heading",
            background=[("pressed", SELECTED_BG), ("active", SELECTED_BG), ("disabled", HEADING_BG)],
            foreground=[("pressed", TEXT), ("active", TEXT), ("disabled", MUTED)],
            relief=[("pressed", "flat"), ("active", "flat")],
        )

        # Поля вибору та смуги прокручування теж використовують лише сірі стани.
        style.configure(
            "TCombobox", fieldbackground=PANEL_ALT, background=PANEL_ALT,
            foreground=TEXT, arrowcolor=TEXT, padding=5, bordercolor=BORDER,
            lightcolor=BORDER, darkcolor=BORDER, insertcolor=TEXT,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PANEL_ALT), ("disabled", PANEL)],
            background=[("active", SELECTED_BG), ("readonly", PANEL_ALT), ("disabled", PANEL)],
            foreground=[("readonly", TEXT), ("disabled", MUTED)],
            arrowcolor=[("active", TEXT), ("!disabled", TEXT)],
        )
        for scrollbar_style in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
            style.configure(
                scrollbar_style, background=ACCENT, troughcolor=PANEL,
                bordercolor=BG, arrowcolor=TEXT, relief="flat",
            )
            style.map(
                scrollbar_style,
                background=[("pressed", ACCENT_HOVER), ("active", ACCENT_HOVER)],
            )
        style.configure(
            "Horizontal.TProgressbar", troughcolor=PANEL_ALT, background=ACCENT,
            bordercolor=PANEL_ALT, lightcolor=ACCENT, darkcolor=ACCENT,
        )
        style.configure(
            "Running.Horizontal.TProgressbar", troughcolor=PANEL_ALT, background=STATUS_RUNNING_BG,
            bordercolor=PANEL_ALT, lightcolor=STATUS_RUNNING_BG, darkcolor=STATUS_RUNNING_BG,
        )
        style.configure(
            "Success.Horizontal.TProgressbar", troughcolor=PANEL_ALT, background=STATUS_SUCCESS_BG,
            bordercolor=PANEL_ALT, lightcolor=STATUS_SUCCESS_BG, darkcolor=STATUS_SUCCESS_BG,
        )
        style.configure(
            "Cancelled.Horizontal.TProgressbar", troughcolor=PANEL_ALT, background=STATUS_CANCEL_BG,
            bordercolor=PANEL_ALT, lightcolor=STATUS_CANCEL_BG, darkcolor=STATUS_CANCEL_BG,
        )
        style.configure(
            "Error.Horizontal.TProgressbar", troughcolor=PANEL_ALT, background=STATUS_ERROR_BG,
            bordercolor=PANEL_ALT, lightcolor=STATUS_ERROR_BG, darkcolor=STATUS_ERROR_BG,
        )

    def _style_combobox_dropdown(self, combo: ttk.Combobox) -> None:
        """Оформлює внутрішній список ttk.Combobox, який інакше може бути білим."""
        try:
            # ttk стилізує поле Combobox, але випадаючий Listbox має окремі
            # класичні параметри Tk. Налаштовуємо саме цей внутрішній список.
            popdown = str(self.root.tk.call("ttk::combobox::PopdownWindow", str(combo)))
            listbox = f"{popdown}.f.l"
            self.root.tk.call(
                listbox, "configure",
                "-background", PANEL_ALT,
                "-foreground", TEXT,
                "-selectbackground", START_BUTTON_BG,
                "-selectforeground", START_BUTTON_FG,
                "-activestyle", "none",
                "-selectborderwidth", 0,
                "-highlightbackground", BORDER,
                "-highlightcolor", ACCENT_HOVER,
                "-borderwidth", 0,
                "-relief", "flat",
                "-font", ("Segoe UI", 10),
            )
        except tk.TclError:
            # Деякі Tcl/Tk збірки можуть мати іншу внутрішню реалізацію popdown.
            # Інтерфейс все одно запускається, навіть якщо цей штрих недоступний.
            return

    def _build_variables(self) -> None:
        self.dataset_var = tk.StringVar(value=DATASET_NAMES["moons"])
        self.all_datasets_var = tk.BooleanVar(value=False)
        self.optimizer_vars = {name: tk.BooleanVar(value=True) for name in OPTIMIZER_NAMES}
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
        self.status_var = tk.StringVar(value="Готово до запуску експерименту")
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_label_var = tk.StringVar(value="0 / 0 запусків")
        self.total_runs_var = tk.StringVar(value="—")
        self.avg_accuracy_var = tk.StringVar(value="—")
        self.best_loss_var = tk.StringVar(value="—")
        self.plot_choice_var = tk.StringVar(value="Графіки з’являться після запуску")

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(22, 17, 22, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        title_box = ttk.Frame(header)
        title_box.grid(row=0, column=0, sticky="ew")
        title_box.columnconfigure(0, weight=1)
        project_title = (
            "Порівняльний аналіз алгоритмів оптимізації для мінімізації функції "
            "втрат нейронної мережі на нелінійно роздільних наборах даних"
        )
        ttk.Label(
            title_box, text=project_title, style="Title.TLabel",
            wraplength=980, justify="left",
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(
            header, text="Власна реалізація backpropagation · 4 алгоритми · локальні експерименти",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.status_pill = tk.Label(
            header, text="● ГОТОВО", bg=STATUS_READY_BG, fg=TEXT,
            font=("Segoe UI Semibold", 9), padx=12, pady=7,
        )
        self.status_pill.grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0))

        main = ttk.Frame(self.root, padding=(20, 0, 20, 12))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=0, minsize=295)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.sidebar_container = ttk.Frame(main, style="Panel.TFrame")
        self.sidebar_container.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        self.sidebar_container.columnconfigure(0, weight=1)
        self.sidebar_container.rowconfigure(0, weight=1)
        self.sidebar_canvas = tk.Canvas(self.sidebar_container, bg=PANEL, highlightthickness=0, width=295)
        self.sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        sidebar_scroll = ttk.Scrollbar(self.sidebar_container, orient="vertical", command=self.sidebar_canvas.yview)
        sidebar_scroll.grid(row=0, column=1, sticky="ns")
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        self.sidebar = ttk.Frame(self.sidebar_canvas, style="Panel.TFrame", padding=17)
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=self.sidebar, anchor="nw")
        self.sidebar.columnconfigure(0, weight=1)
        self.sidebar.bind("<Configure>", lambda _event: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind("<Configure>", lambda event: self.sidebar_canvas.itemconfigure(self.sidebar_window, width=event.width))
        self.sidebar_canvas.bind_all("<MouseWheel>", self._scroll_sidebar, add="+")
        self._build_sidebar()

        self.content = ttk.Frame(main)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(1, weight=1)
        self._build_dashboard()

        footer = ttk.Frame(self.root, padding=(22, 0, 22, 15))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.progress = ttk.Progressbar(
            footer, variable=self.progress_var, maximum=100, mode="determinate",
            style="Horizontal.TProgressbar",
        )
        self.progress.grid(row=1, column=0, sticky="ew")
        ttk.Label(footer, textvariable=self.progress_label_var, style="Muted.TLabel").grid(row=1, column=1, sticky="e", padx=(14, 0))

    def _add_section(self, parent: ttk.Frame, text: str, row: int) -> None:
        ttk.Label(parent, text=text.upper(), style="Section.TLabel").grid(row=row, column=0, sticky="w", pady=(12, 7))

    def _field(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int, width: int = 12) -> None:
        line = ttk.Frame(parent, style="Panel.TFrame")
        line.grid(row=row, column=0, sticky="ew", pady=3)
        line.columnconfigure(0, weight=1)
        ttk.Label(line, text=label, style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        entry = tk.Entry(line, textvariable=variable, width=width, bg=PANEL_ALT, fg=TEXT,
                         insertbackground=TEXT, selectbackground=SELECTED_BG, selectforeground=TEXT,
                         relief="flat", highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT_HOVER, font=("Consolas", 10))
        entry.grid(row=0, column=1, sticky="e", ipady=4, padx=(7, 0))

    def _build_sidebar(self) -> None:
        self._add_section(self.sidebar, "Дані", 0)
        dataset_box = ttk.Combobox(
            self.sidebar, textvariable=self.dataset_var, state="readonly",
            values=[DATASET_NAMES[name] for name in DATASET_NAMES],
        )
        self._style_combobox_dropdown(dataset_box)
        dataset_box.grid(row=1, column=0, sticky="ew", pady=(0, 4), ipady=3)
        tk.Label(self.sidebar, text="moons — півмісяці · circles — кільця · xor — XOR",
                 bg=PANEL, fg=MUTED, justify="left", wraplength=260, font=("Segoe UI", 8)).grid(row=2, column=0, sticky="w")
        tk.Checkbutton(self.sidebar, text="Порівняти всі три геометрії", variable=self.all_datasets_var,
                       bg=PANEL, fg=TEXT, activebackground=SELECTED_BG, activeforeground=TEXT,
                       selectcolor=PANEL_ALT, font=("Segoe UI", 9)).grid(row=3, column=0, sticky="w", pady=(5, 0))

        self._add_section(self.sidebar, "Оптимізатори", 4)
        opt_frame = ttk.Frame(self.sidebar, style="Panel.TFrame")
        opt_frame.grid(row=5, column=0, sticky="ew")
        for i, name in enumerate(OPTIMIZER_NAMES):
            tk.Checkbutton(opt_frame, text=name, variable=self.optimizer_vars[name], bg=PANEL, fg=TEXT,
                           activebackground=SELECTED_BG, activeforeground=TEXT, selectcolor=PANEL_ALT,
                           font=("Segoe UI", 9)).grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 10), pady=2)

        self._add_section(self.sidebar, "Параметри навчання", 6)
        fields = [
            ("Learning rate", self.learning_rate_var),
            ("Максимум епох", self.epochs_var),
            ("Розмір мініпакета", self.batch_var),
            ("Повторів", self.repeats_var),
            ("Цільова BCE", self.target_loss_var),
            ("Терпіння (епохи)", self.patience_var),
            ("Мінімальне покращення", self.min_delta_var),
            ("Кількість зразків", self.samples_var),
            ("Seed", self.seed_var),
        ]
        for i, (label, var) in enumerate(fields, start=7):
            self._field(self.sidebar, label, var, i)
        tk.Checkbutton(self.sidebar, text="Додати sweep швидкості навчання (×0.1, ×1, ×10)",
                       variable=self.sweep_var, bg=PANEL, fg=TEXT, activebackground=SELECTED_BG,
                       activeforeground=TEXT, selectcolor=PANEL_ALT, wraplength=260, justify="left",
                       font=("Segoe UI", 9)).grid(row=16, column=0, sticky="w", pady=(8, 3))

        buttons = ttk.Frame(self.sidebar, style="Panel.TFrame")
        buttons.grid(row=17, column=0, sticky="ew", pady=(12, 0))
        buttons.columnconfigure(0, weight=1)
        self.run_button = tk.Button(
            buttons, text="▶  Запустити експерименти", command=self._start_run,
            bg=START_BUTTON_BG, fg=START_BUTTON_FG,
            activebackground=START_BUTTON_PRESSED, activeforeground=START_BUTTON_FG,
            disabledforeground=RUNNING_BUTTON_FG,
            relief="flat", borderwidth=0, highlightthickness=1,
            highlightbackground=START_BUTTON_BG, highlightcolor=START_BUTTON_HOVER,
            cursor="hand2", font=("Segoe UI Semibold", 10), pady=11,
        )
        self.run_button.grid(row=0, column=0, sticky="ew")
        self.cancel_button = tk.Button(buttons, text="Скасувати запуск", command=self._cancel_run,
                                       bg=PANEL_ALT, fg=MUTED, activebackground=BORDER,
                                       activeforeground=TEXT, relief="flat", state="disabled",
                                       font=("Segoe UI", 9), pady=7)
        self.cancel_button.grid(row=1, column=0, sticky="ew", pady=(7, 0))
        self.open_button = tk.Button(buttons, text="Відкрити теку результатів", command=self._open_results,
                                     bg=PANEL_ALT, fg=TEXT, activebackground=BORDER,
                                     activeforeground=TEXT, relief="flat", cursor="hand2",
                                     font=("Segoe UI", 9), pady=7)
        self.open_button.grid(row=2, column=0, sticky="ew", pady=(7, 0))
        tk.Label(self.sidebar, text="Усі розрахунки локальні. Інтернет для запуску не потрібен після встановлення пакетів.",
                 bg=PANEL, fg=MUTED, wraplength=255, justify="left", font=("Segoe UI", 8)).grid(row=18, column=0, sticky="w", pady=(13, 0))

    def _build_dashboard(self) -> None:
        cards = ttk.Frame(self.content)
        cards.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        for col in range(3):
            cards.columnconfigure(col, weight=1)
        self._metric_card(cards, 0, "ЗАВЕРШЕНІ ЗАПУСКИ", self.total_runs_var, "фактичні результати")
        self._metric_card(cards, 1, "СЕРЕДНЯ ТЕСТОВА ТОЧНІСТЬ", self.avg_accuracy_var, "успішні завершені запуски")
        self._metric_card(cards, 2, "НАЙМЕНША ТЕСТОВА BCE", self.best_loss_var, "серед записаних запусків")

        self.notebook = ttk.Notebook(self.content)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        results_tab = ttk.Frame(self.notebook, style="Panel.TFrame", padding=12)
        plots_tab = ttk.Frame(self.notebook, style="Panel.TFrame", padding=12)
        help_tab = ttk.Frame(self.notebook, style="Panel.TFrame", padding=16)
        self.notebook.add(results_tab, text="Результати")
        self.notebook.add(plots_tab, text="Графіки")
        self.notebook.add(help_tab, text="Довідка")
        self._build_results_tab(results_tab)
        self._build_plots_tab(plots_tab)
        self._build_help_tab(help_tab)

    def _metric_card(self, parent: ttk.Frame, col: int, title: str, variable: tk.StringVar, subtitle: str) -> None:
        card = ttk.Frame(parent, style="Panel.TFrame", padding=(15, 12))
        card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 7, 7 if col < 2 else 0))
        tk.Label(card, text=title, bg=PANEL, fg=MUTED, font=("Segoe UI Semibold", 8)).pack(anchor="w")
        tk.Label(card, textvariable=variable, bg=PANEL, fg=TEXT, font=("Segoe UI Semibold", 21)).pack(anchor="w", pady=(5, 2))
        tk.Label(card, text=subtitle, bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")

    def _build_results_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        heading = ttk.Frame(parent, style="Panel.TFrame")
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        heading.columnconfigure(0, weight=1)
        ttk.Label(
            heading, text="Експериментальні запуски", style="Section.TLabel"
        ).grid(row=0, column=0, sticky="w")
        legend = ttk.Frame(heading, style="Panel.TFrame")
        legend.grid(row=0, column=1, sticky="e")
        for optimizer_name, color in (
            ("SGD", "#56C8FF"),
            ("Momentum", "#FFC857"),
            ("RMSProp", "#50E3C2"),
            ("Adam", "#FF5DB1"),
        ):
            tk.Label(
                legend, text=f"● {optimizer_name}", bg=PANEL, fg=color,
                font=("Segoe UI Semibold", 8), padx=4,
            ).pack(side="left")
        ttk.Label(
            heading, text="Натисни на заголовок стовпця, щоб відсортувати результати",
            style="PanelMuted.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="e", pady=(5, 0))
        table_frame = ttk.Frame(parent, style="Panel.TFrame")
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("dataset", "optimizer", "lr", "train_loss", "min_loss", "test_loss", "accuracy", "epochs", "time", "status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        definitions = {
            "dataset": ("Набір даних", 110), "optimizer": ("Оптимізатор", 95), "lr": ("LR", 65),
            "train_loss": ("Train BCE", 82), "min_loss": ("Min BCE", 82),
            "test_loss": ("Test BCE", 82), "accuracy": ("Точність", 82),
            "epochs": ("Епох до цілі", 92), "time": ("Час, с", 65), "status": ("Статус", 88),
        }
        for col in columns:
            label, width = definitions[col]
            self.tree.heading(col, text=label, command=lambda c=col: self._sort_table(c))
            self.tree.column(col, width=width, minwidth=65, anchor="center", stretch=(col == "dataset"))
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        # Колір фону рядка показує оптимізатор; легенда над таблицею пояснює палітру.
        for optimizer_name, (row_bg, row_fg) in OPTIMIZER_ROW_COLORS.items():
            self.tree.tag_configure(f"optimizer_{optimizer_name}", background=row_bg, foreground=row_fg)
        self.tree.tag_configure("bad", background="#3B111B", foreground="#FFD5DC")
        self.tree.tag_configure("neutral", background="#282828", foreground="#E3E3E3")
        self.empty_table_label = ttk.Label(parent, text="Після запуску тут з’являться фактичні результати. Вигадані дані не показуються.", style="PanelMuted.TLabel")
        self.empty_table_label.grid(row=2, column=0, sticky="w", pady=(10, 0))

    def _build_plots_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        bar = ttk.Frame(parent, style="Panel.TFrame")
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(bar, text="Візуалізація", style="Panel.TLabel").pack(side="left", padx=(0, 10))
        self.plot_combo = ttk.Combobox(bar, textvariable=self.plot_choice_var, state="readonly", width=44, values=[])
        self._style_combobox_dropdown(self.plot_combo)
        self.plot_combo.pack(side="left", fill="x", expand=True)
        self.plot_combo.bind("<<ComboboxSelected>>", self._show_selected_plot)
        self.plot_holder = ttk.Frame(parent, style="Panel.TFrame")
        self.plot_holder.grid(row=1, column=0, sticky="nsew")
        self.plot_holder.columnconfigure(0, weight=1)
        self.plot_holder.rowconfigure(0, weight=1)
        self.plot_placeholder = tk.Label(self.plot_holder, text="Після експерименту графіки будуть доступні тут і як PNG у папці результатів.",
                                         bg=PANEL, fg=MUTED, font=("Segoe UI", 11), wraplength=600)
        self.plot_placeholder.grid(row=0, column=0, sticky="nsew")

    def _build_help_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        help_text = (
            "ЯК ЧИТАТИ РЕЗУЛЬТАТИ\n\n"
            "Функція втрат (BCE) вимірює, наскільки передбачені ймовірності відрізняються від правильних міток. Менше значення краще, але це не те саме, що точність.\n\n"
            "Точність — частка правильних передбачень. Precision показує, яка частина позитивних передбачень правильна; Recall — яку частину справжнього класу 1 знайдено; F1 — гармонійне середнє Precision і Recall.\n\n"
            "SGD оновлює параметри безпосередньо за градієнтом. Momentum накопичує напрям попередніх кроків. RMSProp масштабує крок за середнім квадратом градієнта. Adam поєднує оцінки першого та другого моментів і коригує їх початкове зміщення.\n\n"
            "Критерій зупинки: або навчальна BCE досягла заданого порога, або покращення було меншим за min_delta протягом визначеної кількості епох. Важливо: зупинка через плато не означає, що цільова BCE досягнута. У таблиці ці випадки розрізняються.\n\n"
            "Експеримент є повторюваним за фіксованих seed та параметрів. У межах повтору оптимізатори отримують однакові початкові ваги та порядок мініпакетів. Тестова вибірка не бере участі в оновленні параметрів.\n\n"
            "Час залежить від процесора, версії Python, бібліотек і навантаження комп’ютера. Стандартне відхилення показує розкид між повторами; за одного повтору воно математично не визначене.\n\n"
            "Усі обчислення виконуються локально. Автоматично зберігаються CSV-таблиці, JSON з деталями, текстовий звіт і PNG-графіки. Докладні математичні виведення дивіться у файлі MATH_EXPLANATION.md."
        )
        text_box = tk.Text(parent, wrap="word", bg=PANEL, fg=TEXT, insertbackground=TEXT,
                           selectbackground=SELECTED_BG, selectforeground=TEXT, relief="flat",
                           padx=18, pady=18, font=("Segoe UI", 10), spacing1=2, spacing3=8)
        text_box.insert("1.0", help_text)
        text_box.configure(state="disabled")
        text_box.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=text_box.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text_box.configure(yscrollcommand=scrollbar.set)

    def _scroll_sidebar(self, event) -> None:
        # Прокручуємо налаштування лише тоді, коли курсор перебуває над лівою панеллю.
        if not self.sidebar_canvas.winfo_exists():
            return
        left = self.sidebar_container.winfo_rootx()
        top = self.sidebar_container.winfo_rooty()
        right = left + self.sidebar_container.winfo_width()
        bottom = top + self.sidebar_container.winfo_height()
        if left <= event.x_root <= right and top <= event.y_root <= bottom:
            self.sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _read_config(self) -> dict[str, Any]:
        display_to_key = {title: key for key, title in DATASET_NAMES.items()}
        datasets = list(DATASET_NAMES) if self.all_datasets_var.get() else [display_to_key[self.dataset_var.get()]]
        selected_optimizers = [name for name, var in self.optimizer_vars.items() if var.get()]
        try:
            lr = float(self.learning_rate_var.get().strip())
            target = float(self.target_loss_var.get().strip())
            min_delta = float(self.min_delta_var.get().strip())
            epochs = int(self.epochs_var.get().strip())
            batch = int(self.batch_var.get().strip())
            repeats = int(self.repeats_var.get().strip())
            patience = int(self.patience_var.get().strip())
            samples = int(self.samples_var.get().strip())
            seed = int(self.seed_var.get().strip())
        except ValueError as exc:
            raise ValueError("Перевірте числові поля: усі значення мають бути числами правильного формату.") from exc
        config = {
            "datasets": datasets,
            "optimizers": selected_optimizers,
            "learning_rate": lr,
            "learning_rate_sweep": self.sweep_var.get(),
            "max_epochs": epochs,
            "batch_size": batch,
            "repeats": repeats,
            "target_loss": target,
            "patience": patience,
            "min_delta": min_delta,
            "seed": seed,
            "n_samples": samples,
        }
        return validate_config(config)

    def _start_run(self) -> None:
        if self.is_running:
            return
        try:
            config = self._read_config()
        except ValueError as exc:
            messagebox.showerror("Некоректні параметри", str(exc), parent=self.root)
            return
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.cancel_event = threading.Event()
        self.is_running = True
        self.run_button.configure(
            state="disabled", text="Обчислення тривають…",
            bg=RUNNING_BUTTON_BG, fg=RUNNING_BUTTON_FG,
            disabledforeground=RUNNING_BUTTON_FG,
            activebackground=RUNNING_BUTTON_BG, activeforeground=RUNNING_BUTTON_FG,
            cursor="watch",
        )
        self.cancel_button.configure(state="normal")
        self.progress.configure(style="Running.Horizontal.TProgressbar")
        self.status_var.set("Ініціалізація даних та моделі…")
        self.status_pill.configure(text="● ВИКОНУЄТЬСЯ", bg=STATUS_RUNNING_BG, fg="#111111")
        self.progress_var.set(0)
        self.progress_label_var.set("0 / підготовка")
        self.total_runs_var.set("…")
        self.avg_accuracy_var.set("—")
        self.best_loss_var.set("—")
        self.current_output_dir = None

        def worker_func() -> None:
            try:
                result = run_experiments(
                    config,
                    output_root=RESULTS_DIR,
                    progress_callback=lambda done, total, msg: self.event_queue.put(("progress", done, total, msg)),
                    cancel_event=self.cancel_event,
                )
                self.event_queue.put(("done", result))
            except Exception as exc:
                # Повідомляємо про конкретну помилку; попередні запуски вже збережені на диск.
                self.event_queue.put(("error", type(exc).__name__, str(exc)))

        self.worker = threading.Thread(target=worker_func, name="OptiLabExperimentWorker", daemon=True)
        self.worker.start()

    def _cancel_run(self) -> None:
        if self.cancel_event is not None and self.is_running:
            self.cancel_event.set()
            self.status_var.set("Запит на скасування надіслано; поточний крок безпечно завершується…")
            self.cancel_button.configure(state="disabled")

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.event_queue.get_nowait()
                if event[0] == "progress":
                    _, done, total, message = event
                    self.progress_var.set(100 * done / max(total, 1))
                    self.progress_label_var.set(f"{done} / {total} запусків")
                    self.status_var.set(message)
                elif event[0] == "done":
                    self._finish_run(event[1])
                elif event[0] == "error":
                    self._handle_error(event[1], event[2])
        except queue.Empty:
            pass
        self.root.after(120, self._poll_events)

    def _finish_run(self, result: dict[str, Any]) -> None:
        self.is_running = False
        self.run_button.configure(
            state="normal", text="▶  Запустити експерименти", bg=START_BUTTON_BG, fg=START_BUTTON_FG,
            activebackground=START_BUTTON_PRESSED, activeforeground=START_BUTTON_FG,
            highlightbackground=START_BUTTON_BG, highlightcolor=START_BUTTON_HOVER, cursor="hand2",
        )
        self.cancel_button.configure(state="disabled")
        self.current_output_dir = Path(result["output_dir"])
        self.progress_var.set(100 if not result["cancelled"] else self.progress_var.get())
        self.progress_label_var.set(f"{result['completed_runs']} / {result['planned_runs']} запусків")
        status = "виконання скасовано" if result["cancelled"] else "експерименти завершено"
        self.status_var.set(f"{status.capitalize()}. Результати: {self.current_output_dir}")
        if result["cancelled"]:
            self.status_pill.configure(text="● СКАСОВАНО", bg=STATUS_CANCEL_BG, fg="#171006")
            self.progress.configure(style="Cancelled.Horizontal.TProgressbar")
        else:
            self.status_pill.configure(text="● ЗАВЕРШЕНО", bg=STATUS_SUCCESS_BG, fg="#062010")
            self.progress.configure(style="Success.Horizontal.TProgressbar")
        runs = result["runs"]
        completed = [r for r in runs if r["status"] == "completed"]
        self.total_runs_var.set(f"{len(completed)} / {result['planned_runs']}")
        if completed:
            self.avg_accuracy_var.set(f"{np.mean([r['test_accuracy'] for r in completed]) * 100:.1f}%")
            self.best_loss_var.set(f"{min(r['final_test_loss'] for r in completed):.4f}")
        else:
            self.avg_accuracy_var.set("—")
            self.best_loss_var.set("—")
        self._populate_table(runs)
        self._load_plot_choices(self.current_output_dir / "plots")
        if result["cancelled"]:
            messagebox.showinfo("Експеримент скасовано", "Завершені запуски та проміжні результати збережено.", parent=self.root)

    def _handle_error(self, error_type: str, message: str) -> None:
        self.is_running = False
        self.run_button.configure(
            state="normal", text="▶  Запустити експерименти", bg=START_BUTTON_BG, fg=START_BUTTON_FG,
            activebackground=START_BUTTON_PRESSED, activeforeground=START_BUTTON_FG,
            highlightbackground=START_BUTTON_BG, highlightcolor=START_BUTTON_HOVER, cursor="hand2",
        )
        self.cancel_button.configure(state="disabled")
        self.progress.configure(style="Error.Horizontal.TProgressbar")
        self.status_pill.configure(text="● ПОМИЛКА", bg=STATUS_ERROR_BG, fg="#21040A")
        self.status_var.set(f"Помилка {error_type}: {message}")
        messagebox.showerror(
            "Не вдалося завершити експеримент",
            f"{error_type}: {message}\n\nЯкщо попередні запуски вже завершилися, вони збережені в папці results.",
            parent=self.root,
        )

    @staticmethod
    def _display_num(value: Any, digits: int = 4) -> str:
        try:
            number = float(value)
            return f"{number:.{digits}f}" if math.isfinite(number) else "—"
        except (TypeError, ValueError):
            return "—"

    def _populate_table(self, runs: list[dict[str, Any]]) -> None:
        self.tree.delete(*self.tree.get_children())
        for row in runs:
            optimizer_name = str(row.get("optimizer", ""))
            if row["status"] == "completed":
                status = "● Успішно"
                # Прив'язуємо затемнений акцентний колір до алгоритму.
                tag = f"optimizer_{optimizer_name}" if optimizer_name in OPTIMIZER_ROW_COLORS else "neutral"
            elif row["status"] == "diverged":
                status = "● Числова помилка"
                tag = "bad"
            else:
                status = "● Перервано"
                tag = "neutral"
            epoch_display = self._display_num(row.get("epochs_to_target"), 0)
            if not row.get("target_reached", False):
                reason = row.get("stopping_reason", "")
                epoch_display = "не досягнуто" if reason != "cancelled" else "—"
            vals = (
                row.get("dataset_title", row.get("dataset", "")), row.get("optimizer", ""),
                self._display_num(row.get("learning_rate"), 5), self._display_num(row.get("final_train_loss"), 4),
                self._display_num(row.get("minimum_train_loss"), 4), self._display_num(row.get("final_test_loss"), 4),
                (self._display_num(float(row.get("test_accuracy")) * 100, 1) + "%") if math.isfinite(float(row.get("test_accuracy", float("nan")))) else "—",
                epoch_display, self._display_num(row.get("elapsed_seconds"), 3), status,
            )
            self.tree.insert("", "end", values=vals, tags=(tag,))
        self.empty_table_label.configure(text=f"Записів у таблиці: {len(runs)} · пропущені/невдалі запуски позначаються окремо.")

    def _sort_table(self, column: str) -> None:
        rows = [(self.tree.set(item, column), item) for item in self.tree.get_children("")]
        reverse = self._sort_state.get(column, False)
        def sort_key(entry):
            value = entry[0]
            try:
                return (0, float(value.rstrip("%")))
            except (ValueError, AttributeError):
                return (1, value.casefold())
        rows.sort(key=sort_key, reverse=reverse)
        for index, (_, item) in enumerate(rows):
            self.tree.move(item, "", index)
        self._sort_state[column] = not reverse

    def _load_plot_choices(self, plots_dir: Path) -> None:
        files = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []
        self._plot_files = {path.stem: path for path in files}
        values = list(self._plot_files)
        self.plot_combo.configure(values=values)
        if values:
            preferred = next((item for item in values if item.startswith("loss_")), values[0])
            self.plot_choice_var.set(preferred)
            self._show_selected_plot()
            self.notebook.select(1)
        else:
            self.plot_choice_var.set("Для цих запусків графіки ще не створено")

    def _show_selected_plot(self, _event=None) -> None:
        choice = self.plot_choice_var.get()
        path = getattr(self, "_plot_files", {}).get(choice)
        if path is None or not path.exists():
            return
        image = imread(path)
        if self._plot_canvas is not None:
            self._plot_canvas.get_tk_widget().destroy()
        if self._plot_figure is not None:
            self._plot_figure.clear()
        self._plot_figure = Figure(figsize=(8.2, 5.4), dpi=100, facecolor=BG)
        ax = self._plot_figure.add_subplot(111)
        ax.imshow(image)
        ax.axis("off")
        self._plot_figure.tight_layout(pad=0.2)
        self._plot_canvas = FigureCanvasTkAgg(self._plot_figure, master=self.plot_holder)
        self._plot_canvas.draw()
        self._plot_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.plot_placeholder.grid_forget()

    def _open_results(self) -> None:
        target = self.current_output_dir if self.current_output_dir and self.current_output_dir.exists() else RESULTS_DIR
        target.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except OSError as exc:
            messagebox.showerror("Не вдалося відкрити теку", f"Відкрийте її вручну:\n{target}\n\n{exc}", parent=self.root)


def main() -> None:
    root = tk.Tk()
    OptimizerLabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
