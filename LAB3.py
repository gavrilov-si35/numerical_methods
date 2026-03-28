from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from interp import (
    DEFAULT_FUNCTION,
    DEFAULT_NODES_TEXT,
    DEFAULT_POINTS_TEXT,
    EqualNodeInterpolationSolver,
)


class Lab3App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Практическая работа №3")
        self.geometry("1180x780")
        self.minsize(980, 680)
        self.current_solver: EqualNodeInterpolationSolver | None = None
        self.status_var = tk.StringVar(value="Заполните данные и нажмите «Рассчитать».")
        self._configure_style()
        self._build_ui()
        self._restore_defaults(run_calculation=True)

    def _configure_style(self) -> None:
        self.option_add("*Font", ("Segoe UI", 10))
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Subtitle.TLabel", foreground="#475569")
        style.configure("Header.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", padding=(14, 8), font=("Segoe UI", 10, "bold"))

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="Интерполирование функции в случае равноотстоящих узлов",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Вариант по умолчанию: f(x)=2*x^2+x^5, узлы 0,1,2,3,4, точки 4.4 и 1.8",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        input_frame = ttk.LabelFrame(container, text="Исходные данные", padding=12)
        input_frame.grid(row=1, column=0, sticky="ew", pady=(12, 10))
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Функция f(x):", style="Header.TLabel").grid(
            row=0, column=0, sticky="w", pady=4
        )
        self.function_entry = ttk.Entry(input_frame)
        self.function_entry.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(input_frame, text="Узлы:", style="Header.TLabel").grid(
            row=1, column=0, sticky="w", pady=4
        )
        self.nodes_entry = ttk.Entry(input_frame)
        self.nodes_entry.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(input_frame, text="Точки x:", style="Header.TLabel").grid(
            row=2, column=0, sticky="w", pady=4
        )
        self.points_entry = ttk.Entry(input_frame)
        self.points_entry.grid(row=2, column=1, sticky="ew", pady=4)

        controls = ttk.Frame(input_frame)
        controls.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        controls.columnconfigure(4, weight=1)

        ttk.Label(controls, text="Знаков после запятой:").grid(row=0, column=0, sticky="w")
        self.precision_spin = ttk.Spinbox(controls, from_=0, to=12, width=6)
        self.precision_spin.set("6")
        self.precision_spin.grid(row=0, column=1, sticky="w", padx=(8, 14))

        ttk.Button(
            controls,
            text="Рассчитать",
            style="Accent.TButton",
            command=self._calculate,
        ).grid(row=0, column=2, sticky="w")

        ttk.Button(
            controls,
            text="Справка",
            command=self._show_help,
        ).grid(row=0, column=3, sticky="w", padx=(10, 0))

        ttk.Button(
            controls,
            text="Очистить вывод",
            command=self._clear_outputs,
        ).grid(row=0, column=4, sticky="e")

        notebook = ttk.Notebook(container)
        notebook.grid(row=2, column=0, sticky="nsew")

        result_tab = ttk.Frame(notebook, padding=10)
        result_tab.columnconfigure(0, weight=1)
        result_tab.rowconfigure(1, weight=1)
        notebook.add(result_tab, text="Результаты")

        summary_frame = ttk.LabelFrame(result_tab, text="Сводная таблица")
        summary_frame.grid(row=0, column=0, sticky="ew")
        summary_frame.columnconfigure(0, weight=1)

        summary_columns = ("x", "formula", "approx", "exact", "error")
        self.summary_tree = ttk.Treeview(
            summary_frame,
            columns=summary_columns,
            show="headings",
            height=4,
        )
        self.summary_tree.grid(row=0, column=0, sticky="nsew")
        summary_frame.rowconfigure(0, weight=1)
        summary_headings = {
            "x": ("x", 120),
            "formula": ("Формула", 220),
            "approx": ("Приближённо", 160),
            "exact": ("Точно", 160),
            "error": ("|Ошибка|", 140),
        }
        for column, (title, width) in summary_headings.items():
            self.summary_tree.heading(column, text=title)
            self.summary_tree.column(column, width=width, anchor="center")

        report_frame = ttk.LabelFrame(result_tab, text="Подробный отчёт")
        report_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        report_frame.columnconfigure(0, weight=1)
        report_frame.rowconfigure(0, weight=1)
        self.report_text = scrolledtext.ScrolledText(
            report_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
        )
        self.report_text.grid(row=0, column=0, sticky="nsew")
        self.report_text.configure(state="disabled")

        table_tab = ttk.Frame(notebook, padding=10)
        table_tab.columnconfigure(0, weight=1)
        table_tab.rowconfigure(0, weight=1)
        notebook.add(table_tab, text="Конечные разности")

        table_frame = ttk.Frame(table_tab)
        table_frame.grid(row=0, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self.difference_tree = ttk.Treeview(table_frame, show="headings")
        self.difference_tree.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self.difference_tree.yview
        )
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(
            table_frame, orient=tk.HORIZONTAL, command=self.difference_tree.xview
        )
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.difference_tree.configure(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set,
        )

        status_bar = ttk.Label(
            container,
            textvariable=self.status_var,
            anchor="w",
            padding=(4, 8, 4, 0),
            style="Subtitle.TLabel",
        )
        status_bar.grid(row=3, column=0, sticky="ew")

        self.bind("<Return>", lambda _event: self._calculate())

    def _restore_defaults(self, run_calculation: bool = False) -> None:
        self._set_entry_text(self.function_entry, DEFAULT_FUNCTION)
        self._set_entry_text(self.nodes_entry, DEFAULT_NODES_TEXT)
        self._set_entry_text(self.points_entry, DEFAULT_POINTS_TEXT)
        self.precision_spin.set("6")
        if run_calculation:
            self._calculate()

    @staticmethod
    def _set_entry_text(entry: ttk.Entry, value: str) -> None:
        entry.delete(0, tk.END)
        entry.insert(0, value)

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _clear_outputs(self) -> None:
        self._clear_tree(self.summary_tree)
        self._clear_tree(self.difference_tree)
        self._set_report_text("")
        self.status_var.set("Вывод очищен.")

    def _set_report_text(self, text: str) -> None:
        self.report_text.configure(state="normal")
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert("1.0", text)
        self.report_text.configure(state="disabled")

    def _validate_precision(self) -> int:
        try:
            digits = int(self.precision_spin.get())
        except ValueError as error:
            raise ValueError("Точность должна быть целым числом.") from error
        if digits < 0 or digits > 12:
            raise ValueError("Точность должна быть в диапазоне от 0 до 12.")
        return digits

    def _fill_summary(self, solver: EqualNodeInterpolationSolver, digits: int) -> None:
        self._clear_tree(self.summary_tree)
        for row in solver.summary_rows(digits):
            self.summary_tree.insert("", tk.END, values=row)

    def _fill_difference_table(
        self, solver: EqualNodeInterpolationSolver, digits: int
    ) -> None:
        headers, rows = solver.difference_table_view(digits)
        self._clear_tree(self.difference_tree)
        self.difference_tree.configure(columns=headers)
        for header in headers:
            width = 100 if header == "i" else 140
            self.difference_tree.heading(header, text=header)
            self.difference_tree.column(header, width=width, anchor="center")
        for row in rows:
            self.difference_tree.insert("", tk.END, values=row)

    def _calculate(self) -> None:
        try:
            digits = self._validate_precision()
            solver = EqualNodeInterpolationSolver.from_raw_inputs(
                self.function_entry.get(),
                self.nodes_entry.get(),
                self.points_entry.get(),
            )
            self.current_solver = solver
            self._fill_summary(solver, digits)
            self._fill_difference_table(solver, digits)
            self._set_report_text(solver.build_report(digits))
            self.status_var.set(
                f"Рассчитано: {len(solver.evaluation_points)} точк(и), шаг h = {solver.step}"
            )
        except Exception as error:
            messagebox.showerror("Ошибка", str(error))
            self.status_var.set("Не удалось выполнить расчёт.")

    def _show_help(self) -> None:
        help_window = tk.Toplevel(self)
        help_window.title("Справка")
        help_window.geometry("760x560")
        help_window.minsize(680, 500)

        frame = ttk.Frame(help_window, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        text = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
        )
        text.grid(row=0, column=0, sticky="nsew")
        text.insert(
            "1.0",
            (
                "Практическая работа №3\n"
                "Интерполирование функции в случае равноотстоящих узлов\n\n"
                "Что делает программа:\n"
                "1. Проверяет, что узлы заданы по возрастанию и являются равноотстоящими.\n"
                "2. Строит таблицу конечных разностей до последнего возможного порядка.\n"
                "3. Для каждой заданной точки автоматически выбирает подходящую формулу Ньютона:\n"
                "   прямую или обратную.\n"
                "4. Вычисляет приближённое значение, точное значение функции и абсолютную погрешность.\n\n"
                "Как вводить данные:\n"
                "- Функция f(x): выражение в синтаксисе Python/SymPy.\n"
                "  Примеры: 2*x**2 + x**5, sin(x) + x**2, exp(x)/(1+x)\n"
                "- Узлы: числа через запятую.\n"
                "  Пример: 0, 1, 2, 3, 4\n"
                "- Точки x: одна или несколько точек через запятую.\n"
                "  Пример: 4.4, 1.8\n\n"
                "Что выводится:\n"
                "- вкладка 'Результаты' содержит итоговую таблицу и подробный отчёт;\n"
                "- вкладка 'Конечные разности' содержит полную таблицу разностей.\n\n"
                "Вариант по умолчанию:\n"
                "f(x) = 2*x**2 + x**5\n"
                "Узлы: 0, 1, 2, 3, 4\n"
                "Точки: 4.4, 1.8\n\n"
                "Выполнил: Гаврилов Даниил\n"
                "Группа: СИ-35\n"
                "28.03.2026\n"
            ),
        )
        text.configure(state="disabled")


if __name__ == "__main__":
    app = Lab3App()
    app.mainloop()
