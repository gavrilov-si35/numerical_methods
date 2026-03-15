from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, font
from typing import List, Tuple
import sympy as sp
import numpy as np
import math
import sys

from interp import LagrangeInterpolator, NewtonInterpolator

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    _has_matplotlib = True
except Exception:
    _has_matplotlib = False

class LagrangeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Практическая работа №2. Интерполирование с помощью многочлена Ньютона")
        self.minsize(900, 650)
        self.geometry("900x700")
        self._init_style()
        self._build_widgets()

    def _init_style(self):
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=10)
        self.option_add("*Font", default_font)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        self._bg = "#f4f6f8"
        self._accent = "#2b6ea3"
        self._accent_hover = "#1e4b6c"
        self._muted = "#6b7280"
        style.configure("TFrame", background=self._bg)
        style.configure("TLabel", background=self._bg, foreground=self._muted)
        style.configure("Title.TLabel", font=("TkDefaultFont", 12, "bold"), foreground="#1f2937", background=self._bg)
        style.configure("Accent.TButton",
                        font=("TkDefaultFont", 10, "bold"),
                        foreground="white",
                        background=self._accent,
                        padding=(8, 6),
                        relief="flat",
                        borderwidth=0)
        style.configure("AccentHover.TButton",
                        font=("TkDefaultFont", 10, "bold"),
                        foreground="white",
                        background=self._accent_hover,
                        padding=(8, 6),
                        relief="flat",
                        borderwidth=0)
        style.map("Accent.TButton",
                  foreground=[("active", "white"), ("disabled", "#d1d5db")],
                  background=[("active", self._accent_hover), ("disabled", "#c0c4c8")])

    def _bind_hover(self, button_widget: ttk.Button, border_frame: tk.Frame = None):
        button_widget.bind("<Enter>", lambda e: button_widget.configure(style="AccentHover.TButton"))
        button_widget.bind("<Leave>", lambda e: button_widget.configure(style="Accent.TButton"))
        if border_frame is not None:
            button_widget.bind("<Enter>", lambda e: border_frame.configure(bg="#111111"))
            button_widget.bind("<Leave>", lambda e: border_frame.configure(bg="#000000"))

    def _make_bordered_button(self, parent_frame, **kwargs):
        border_frame = tk.Frame(parent_frame, bg="#000000", bd=0)
        button_widget = ttk.Button(border_frame, **kwargs)
        button_widget.pack(fill="both", expand=True, padx=1, pady=1)
        return border_frame, button_widget

    def _build_widgets(self):
        PAD_OUTER = 8
        PAD_IN = 6
        main_frame = ttk.Frame(self, padding=PAD_OUTER, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        top_frame = ttk.Frame(main_frame, padding=(PAD_IN, PAD_IN), style="TFrame")
        top_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.rowconfigure(0, weight=0)
        main_frame.rowconfigure(1, weight=0)
        main_frame.rowconfigure(2, weight=0)
        main_frame.rowconfigure(3, weight=1)
        main_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)
        top_frame.columnconfigure(2, weight=0)
        top_frame.columnconfigure(3, weight=0)
        ttk.Label(top_frame, text="f(x) =", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.function_entry = ttk.Entry(top_frame)
        self.function_entry.insert(0, "1 + 2/(x**3)")
        self.function_entry.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(5, 6))
        ttk.Label(top_frame, text="Узлы x_k (через запятую):").grid(row=1, column=0, sticky="w")
        self.nodes_entry = ttk.Entry(top_frame)
        self.nodes_entry.insert(0, "1, 2.7, 4, 3")
        self.nodes_entry.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(5, 6))
        ttk.Label(top_frame, text="Точка x:").grid(row=2, column=0, sticky="w")
        self.x_entry = ttk.Entry(top_frame, width=12)
        self.x_entry.insert(0, "1.5")
        self.x_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(5, 6))
        ttk.Label(top_frame, text="Точность (знаков):").grid(row=2, column=2, sticky="e")
        self.precision_spin = ttk.Spinbox(top_frame, from_=0, to=12, width=6)
        self.precision_spin.set("4")
        self.precision_spin.grid(row=2, column=3, sticky="w", pady=(3, 0))
        btn_frame = ttk.Frame(main_frame, padding=(6, 6), style="TFrame")
        btn_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 6))
        for column_index in range(4):
            btn_frame.columnconfigure(column_index, weight=1, uniform="buttongrp")
        btn_frame.rowconfigure(0, weight=1)
        btn_frame.rowconfigure(1, weight=1)
        border1, self.lagrange_button = self._make_bordered_button(
            btn_frame,
            text="Многочлен Лагранжа:\nполином, значение, остаток",
            style="Accent.TButton",
            command=self.on_lagrange
        )
        border1.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 6), pady=(0, 0))
        border2, self.newton_button = self._make_bordered_button(
            btn_frame, text="Задание 1:\nМногочлен Ньютона", style="Accent.TButton", command=self.on_show_newton)
        border2.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=6, pady=(0, 0))
        border3, self.compare_button = self._make_bordered_button(
            btn_frame, text="Задание 2:\nСравнение Лагранж/Ньютон", style="Accent.TButton", command=self.on_compare)
        border3.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=6, pady=(0, 0))
        border4, self.plot_button = self._make_bordered_button(
            btn_frame,
            text="Построить график" if _has_matplotlib else "График недоступен",
            style="Accent.TButton",
            command=self.on_plot if _has_matplotlib else (lambda: messagebox.showwarning("matplotlib", "matplotlib не установлен")))
        border4.grid(row=0, column=3, sticky="nsew", padx=(6, 0), pady=(0, 6))
        border_help, self.help_button = self._make_bordered_button(
            btn_frame, text="Справка", style="Accent.TButton", command=self.show_help)
        border_help.grid(row=1, column=3, sticky="nsew", padx=(6, 0), pady=(0, 0))
        for border_frame, button_widget in ((border1, self.lagrange_button), (border2, self.newton_button), (border3, self.compare_button),
                            (border4, self.plot_button), (border_help, self.help_button)):
            self._bind_hover(button_widget, border_frame)
        sep = ttk.Separator(main_frame, orient=tk.HORIZONTAL)
        sep.grid(row=2, column=0, sticky="ew", pady=(6, 8))
        out_frame = ttk.Frame(main_frame, padding=(PAD_IN, PAD_IN), style="TFrame")
        out_frame.grid(row=3, column=0, sticky="nsew")
        main_frame.rowconfigure(3, weight=1)
        out_frame.columnconfigure(0, weight=1)
        out_frame.rowconfigure(0, weight=1)
        self.out = scrolledtext.ScrolledText(out_frame, wrap=tk.WORD, font=("Courier", 10))
        self.out.grid(row=0, column=0, sticky="nsew")
        self.out.configure(selectbackground="#3399ff", selectforeground="white")
        self._out_menu = tk.Menu(self, tearoff=0)
        self._out_menu.add_command(label="Копировать", command=lambda: self.out.event_generate("<<Copy>>"))
        self._out_menu.add_command(label="Выделить всё", command=lambda: self.out.tag_add("sel", "1.0", "end"))
        self._out_menu.add_command(label="Копировать всё", command=lambda: (self.clipboard_clear(),
                                                                            self.clipboard_append(
                                                                                self.out.get("1.0", "end"))))
        def _show_out_menu(event):
            try:
                self._out_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self._out_menu.grab_release()
        self.out.bind("<Button-3>", _show_out_menu)
        self.out.bind("<Button-2>", _show_out_menu)
        if sys.platform == "darwin":
            self.out.bind("<Command-a>", lambda e: (self.out.tag_add("sel", "1.0", "end"), "break"))
            self.out.bind("<Command-A>", lambda e: (self.out.tag_add("sel", "1.0", "end"), "break"))
            self.out.bind("<Command-c>", lambda e: (self.out.event_generate("<<Copy>>"), "break"))
            self.out.bind("<Command-C>", lambda e: (self.out.event_generate("<<Copy>>"), "break"))
        else:
            self.out.bind("<Control-a>", lambda e: (self.out.tag_add("sel", "1.0", "end"), "break"))
            self.out.bind("<Control-A>", lambda e: (self.out.tag_add("sel", "1.0", "end"), "break"))
            self.out.bind("<Control-c>", lambda e: (self.out.event_generate("<<Copy>>"), "break"))
            self.out.bind("<Control-C>", lambda e: (self.out.event_generate("<<Copy>>"), "break"))
        self._writeln("Практическая работа 2. Интерполирование с помощью многочлена Ньютона")
        self._writeln("В поле f(x) вводите выражение на Python/sympy, например: 1 + 2/(x**3)")
        self._writeln("Узлы: через запятую, например: 1, 2.7, 4, 3")
        self._writeln(
            "Нажмите 'Многочлен Лагранжа' для вывода полинома Лагранжа + значение + оценка остатка,\n"
            "'Задание 1: Многочлен Ньютона' — для Ньютона, 'Задание 2: Сравнение' — для сравнения.\n")

    def _get_inputs(self) -> Tuple[str, List[float], float, int]:
        function_expression = self.function_entry.get().strip()
        nodes_raw = self.nodes_entry.get().strip()
        if not nodes_raw:
            raise ValueError("Введите узлы через запятую.")
        node_values = []
        for token in nodes_raw.split(","):
            token = token.strip()
            if token == "":
                continue
            node_values.append(float(sp.N(sp.sympify(token))))
        if len(set(node_values)) != len(node_values):
            raise ValueError("Узлы должны быть различными (нет дубликатов).")
        evaluation_point = float(sp.N(sp.sympify(self.x_entry.get().strip())))
        precision_digits = int(self.precision_spin.get())
        return function_expression, node_values, evaluation_point, precision_digits

    def _format_number(self, value, precision_digits):
        try:
            numeric_value = float(value)
        except Exception:
            return str(value)
        if not np.isfinite(numeric_value):
            return str(numeric_value)
        return f"{numeric_value:.{precision_digits}f}"

    def on_lagrange(self):
        self.out.delete("1.0", tk.END)
        try:
            function_expression, node_values, evaluation_point, precision_digits = self._get_inputs()
            interpolator = LagrangeInterpolator(function_expression=function_expression, nodes=node_values)
            fmt = lambda v: self._format_number(v, precision_digits)
            self._writeln("=== Многочлен Лагранжа, значение и оценка остатка ===\n")
            self._writeln(f"f(x) = {function_expression}")
            self._writeln(f"Узлы: {node_values}")
            self._writeln(f"Точка x = {evaluation_point}\n")
            polynomial_expr = interpolator.polynomial()
            self._writeln("Полином P(x) (символьный вид):")
            self._writeln(sp.pretty(polynomial_expr, use_unicode=True) + "\n")
            try:
                factored_expr = sp.factor(polynomial_expr)
                self._writeln("Факторизованный вид P(x):")
                self._writeln(sp.pretty(factored_expr, use_unicode=True) + "\n")
            except Exception:
                pass
            poly_obj = sp.Poly(polynomial_expr, interpolator.x_symbol)
            coeffs_symbolic = poly_obj.all_coeffs()
            coeffs_numeric = [float(c.evalf()) for c in coeffs_symbolic]
            self._writeln("Коэффициенты (старший -> младший):")
            self._writeln("  Символьные: " + ", ".join(str(c) for c in coeffs_symbolic))
            self._writeln("  Численные:  " + ", ".join(fmt(c) for c in coeffs_numeric))
            self._writeln(f"\nСтепень полинома: {poly_obj.degree()}")
            self._writeln(f"Старший коэффициент: {coeffs_symbolic[0]}\n")
            self._writeln("Базисные многочлены L_k(x) и проверка значений:")
            basis_list = interpolator.basis_Lk()
            for index_i, basis_poly in enumerate(basis_list):
                self._writeln(f"L_{index_i}(x) =")
                self._writeln(sp.pretty(basis_poly, use_unicode=True))
                x_sym = interpolator.x_symbol
                try:
                    basis_func = sp.lambdify(x_sym, basis_poly, 'numpy')
                    basis_values = []
                    for node_value in node_values:
                        try:
                            v = float(basis_func(node_value))
                        except Exception:
                            v = float('nan')
                        basis_values.append(v)
                    basis_values_str = ", ".join(self._format_number(v, precision_digits) for v in basis_values)
                    self._writeln(f"  Значения L_{index_i}(x_j) для узлов (j=0..n): [{basis_values_str}]  (должно быть 0/1)")
                except Exception:
                    pass
                self._writeln("")
            self._writeln("Проверка P(x_k) = f(x_k) на узлах:")
            for node_value in node_values:
                try:
                    p_at_node = interpolator.evaluate_polynomial(node_value)
                    f_at_node = interpolator.evaluate_function(node_value)
                    diff_value = abs(p_at_node - f_at_node)
                    self._writeln(f"  x={node_value}: P={fmt(p_at_node)}, f={fmt(f_at_node)}, |P-f|={fmt(diff_value)}")
                except Exception as ex:
                    self._writeln(f"  x={node_value}: Ошибка при вычислении: {ex}")
            self._writeln("")
            polynomial_at_point = interpolator.evaluate_polynomial(evaluation_point)
            function_at_point = interpolator.evaluate_function(evaluation_point)
            real_error_value = interpolator.real_error(evaluation_point)
            self._writeln(f"P({evaluation_point}) = {fmt(polynomial_at_point)}")
            self._writeln(f"f({evaluation_point}) = {fmt(function_at_point)}")
            self._writeln(f"Реальная погрешность |f-P| = {fmt(real_error_value)}\n")
            derivative_expr = interpolator.remainder_derivative()
            derivative_func = sp.lambdify(interpolator.x_symbol, derivative_expr, 'numpy')
            interval_a = min(min(node_values), evaluation_point)
            interval_b = max(max(node_values), evaluation_point)
            sample_points = np.linspace(interval_a, interval_b, 2001)
            M_value = 0.0
            t_at_max = None
            for sample_point in sample_points:
                try:
                    val = float(abs(derivative_func(sample_point)))
                    if np.isfinite(val) and val > M_value:
                        M_value = val
                        t_at_max = float(sample_point)
                except Exception:
                    continue
            self._writeln("--- Оценка остатка ---")
            self._writeln(f"Символьное выражение f^({interpolator.degree + 1})(x) = {sp.pretty(derivative_expr, use_unicode=True)}")
            if t_at_max is not None:
                self._writeln(f"Максимум |f^({interpolator.degree + 1})(t)| на [{interval_a}, {interval_b}] ≈ {fmt(M_value)} при t ≈ {fmt(t_at_max)}")
            else:
                self._writeln(f"Не удалось найти максимум численно на [{interval_a}, {interval_b}], M ≈ {fmt(M_value)}")
            omega_value = 1.0
            for node_value in node_values:
                omega_value *= (evaluation_point - node_value)
            bound_value = (M_value * abs(omega_value)) / math.factorial(interpolator.degree + 1)
            self._writeln(f"omega({evaluation_point}) = {fmt(omega_value)}; |omega| = {fmt(abs(omega_value))}")
            self._writeln(f"(n+1)! = {math.factorial(interpolator.degree + 1)}")
            self._writeln(f"Тогда |R_n(x)| ≤ (M / (n+1)!) * |omega(x)| ≈ {fmt(bound_value)}")
            fits_flag = False
            if bound_value == 0.0:
                fits_flag = (real_error_value == 0.0)
            else:
                fits_flag = real_error_value <= bound_value * (1 + 1e-12)
            self._writeln(f"\nРеальная погрешность укладывается в теоретическую оценку? {'Да' if fits_flag else 'Нет'}")
            try:
                V = np.vander(np.array(node_values, dtype=float), increasing=False)
                condV = np.linalg.cond(V)
                detV = np.linalg.det(V)
                self._writeln("\nЧисленные характеристики Вандермондовой матрицы:")
                self._writeln(f"  cond(V) ≈ {self._format_number(condV, precision_digits)}")
                self._writeln(f"  det(V)  ≈ {self._format_number(detV, precision_digits)}")
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def on_show_newton(self):
        self.out.delete("1.0", tk.END)
        try:
            function_expression, node_values, evaluation_point, precision_digits = self._get_inputs()
            newton_interpolator = NewtonInterpolator(function_expression=function_expression, nodes=node_values)
            divided_diff_table = newton_interpolator.divided_differences()
            coeffs = newton_interpolator.coefficients()
            polynomial_newton = newton_interpolator.polynomial()
            fmt = lambda v: self._format_number(v, precision_digits)
            self._writeln("=== Интерполяционный многочлен Ньютона ===\n")
            self._writeln(f"Узлы: {node_values}")
            self._writeln(f"f(x) = {function_expression}\n")
            self._writeln("--- Таблица разделённых разностей (первые элементы каждого порядка) ---")
            for order_index, column_vals in enumerate(divided_diff_table):
                row_values_strs = []
                for value_expr in column_vals:
                    try:
                        row_values_strs.append(fmt(float(value_expr.evalf())))
                    except Exception:
                        try:
                            row_values_strs.append(fmt(float(value_expr)))
                        except Exception:
                            row_values_strs.append(str(value_expr))
                self._writeln(f"Порядок {order_index}: {', '.join(row_values_strs)}")
            self._writeln("")
            self._writeln("Коэффициенты (первые элементы столбцов):")
            for k_index, coeff_expr in enumerate(coeffs):
                try:
                    coeff_numeric = float(coeff_expr.evalf())
                    self._writeln(f"a_{k_index} = {sp.pretty(coeff_expr)} ≈ {fmt(coeff_numeric)}")
                except Exception:
                    self._writeln(f"a_{k_index} = {sp.pretty(coeff_expr)}")
            self._writeln("")
            self._writeln("Многочлен Ньютона P_n(x):\n")
            self._writeln(sp.pretty(polynomial_newton, use_unicode=True))
            self._writeln("\nПроверка P_n(x_k) = f(x_k):")
            for node_value in node_values:
                try:
                    p_at_node = newton_interpolator.evaluate_polynomial(node_value)
                    f_at_node = newton_interpolator.evaluate_function(node_value)
                    diff_value = abs(p_at_node - f_at_node)
                    self._writeln(f"  x={node_value}: P={fmt(p_at_node)}, f={fmt(f_at_node)}, |P-f|={fmt(diff_value)}")
                except Exception as ex:
                    self._writeln(f"  x={node_value}: Ошибка при вычислении: {ex}")
            polynomial_at_eval = newton_interpolator.evaluate_polynomial(evaluation_point)
            function_at_eval = newton_interpolator.evaluate_function(evaluation_point)
            self._writeln(f"\nP_n({evaluation_point}) = {fmt(polynomial_at_eval)}")
            self._writeln(f"f({evaluation_point}) = {fmt(function_at_eval)}")
            self._writeln(f"Реальная погрешность |f-P_n| = {fmt(abs(function_at_eval - polynomial_at_eval))}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def on_compare(self):
        self.out.delete("1.0", tk.END)
        try:
            function_expression, node_values, evaluation_point, precision_digits = self._get_inputs()
            lag_interpolator = LagrangeInterpolator(function_expression=function_expression, nodes=node_values)
            newton_interpolator = NewtonInterpolator(function_expression=function_expression, nodes=node_values)
            polynomial_lagrange = lag_interpolator.polynomial()
            polynomial_newton = newton_interpolator.polynomial()
            fmt = lambda v: self._format_number(v, precision_digits)
            self._writeln("=== Сравнение Лагранж / Ньютон ===\n")
            self._writeln("Полином Лагранжа P_L(x):")
            self._writeln(sp.pretty(polynomial_lagrange, use_unicode=True) + "\n")
            self._writeln("Полином Ньютона P_N(x):")
            self._writeln(sp.pretty(polynomial_newton, use_unicode=True) + "\n")
            diff_sym = sp.simplify(sp.expand(polynomial_lagrange - polynomial_newton))
            self._writeln("Символьная разность P_L(x) - P_N(x) (упрощённо):")
            self._writeln(sp.pretty(diff_sym, use_unicode=True) + "\n")
            identical_flag = diff_sym == 0
            self._writeln(f"Тождественно равны? {'Да' if identical_flag else 'Нет (проверьте численно)'}")
            test_points = np.linspace(min(node_values) - 0.5, max(node_values) + 0.5, 9)
            lag_func = sp.lambdify(lag_interpolator.x_symbol, polynomial_lagrange, 'numpy')
            newt_func = sp.lambdify(newton_interpolator.x_symbol, polynomial_newton, 'numpy')
            max_difference = 0.0
            for test_point in test_points:
                try:
                    value_diff = abs(float(lag_func(test_point) - newt_func(test_point)))
                    if np.isfinite(value_diff) and value_diff > max_difference:
                        max_difference = value_diff
                except Exception:
                    continue
            self._writeln(f"Максимум |P_L-P_N| на тестовой сетке ≈ {fmt(max_difference)}")
            val_l = lag_interpolator.evaluate_polynomial(evaluation_point)
            val_n = newton_interpolator.evaluate_polynomial(evaluation_point)
            f_at_point = lag_interpolator.evaluate_function(evaluation_point)
            self._writeln(f"\nВ точке x = {evaluation_point}:")
            self._writeln(f"P_L({evaluation_point}) = {fmt(val_l)}")
            self._writeln(f"P_N({evaluation_point}) = {fmt(val_n)}")
            self._writeln(f"f({evaluation_point})   = {fmt(f_at_point)}")
            self._writeln(f"|P_L - P_N| = {fmt(abs(val_l - val_n))}")
            self._writeln(f"|f - P_L| = {fmt(abs(f_at_point - val_l))}")
            self._writeln(f"|f - P_N| = {fmt(abs(f_at_point - val_n))}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def on_plot(self):
        if not _has_matplotlib:
            messagebox.showwarning("matplotlib", "matplotlib не установлен, график недоступен.")
            return
        try:
            function_expression, node_values, evaluation_point, precision_digits = self._get_inputs()
            lag_interpolator = LagrangeInterpolator(function_expression=function_expression, nodes=node_values)
            newton_interpolator = NewtonInterpolator(function_expression=function_expression, nodes=node_values)
            interval_a = min(node_values) - 0.5
            interval_b = max(node_values) + 0.5
            sample_xs = np.linspace(interval_a, interval_b, 500)
            f_func = sp.lambdify(lag_interpolator.x_symbol, lag_interpolator.f_symbol, 'numpy')
            Pl_func = sp.lambdify(lag_interpolator.x_symbol, lag_interpolator.polynomial(), 'numpy')
            Pn_func = sp.lambdify(newton_interpolator.x_symbol, newton_interpolator.polynomial(), 'numpy')
            f_vals = []
            Pl_vals = []
            Pn_vals = []
            for sample_x in sample_xs:
                try:
                    f_vals.append(float(f_func(sample_x)))
                except Exception:
                    f_vals.append(np.nan)
                try:
                    Pl_vals.append(float(Pl_func(sample_x)))
                except Exception:
                    Pl_vals.append(np.nan)
                try:
                    Pn_vals.append(float(Pn_func(sample_x)))
                except Exception:
                    Pn_vals.append(np.nan)
            top = tk.Toplevel(self)
            top.title("График f(x), P_L(x) и P_N(x)")
            top.geometry("1024x768")
            top.minsize(1024, 768)
            fig = Figure(figsize=(9.5, 5.5), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(sample_xs, f_vals, label='f(x)')
            ax.plot(sample_xs, Pl_vals, linestyle='--', label='P_L(x) (Лагранж)')
            ax.plot(sample_xs, Pn_vals, linestyle=':', label='P_N(x) (Ньютон)')
            ynodes = [lag_interpolator.evaluate_function(node_value) for node_value in node_values]
            ax.scatter(node_values, ynodes, marker='o', label='узлы', zorder=5)
            ax.set_title("f(x) и интерполяционные многочлены")
            ax.grid(True)
            ax.legend()
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            toolbar = NavigationToolbar2Tk(canvas, top)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def show_help(self):
        help_win = tk.Toplevel(self)
        help_win.title("Справка")
        help_win.geometry("640x480")
        help_win.minsize(640, 480)
        frm = ttk.Frame(help_win, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)
        txt = scrolledtext.ScrolledText(frm, wrap=tk.WORD, font=("TkDefaultFont", 10))
        txt.pack(fill=tk.BOTH, expand=True)
        help_text = (
            "Формат ввода данных\n\n"
            "1) Формула f(x):\n"
            "- Вводите выражение в синтаксисе Python / sympy. Примеры:\n"
            "  1 + 2/(x**3)\n"
            "  sin(x) + x**2\n"
            "  exp(x) / (1 + x)\n"
            "- Разрешены функции sympy (sin, cos, exp, log и т.д.).\n\n"
            "2) Узлы x_k:\n"
            "- Перечислите числа через запятую, например: 1, 2.7, 4, 3\n"
            "- Можно использовать выражения, допустимые для sympy: 1, 3/2, 2.5\n"
            "- Узлы должны быть различны (нет дубликатов).\n\n"
            "3) Точка x:\n"
            "- Одна числовая величина (например 1.5 или 3/2).\n\n"
            "4) Точность вывода:\n"
            "- Количество значащих цифр для вывода чисел.\n\n"
            "Выполнил:\n"
            "Гаврилов Даниил\n"
            "Группа:\n"
            "СИ-35\n\n"
            "15.03.2026"
        )
        txt.insert("1.0", help_text)
        txt.configure(state="disabled")

    def _writeln(self, s: str = ""):
        self.out.insert(tk.END, s + "\n")
        self.out.see(tk.END)

if __name__ == "__main__":
    app = LagrangeApp()
    app.mainloop()