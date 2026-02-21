from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, font
from dataclasses import dataclass
from typing import List, Optional, Tuple
import sympy as sp
import numpy as np
import math
import sys

try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    _has_matplotlib = True
except Exception:
    _has_matplotlib = False


@dataclass
class LagrangeInterpolator:
    f_expr: str
    nodes: List[float]
    x_sym: sp.Symbol = sp.symbols('x')

    def __post_init__(self):
        self.f_sym: sp.Expr = sp.sympify(self.f_expr)
        if len(self.nodes) < 1:
            raise ValueError("Нужен хотя бы один узел интерполирования.")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Узлы должны быть различными.")
        self.n = len(self.nodes) - 1
        self._Lk_cache = None
        self._P_cache = None

    def basis_Lk(self) -> List[sp.Expr]:
        if self._Lk_cache is not None:
            return self._Lk_cache
        x = self.x_sym
        nodes_r = [sp.Rational(str(v)) for v in self.nodes]
        Lk_list = []
        for k, xk in enumerate(nodes_r):
            if len(nodes_r) == 1:
                Lk = sp.Integer(1)
            else:
                numer = sp.prod([(x - xj) for j, xj in enumerate(nodes_r) if j != k])
                denom = sp.prod([(xk - xj) for j, xj in enumerate(nodes_r) if j != k])
                Lk = sp.simplify(numer / denom)
            Lk_list.append(sp.simplify(Lk))
        self._Lk_cache = Lk_list
        return Lk_list

    def polynomial(self, force: bool = False) -> sp.Expr:
        if self._P_cache is not None and not force:
            return self._P_cache
        Lk = self.basis_Lk()
        x = self.x_sym
        fk_vals = [sp.simplify(self.f_sym.subs(x, sp.Rational(str(xk)))) for xk in self.nodes]
        P = sum(fk_vals[k] * Lk[k] for k in range(len(self.nodes)))
        P = sp.simplify(sp.expand(P))
        self._P_cache = P
        return P

    def evaluate_P(self, x0: float) -> float:
        P = self.polynomial()
        Pn = sp.lambdify(self.x_sym, P, 'numpy')
        return float(Pn(x0))

    def evaluate_f(self, x0: float) -> float:
        fn = sp.lambdify(self.x_sym, self.f_sym, 'numpy')
        return float(fn(x0))

    def real_error(self, x0: float) -> float:
        return abs(self.evaluate_f(x0) - self.evaluate_P(x0))

    def remainder_derivative(self) -> sp.Expr:
        order = self.n + 1
        return sp.simplify(sp.diff(self.f_sym, self.x_sym, order))

    def estimate_M(self, a: float, b: float, samples: int = 2001) -> float:
        deriv = self.remainder_derivative()
        deriv_num = sp.lambdify(self.x_sym, deriv, 'numpy')
        xs = np.linspace(a, b, samples)
        maxv = 0.0
        for xi in xs:
            try:
                v = float(abs(deriv_num(xi)))
                if np.isfinite(v) and v > maxv:
                    maxv = v
            except Exception:
                continue
        return maxv

    def remainder_bound(self, x0: float, M: Optional[float] = None,
                        samples_for_M: int = 2001) -> float:
        if M is None:
            a = min(min(self.nodes), x0)
            b = max(max(self.nodes), x0)
            M = self.estimate_M(a, b, samples_for_M)
        prod = 1.0
        for xk in self.nodes:
            prod *= (x0 - xk)
        return (M * abs(prod)) / math.factorial(self.n + 1)


class LagrangeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Практическая работа №1. Интерполирование с помощью многочлена Лагранжа")
        self.minsize(800, 600)
        self.geometry("800x600")
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

    def _bind_hover(self, btn: ttk.Button, border_frame: tk.Frame = None):
        btn.bind("<Enter>", lambda e: btn.configure(style="AccentHover.TButton"))
        btn.bind("<Leave>", lambda e: btn.configure(style="Accent.TButton"))
        if border_frame is not None:
            btn.bind("<Enter>", lambda e: border_frame.configure(bg="#111111"))
            btn.bind("<Leave>", lambda e: border_frame.configure(bg="#000000"))

    def _make_bordered_button(self, parent, **kwargs) -> Tuple[tk.Frame, ttk.Button]:
        border = tk.Frame(parent, bg="#000000", bd=0)
        btn = ttk.Button(border, **kwargs)
        btn.pack(fill="both", expand=True, padx=1, pady=1)
        return border, btn

    def _build_widgets(self):
        PAD_OUTER = 8
        PAD_IN = 6

        main = ttk.Frame(self, padding=PAD_OUTER, style="TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(main, padding=(PAD_IN, PAD_IN), style="TFrame")
        top.grid(row=0, column=0, sticky="nsew")
        main.rowconfigure(0, weight=0)
        main.rowconfigure(1, weight=0)
        main.rowconfigure(2, weight=0)
        main.rowconfigure(3, weight=1)
        main.columnconfigure(0, weight=1)

        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=0)
        top.columnconfigure(3, weight=0)

        ttk.Label(top, text="f(x) =", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.f_entry = ttk.Entry(top)
        self.f_entry.insert(0, "1 + 2/(x**3)")
        self.f_entry.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(5, 6))

        ttk.Label(top, text="Узлы x_k (через запятую):").grid(row=1, column=0, sticky="w")
        self.nodes_entry = ttk.Entry(top)
        self.nodes_entry.insert(0, "1, 2.7, 4, 3")
        self.nodes_entry.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(5, 6))

        ttk.Label(top, text="Точка x:").grid(row=2, column=0, sticky="w")
        self.x_entry = ttk.Entry(top, width=12)
        self.x_entry.insert(0, "1.5")
        self.x_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(5, 6))

        ttk.Label(top, text="Точность (знаков):").grid(row=2, column=2, sticky="e")
        self.prec_spin = ttk.Spinbox(top, from_=0, to=12, width=6)
        self.prec_spin.set("6")
        self.prec_spin.grid(row=2, column=3, sticky="w", pady=(3, 0))

        btn_frame = ttk.Frame(main, padding=(6, 6), style="TFrame")
        btn_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 6))
        for c in range(4):
            btn_frame.columnconfigure(c, weight=1, uniform="buttongrp")
        btn_frame.rowconfigure(0, weight=1)
        btn_frame.rowconfigure(1, weight=1)

        border1, self.poly_btn = self._make_bordered_button(
            btn_frame, text="Задание 1: \nинтерполяционный \nмногочлен Лагранжа", style="Accent.TButton",
            command=self.on_show_poly)
        border1.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 6), pady=(0, 0))

        border2, self.calc_btn = self._make_bordered_button(
            btn_frame, text="Задание 2: значение \nинтерполяционного \nполинома и оценка \nпогрешности",
            style="Accent.TButton", command=self.on_compute)
        border2.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=6, pady=(0, 0))

        border3, self.bound_btn = self._make_bordered_button(
            btn_frame, text="Оценить остаток", style="Accent.TButton", command=self.on_bound)
        border3.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=6, pady=(0, 0))

        border4, self.plot_btn = self._make_bordered_button(
            btn_frame,
            text="Построить график" if _has_matplotlib else "График недоступен",
            style="Accent.TButton",
            command=self.on_plot if _has_matplotlib else (
                lambda: messagebox.showwarning("matplotlib", "matplotlib не установлен")))

        border4.grid(row=0, column=3, sticky="nsew", padx=(6, 0), pady=(0, 6))

        border_help, self.help_btn = self._make_bordered_button(
            btn_frame, text="Справка", style="Accent.TButton", command=self.show_help)
        border_help.grid(row=1, column=3, sticky="nsew", padx=(6, 0), pady=(0, 0))

        for border, btn in ((border1, self.poly_btn), (border2, self.calc_btn), (border3, self.bound_btn),
                            (border4, self.plot_btn), (border_help, self.help_btn)):
            self._bind_hover(btn, border)

        sep = ttk.Separator(main, orient=tk.HORIZONTAL)
        sep.grid(row=2, column=0, sticky="ew", pady=(6, 8))

        out_frame = ttk.Frame(main, padding=(PAD_IN, PAD_IN), style="TFrame")
        out_frame.grid(row=3, column=0, sticky="nsew")
        main.rowconfigure(3, weight=1)
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

        self._writeln("Практическая работа №1. Интерполирование с помощью многочлена Лагранжа")
        self._writeln("В поле f(x) вводите выражение на Python/sympy, например: 1 + 2/(x**3)")
        self._writeln("Узлы: через запятую, например: 1, 2.7, 4, 3")
        self._writeln(
            "Нажмите 'Задание 1' для построения многочлена, 'Задание 2' — для вычисления значения и оценки.\n")

    def _get_inputs(self) -> Tuple[str, List[float], float, int]:
        f_expr = self.f_entry.get().strip()
        nodes_raw = self.nodes_entry.get().strip()
        if not nodes_raw:
            raise ValueError("Введите узлы через запятую.")
        nodes = []
        for s in nodes_raw.split(","):
            s = s.strip()
            if s == "":
                continue
            nodes.append(float(sp.N(sp.sympify(s))))
        if len(set(nodes)) != len(nodes):
            raise ValueError("Узлы должны быть различными (нет дубликатов).")
        x0 = float(sp.N(sp.sympify(self.x_entry.get().strip())))
        prec = int(self.prec_spin.get())
        return f_expr, nodes, x0, prec

    def on_show_poly(self):
        self.out.delete("1.0", tk.END)
        try:
            f_expr, nodes, x0, prec = self._get_inputs()
            interp = LagrangeInterpolator(f_expr=f_expr, nodes=nodes)
            P = interp.polynomial()

            self._writeln("=== Задание 1: Интерполяционный многочлен P(x) ===\n")

            pretty = sp.pretty(P, use_unicode=True)
            self._writeln("Полином P(x) (символьный вид):\n")
            self._writeln(pretty + "\n")

            try:
                fact = sp.factor(P)
                self._writeln("Факторизованный вид P(x):")
                self._writeln(sp.pretty(fact, use_unicode=True) + "\n")
            except Exception:
                pass

            poly = sp.Poly(P, interp.x_sym)
            coeffs_sym = poly.all_coeffs()
            coeffs_num = [float(c.evalf()) for c in coeffs_sym]
            fmt = lambda v: f"{v:.{prec}g}"
            self._writeln("Коэффициенты (старший -> младший):")
            self._writeln("  Символьные: " + ", ".join(str(c) for c in coeffs_sym))
            self._writeln("  Численные:  " + ", ".join(fmt(c) for c in coeffs_num))
            self._writeln(f"\nСтепень полинома: {poly.degree()}")
            self._writeln(f"Старший коэффициент: {coeffs_sym[0]}\n")

            self._writeln("Базисные многочлены L_k(x):\n")
            Lk_syms = interp.basis_Lk()
            for i, L in enumerate(Lk_syms):
                self._writeln(f"L_{i}(x) =")
                self._writeln(sp.pretty(L, use_unicode=True))
                x = interp.x_sym
                xk_r = sp.Rational(str(nodes[i]))
                denom = 1
                for j, xj in enumerate(nodes):
                    if j != i:
                        denom *= (xk_r - sp.Rational(str(xj)))
                self._writeln(f"  Знаменатель (для L_{i}): {sp.simplify(denom)}")
                try:
                    L_num = sp.lambdify(x, L, 'numpy')
                    vals = []
                    for j, xj in enumerate(nodes):
                        try:
                            v = float(L_num(xj))
                        except Exception:
                            v = float('nan')
                        vals.append(v)
                    vals_str = ", ".join(f"{v:.6g}" for v in vals)
                    self._writeln(f"  Значения L_{i}(x_j) для узлов (j=0..n): [{vals_str}]  (должно быть 0/1)")
                except Exception:
                    pass
                self._writeln("")

            self._writeln("Проверка совпадения P(x_k) = f(x_k) на узлах:")
            node_tol = 1e-9
            mismatches = []
            for xk in nodes:
                try:
                    pk = interp.evaluate_P(xk)
                    fk = interp.evaluate_f(xk)
                    diff = abs(pk - fk)
                    if not np.isfinite(pk) or not np.isfinite(fk) or diff > node_tol:
                        mismatches.append((xk, pk, fk, diff))
                    else:
                        self._writeln(f"  x={xk}: OK (P={fmt(pk)}, f={fmt(fk)}, |P-f|={diff:.2e})")
                except Exception as ex:
                    mismatches.append((xk, None, None, f"error: {ex}"))
            if mismatches:
                self._writeln("Найдены несоответствия на некоторых узлах:")
                for xm, pm, fm, d in mismatches:
                    self._writeln(f"  x={xm}: P={pm}, f={fm}, diff={d}")
                self._writeln("\n(Проверьте особенности функции в узлах или численную нестабильность.)")
            else:
                self._writeln("Все узлы удовлетворяют P(x_k)=f(x_k) в пределах порога " + str(node_tol))

            try:
                V = np.vander(np.array(nodes, dtype=float), increasing=False)
                condV = np.linalg.cond(V)
                detV = np.linalg.det(V)
                self._writeln("\nЧисленные характеристики Вандермондовой матрицы:")
                self._writeln(
                    f"  cond(V) ≈ {condV:.6g}  (значения ≳ 10^6 указывают на возможную численную нестабильность)")
                self._writeln(f"  det(V)  ≈ {detV:.6g}")
            except Exception:
                pass

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def on_compute(self):
        self.out.delete("1.0", tk.END)
        try:
            f_expr, nodes, x0, prec = self._get_inputs()
            interp = LagrangeInterpolator(f_expr=f_expr, nodes=nodes)

            y_nodes = [interp.evaluate_f(xk) for xk in nodes]

            self._writeln("=== Задание 2: Значение и оценка погрешности ===\n")
            self._writeln(f"f(x) = {f_expr}")
            self._writeln(f"Узлы: {nodes}")
            self._writeln(f"Точка x = {x0}")
            fmt = lambda v: f"{v:.{prec}g}"

            P_at = interp.evaluate_P(x0)
            f_at = interp.evaluate_f(x0)
            real_err = interp.real_error(x0)

            self._writeln(f"\nP({x0}) = {fmt(P_at)}")
            self._writeln(f"f({x0}) = {fmt(f_at)}")
            self._writeln(f"Реальная погрешность |f-P| = {fmt(real_err)}")

            self._writeln("\n--- Разложение Лагранжа в точке (по узлам) ---")
            self._writeln(f"{'i':<4} {'xi':<12} {'yi':<14} {'li(x)':<14} {'yi*li(x)':<14}")
            Lk_syms = interp.basis_Lk()
            for i, xk in enumerate(nodes):
                try:
                    li_fn = sp.lambdify(interp.x_sym, Lk_syms[i], 'numpy')
                    li_val = float(li_fn(x0))
                except Exception:
                    li_val = float('nan')
                yi = y_nodes[i]
                term = yi * li_val
                self._writeln(f"{i:<4} {xk:<12g} {yi:<14g} {li_val:<14g} {term:<14g}")

            self._writeln("\nP(x) =")
            self._writeln(sp.pretty(interp.polynomial(), use_unicode=True))

            deriv_expr = interp.remainder_derivative()
            deriv_num = sp.lambdify(interp.x_sym, deriv_expr, 'numpy')
            a = min(min(nodes), x0)
            b = max(max(nodes), x0)
            xs = np.linspace(a, b, 2001)
            M = 0.0
            t_max = None
            for xi in xs:
                try:
                    v = float(abs(deriv_num(xi)))
                    if np.isfinite(v) and v > M:
                        M = v
                        t_max = float(xi)
                except Exception:
                    continue

            self._writeln("\n--- (n+1)-я производная и численный поиск M ---")
            self._writeln(f"Символьное выражение f^({interp.n + 1})(x) = {sp.pretty(deriv_expr, use_unicode=True)}")
            if t_max is not None:
                self._writeln(f"Максимум |f^({interp.n + 1})(t)| на [{a}, {b}] ≈ {fmt(M)} при t ≈ {fmt(t_max)}")
            else:
                self._writeln(f"Не удалось найти максимум численно на [{a}, {b}], M ≈ {fmt(M)}")

            omega_x = 1.0
            for xk in nodes:
                omega_x *= (x0 - xk)
            bound = (M * abs(omega_x)) / math.factorial(interp.n + 1)

            self._writeln("\n--- Оценка остатка по формуле ---")
            self._writeln(f"Интервал [{a}, {b}]")
            self._writeln(f"M ≈ {fmt(M)}")
            self._writeln(f"omega({x0}) = {fmt(omega_x)}; |omega| = {fmt(abs(omega_x))}")
            self._writeln(f"(n+1)! = {math.factorial(interp.n + 1)}")
            self._writeln(f"Тогда |R_n(x)| ≤ (M / (n+1)!) * |omega(x)| ≈ {fmt(bound)}")

            tol_rel = 1e-12
            fits = False
            if bound == 0.0:
                fits = (real_err == 0.0)
            else:
                fits = real_err <= bound * (1 + tol_rel)
            self._writeln(f"\nРеальная погрешность |f-P| = {fmt(real_err)}")
            self._writeln(f"Укладывается ли реальная погрешность в теоретическую оценку? {'Да' if fits else 'Нет'}")

            self._writeln("\n--- Проверка совпадения значений на узлах ---")
            node_tol = 1e-9
            mismatches = []
            for xk in nodes:
                try:
                    pk = interp.evaluate_P(xk)
                    fk = interp.evaluate_f(xk)
                    diff = abs(pk - fk)
                    if not np.isfinite(pk) or not np.isfinite(fk) or diff > node_tol:
                        mismatches.append((xk, pk, fk, diff))
                except Exception as ex:
                    mismatches.append((xk, None, None, f"error: {ex}"))
            if not mismatches:
                self._writeln(f"P(x_k) = f(x_k) выполнено для всех узлов (порог: {node_tol}).")
            else:
                self._writeln("Найдены несоответствия на узлах:")
                for xm, pm, fm, d in mismatches:
                    self._writeln(f"  x={xm}: P={pm}, f={fm}, |P-f|={d}")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def on_bound(self):
        self.out.delete("1.0", tk.END)
        try:
            f_expr, nodes, x0, prec = self._get_inputs()
            interp = LagrangeInterpolator(f_expr=f_expr, nodes=nodes)
            a = min(min(nodes), x0)
            b = max(max(nodes), x0)
            M = interp.estimate_M(a, b, samples=2001)
            bound = interp.remainder_bound(x0, M=M)
            fmt = lambda v: f"{v:.{prec}g}"
            self._writeln("=== Оценка остатка интерполирования ===\n")
            self._writeln(f"Интервал для оценки: [{a}, {b}]")
            self._writeln(f"Оценка M = max |f^{interp.n + 1}(ξ)| ≈ {fmt(M)}")
            self._writeln(f"Тогда |R_n(x)| ≤ M/(n+1)! * Π|x - x_k| ≈ {fmt(bound)}")
            real_err = interp.real_error(x0)
            self._writeln(f"\nРеальная погрешность |f-P| = {fmt(real_err)}")

            tol_rel = 1e-12
            fits = False
            if bound == 0.0:
                fits = (real_err == 0.0)
            else:
                fits = real_err <= bound * (1 + tol_rel)
            self._writeln(f"Реальная погрешность укладывается в верхнюю оценку? {'Да' if fits else 'Нет'}")

            self._writeln("\nПроверка совпадения значений на узлах:")
            node_tol = 1e-9
            mismatches = []
            for xk in nodes:
                try:
                    pk = interp.evaluate_P(xk)
                    fk = interp.evaluate_f(xk)
                    diff = abs(pk - fk)
                    if not np.isfinite(pk) or not np.isfinite(fk) or diff > node_tol:
                        mismatches.append((xk, pk, fk, diff))
                except Exception as ex:
                    mismatches.append((xk, None, None, f"error: {ex}"))
            if not mismatches:
                self._writeln(f"P(x_k) = f(x_k) выполнено для всех узлов (порог: {node_tol}).")
            else:
                for xm, pm, fm, d in mismatches:
                    self._writeln(f"  x={xm}: P={pm}, f={fm}, |P-f|={d}")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def on_plot(self):
        if not _has_matplotlib:
            messagebox.showwarning("matplotlib", "matplotlib не установлен, график недоступен.")
            return
        try:
            f_expr, nodes, x0, prec = self._get_inputs()
            interp = LagrangeInterpolator(f_expr=f_expr, nodes=nodes)

            a = min(nodes) - 0.5
            b = max(nodes) + 0.5
            xs = np.linspace(a, b, 500)
            f_num = sp.lambdify(interp.x_sym, interp.f_sym, 'numpy')
            P_num = sp.lambdify(interp.x_sym, interp.polynomial(), 'numpy')

            f_vals = []
            P_vals = []
            for xi in xs:
                try:
                    f_vals.append(float(f_num(xi)))
                except Exception:
                    f_vals.append(np.nan)
                try:
                    P_vals.append(float(P_num(xi)))
                except Exception:
                    P_vals.append(np.nan)

            top = tk.Toplevel(self)
            top.title("График f(x) и P(x)")
            top.geometry("1024x768")
            top.minsize(1024, 768)

            fig = Figure(figsize=(9.5, 5.5), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(xs, f_vals, label='f(x)')
            ax.plot(xs, P_vals, linestyle='--', label='P(x)')
            ynodes = [interp.evaluate_f(xk) for xk in nodes]
            ax.scatter(nodes, ynodes, marker='o', label='узлы', zorder=5)
            ax.set_title("f(x) и интерполяционный многочлен P(x)")
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
            "21.02.2026"
        )
        txt.insert("1.0", help_text)
        txt.configure(state="disabled")

    def _writeln(self, s: str = ""):
        self.out.insert(tk.END, s + "\n")
        self.out.see(tk.END)


if __name__ == "__main__":
    app = LagrangeApp()
    app.mainloop()
