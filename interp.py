from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Sequence

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

X_SYMBOL = sp.symbols("x")
DEFAULT_FUNCTION = "2*x**2 + x**5"
DEFAULT_NODES_TEXT = "0, 1, 2, 3, 4"
DEFAULT_POINTS_TEXT = "4.4, 1.8"
_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def _parse_sympy_expression(text: str, x_symbol: sp.Symbol = X_SYMBOL) -> sp.Expr:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Пустое выражение вводить нельзя.")
    try:
        expression = parse_expr(
            cleaned,
            local_dict={str(x_symbol): x_symbol},
            global_dict=dict(sp.__dict__),
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as error:
        raise ValueError(f"Не удалось разобрать выражение: {text}") from error
    return sp.simplify(expression)


def parse_scalar(value: str | int | float | sp.Expr, x_symbol: sp.Symbol = X_SYMBOL) -> sp.Expr:
    expression = _parse_sympy_expression(str(value), x_symbol)
    if expression.free_symbols:
        raise ValueError(f"Ожидалось число, но получено выражение: {value}")
    if expression.is_real is False:
        raise ValueError(f"Допустимы только вещественные значения: {value}")
    return sp.nsimplify(expression)


def parse_function(expression: str, x_symbol: sp.Symbol = X_SYMBOL) -> sp.Expr:
    parsed = _parse_sympy_expression(expression, x_symbol)
    unknown_symbols = parsed.free_symbols - {x_symbol}
    if unknown_symbols:
        symbol_names = ", ".join(sorted(str(symbol) for symbol in unknown_symbols))
        raise ValueError(
            f"В выражении функции есть недопустимые символы: {symbol_names}"
        )
    return parsed


def parse_csv_numbers(text: str, field_name: str, x_symbol: sp.Symbol = X_SYMBOL) -> tuple[sp.Expr, ...]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"Поле '{field_name}' не должно быть пустым.")
    return tuple(parse_scalar(part, x_symbol) for part in parts)


def _trimmed_decimal(value: sp.Expr, digits: int) -> str:
    numeric_value = float(sp.N(value))
    text = f"{numeric_value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def describe_value(value: sp.Expr, digits: int) -> str:
    simplified = sp.simplify(value)
    symbolic = sp.sstr(simplified)
    decimal = _trimmed_decimal(simplified, digits)
    if simplified.is_Integer:
        return symbolic
    if simplified.is_Rational:
        return f"{symbolic} ~= {decimal}"
    if symbolic == decimal:
        return decimal
    return f"{symbolic} ~= {decimal}"


@dataclass(frozen=True)
class FormulaTerm:
    order: int
    difference_label: str
    factor_text: str
    difference_value: sp.Expr
    term_value: sp.Expr


@dataclass(frozen=True)
class PointInterpolationResult:
    x_value: sp.Expr
    formula_kind: str
    formula_name: str
    formula_short_name: str
    reason: str
    parameter_symbol: str
    parameter_value: sp.Expr
    anchor_node: sp.Expr
    formula_template: str
    approximate_value: sp.Expr
    exact_value: sp.Expr
    absolute_error: sp.Expr
    terms: tuple[FormulaTerm, ...]


class EqualNodeInterpolationSolver:
    def __init__(
        self,
        function_expression: str,
        nodes: Sequence[str | int | float | sp.Expr],
        evaluation_points: Sequence[str | int | float | sp.Expr],
        x_symbol: sp.Symbol = X_SYMBOL,
    ) -> None:
        self.x_symbol = x_symbol
        self.function_expression = function_expression.strip()
        self.function_expr = parse_function(self.function_expression, self.x_symbol)
        self.nodes = tuple(parse_scalar(node, self.x_symbol) for node in nodes)
        self.evaluation_points = tuple(
            parse_scalar(point, self.x_symbol) for point in evaluation_points
        )
        self._validate_nodes()
        self.step = sp.simplify(self.nodes[1] - self.nodes[0])
        self.y_values = tuple(
            sp.simplify(self.function_expr.subs(self.x_symbol, node))
            for node in self.nodes
        )
        self.finite_differences = self._build_finite_differences()
        self.interpolation_polynomial = self._build_interpolation_polynomial()
        self._point_results_cache: tuple[PointInterpolationResult, ...] | None = None

    @classmethod
    def from_raw_inputs(
        cls,
        function_expression: str,
        nodes_text: str,
        points_text: str,
        x_symbol: sp.Symbol = X_SYMBOL,
    ) -> "EqualNodeInterpolationSolver":
        nodes = parse_csv_numbers(nodes_text, "Узлы", x_symbol)
        points = parse_csv_numbers(points_text, "Точки вычисления", x_symbol)
        return cls(function_expression, nodes, points, x_symbol)

    def _validate_nodes(self) -> None:
        if len(self.nodes) < 2:
            raise ValueError("Нужно задать минимум два узла интерполирования.")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Узлы должны быть различными.")
        for left, right in zip(self.nodes, self.nodes[1:]):
            if not bool(sp.N(right - left) > 0):
                raise ValueError("Узлы должны быть упорядочены по возрастанию.")
        step = sp.simplify(self.nodes[1] - self.nodes[0])
        if step == 0:
            raise ValueError("Шаг между узлами не может быть равен нулю.")
        for index in range(1, len(self.nodes) - 1):
            current_step = sp.simplify(self.nodes[index + 1] - self.nodes[index])
            if sp.simplify(current_step - step) != 0:
                raise ValueError("Для этой лабораторной узлы должны быть равноотстоящими.")

    def _build_finite_differences(self) -> tuple[tuple[sp.Expr, ...], ...]:
        columns: list[tuple[sp.Expr, ...]] = [self.y_values]
        while len(columns[-1]) > 1:
            previous = columns[-1]
            current = tuple(
                sp.simplify(previous[index + 1] - previous[index])
                for index in range(len(previous) - 1)
            )
            columns.append(current)
        return tuple(columns)

    @staticmethod
    def _forward_multiplier(parameter: sp.Expr, order: int) -> sp.Expr:
        multiplier = sp.Integer(1)
        for index in range(order):
            multiplier *= sp.simplify(parameter - index)
        return sp.simplify(multiplier)

    @staticmethod
    def _backward_multiplier(parameter: sp.Expr, order: int) -> sp.Expr:
        multiplier = sp.Integer(1)
        for index in range(order):
            multiplier *= sp.simplify(parameter + index)
        return sp.simplify(multiplier)

    def _build_interpolation_polynomial(self) -> sp.Expr:
        parameter = sp.simplify((self.x_symbol - self.nodes[0]) / self.step)
        polynomial = self.y_values[0]
        for order in range(1, len(self.nodes)):
            multiplier = self._forward_multiplier(parameter, order)
            polynomial += self.finite_differences[order][0] * multiplier / factorial(order)
        return sp.expand(sp.simplify(polynomial))

    @staticmethod
    def _difference_label(formula_kind: str, order: int) -> str:
        if order == 0:
            return "y0" if formula_kind == "forward" else "yn"
        prefix = "Delta" if formula_kind == "forward" else "Nabla"
        suffix = "y0" if formula_kind == "forward" else "yn"
        if order == 1:
            return f"{prefix} {suffix}"
        return f"{prefix}^{order} {suffix}"

    @staticmethod
    def _factor_text(formula_kind: str, parameter_symbol: str, order: int) -> str:
        if order == 0:
            return "1"
        factors = [parameter_symbol]
        for index in range(1, order):
            sign = "-" if formula_kind == "forward" else "+"
            factors.append(f"({parameter_symbol} {sign} {index})")
        numerator = " * ".join(factors)
        if order == 1:
            return numerator
        return f"{numerator} / {order}!"

    def _formula_template(self, formula_kind: str) -> str:
        base_label = "y0" if formula_kind == "forward" else "yn"
        parameter_symbol = "q" if formula_kind == "forward" else "t"
        parts = [base_label]
        for order in range(1, len(self.nodes)):
            factor_text = self._factor_text(formula_kind, parameter_symbol, order)
            diff_label = self._difference_label(formula_kind, order)
            parts.append(f"{factor_text} * {diff_label}")
        return " + ".join(parts)

    def _select_formula_kind(self, x_value: sp.Expr) -> tuple[str, str]:
        distance_to_start = abs(float(sp.N((x_value - self.nodes[0]) / self.step)))
        distance_to_end = abs(float(sp.N((x_value - self.nodes[-1]) / self.step)))
        if distance_to_start <= distance_to_end:
            if bool(x_value < self.nodes[0]):
                reason = "точка находится левее таблицы, поэтому берём формулу от начала."
            else:
                reason = "точка ближе к началу таблицы конечных разностей."
            return "forward", reason
        if bool(x_value > self.nodes[-1]):
            reason = "точка находится правее таблицы, поэтому берём формулу от конца."
        else:
            reason = "точка ближе к концу таблицы конечных разностей."
        return "backward", reason

    def solve_point(self, x_value: sp.Expr) -> PointInterpolationResult:
        formula_kind, reason = self._select_formula_kind(x_value)
        if formula_kind == "forward":
            formula_name = "Первая формула Ньютона (прямая)"
            formula_short_name = "Прямая Ньютона"
            parameter_symbol = "q"
            anchor_node = self.nodes[0]
            parameter_value = sp.simplify((x_value - anchor_node) / self.step)
            coefficient_index = 0
            multiplier_builder = self._forward_multiplier
        else:
            formula_name = "Вторая формула Ньютона (обратная)"
            formula_short_name = "Обратная Ньютона"
            parameter_symbol = "t"
            anchor_node = self.nodes[-1]
            parameter_value = sp.simplify((x_value - anchor_node) / self.step)
            coefficient_index = -1
            multiplier_builder = self._backward_multiplier

        terms: list[FormulaTerm] = [
            FormulaTerm(
                order=0,
                difference_label=self._difference_label(formula_kind, 0),
                factor_text="1",
                difference_value=self.y_values[coefficient_index],
                term_value=self.y_values[coefficient_index],
            )
        ]

        approximate_value = self.y_values[coefficient_index]
        for order in range(1, len(self.nodes)):
            multiplier = multiplier_builder(parameter_value, order)
            difference_value = self.finite_differences[order][coefficient_index]
            term_value = sp.simplify(
                difference_value * multiplier / factorial(order)
            )
            terms.append(
                FormulaTerm(
                    order=order,
                    difference_label=self._difference_label(formula_kind, order),
                    factor_text=self._factor_text(
                        formula_kind, parameter_symbol, order
                    ),
                    difference_value=difference_value,
                    term_value=term_value,
                )
            )
            approximate_value = sp.simplify(approximate_value + term_value)

        exact_value = sp.simplify(self.function_expr.subs(self.x_symbol, x_value))
        absolute_error = sp.simplify(abs(exact_value - approximate_value))
        return PointInterpolationResult(
            x_value=x_value,
            formula_kind=formula_kind,
            formula_name=formula_name,
            formula_short_name=formula_short_name,
            reason=reason,
            parameter_symbol=parameter_symbol,
            parameter_value=parameter_value,
            anchor_node=anchor_node,
            formula_template=self._formula_template(formula_kind),
            approximate_value=sp.simplify(approximate_value),
            exact_value=exact_value,
            absolute_error=absolute_error,
            terms=tuple(terms),
        )

    def solve_all_points(self) -> tuple[PointInterpolationResult, ...]:
        if self._point_results_cache is None:
            self._point_results_cache = tuple(
                self.solve_point(point) for point in self.evaluation_points
            )
        return self._point_results_cache

    def summary_rows(self, digits: int) -> list[tuple[str, str, str, str, str]]:
        rows: list[tuple[str, str, str, str, str]] = []
        for result in self.solve_all_points():
            rows.append(
                (
                    _trimmed_decimal(result.x_value, digits),
                    result.formula_short_name,
                    _trimmed_decimal(result.approximate_value, digits),
                    _trimmed_decimal(result.exact_value, digits),
                    _trimmed_decimal(result.absolute_error, digits),
                )
            )
        return rows

    def difference_table_view(self, digits: int) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        max_order = len(self.nodes) - 1
        headers = ["i", "x_i", "y_i"]
        for order in range(1, max_order + 1):
            if order == 1:
                headers.append("Delta y_i")
            else:
                headers.append(f"Delta^{order} y_i")

        rows: list[tuple[str, ...]] = []
        for row_index in range(len(self.nodes)):
            row_values = [
                str(row_index),
                _trimmed_decimal(self.nodes[row_index], digits),
                _trimmed_decimal(self.y_values[row_index], digits),
            ]
            for order in range(1, max_order + 1):
                if row_index < len(self.finite_differences[order]):
                    row_values.append(
                        _trimmed_decimal(self.finite_differences[order][row_index], digits)
                    )
                else:
                    row_values.append("")
            rows.append(tuple(row_values))
        return tuple(headers), rows

    def build_report(self, digits: int) -> str:
        lines: list[str] = [
            "Практическая работа №3. Интерполирование функции в случае равноотстоящих узлов",
            "",
            f"f(x) = {sp.sstr(self.function_expr)}",
            "Узлы интерполирования: " + ", ".join(describe_value(node, digits) for node in self.nodes),
            f"Шаг таблицы h = {describe_value(self.step, digits)}",
            "",
            "Значения функции в узлах:",
        ]
        for index, (node, value) in enumerate(zip(self.nodes, self.y_values)):
            lines.append(
                f"  y{index} = f({describe_value(node, digits)}) = {describe_value(value, digits)}"
            )

        max_order = len(self.nodes) - 1
        lines.extend(
            [
                "",
                f"Таблица конечных разностей построена до {max_order}-го порядка.",
                "Интерполяционный многочлен:",
                f"  P(x) = {sp.sstr(self.interpolation_polynomial)}",
            ]
        )

        for result in self.solve_all_points():
            point_text = describe_value(result.x_value, digits)
            lines.extend(
                [
                    "",
                    f"Точка x = {point_text}",
                    f"Выбор формулы: {result.formula_name}, потому что {result.reason}",
                    f"{result.parameter_symbol} = {describe_value(result.parameter_value, digits)}",
                    "Общий вид формулы:",
                    f"  {result.formula_template}",
                    "Подстановка членов:",
                ]
            )
            for term in result.terms:
                if term.order == 0:
                    lines.append(
                        f"  T0 = {term.difference_label} = {describe_value(term.term_value, digits)}"
                    )
                else:
                    lines.append(
                        f"  T{term.order} = {term.factor_text} * {term.difference_label} = {describe_value(term.term_value, digits)}"
                    )
            lines.extend(
                [
                    f"Приближённое значение P({point_text}) = {describe_value(result.approximate_value, digits)}",
                    f"Точное значение f({point_text}) = {describe_value(result.exact_value, digits)}",
                    f"Абсолютная погрешность |f - P| = {describe_value(result.absolute_error, digits)}",
                ]
            )

        return "\n".join(lines)


def build_default_solver() -> EqualNodeInterpolationSolver:
    return EqualNodeInterpolationSolver.from_raw_inputs(
        DEFAULT_FUNCTION,
        DEFAULT_NODES_TEXT,
        DEFAULT_POINTS_TEXT,
    )


if __name__ == "__main__":
    print(build_default_solver().build_report(digits=6))
