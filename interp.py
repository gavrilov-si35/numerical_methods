from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import sympy as sp
import numpy as np
import math

@dataclass
class LagrangeInterpolator:
    function_expression: str
    nodes: List[float]
    x_symbol: sp.Symbol = sp.symbols('x')

    def __post_init__(self):
        self.f_symbol: sp.Expr = sp.sympify(self.function_expression)
        if len(self.nodes) < 1:
            raise ValueError("Нужен хотя бы один узел интерполирования.")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Узлы должны быть различными.")
        self.degree = len(self.nodes) - 1
        self._basis_cache = None
        self._polynomial_cache = None

    def basis_Lk(self) -> List[sp.Expr]:
        if self._basis_cache is not None:
            return self._basis_cache
        x_sym = self.x_symbol
        nodes_rational = [sp.Rational(str(value)) for value in self.nodes]
        basis_list = []
        for index_k, node_k in enumerate(nodes_rational):
            if len(nodes_rational) == 1:
                basis_poly = sp.Integer(1)
            else:
                numerator = sp.prod([(x_sym - node_j) for j, node_j in enumerate(nodes_rational) if j != index_k])
                denominator = sp.prod([(node_k - node_j) for j, node_j in enumerate(nodes_rational) if j != index_k])
                basis_poly = sp.simplify(numerator / denominator)
            basis_list.append(sp.simplify(basis_poly))
        self._basis_cache = basis_list
        return basis_list

    def polynomial(self, force: bool = False) -> sp.Expr:
        if self._polynomial_cache is not None and not force:
            return self._polynomial_cache
        x_sym = self.x_symbol
        basis_list = self.basis_Lk()
        function_values_at_nodes = [sp.simplify(self.f_symbol.subs(x_sym, sp.Rational(str(node_value)))) for node_value in self.nodes]
        polynomial_expr = sum(function_values_at_nodes[k] * basis_list[k] for k in range(len(self.nodes)))
        polynomial_expr = sp.simplify(sp.expand(polynomial_expr))
        self._polynomial_cache = polynomial_expr
        return polynomial_expr

    def evaluate_polynomial(self, x_point: float) -> float:
        polynomial_expr = self.polynomial()
        polynomial_func = sp.lambdify(self.x_symbol, polynomial_expr, 'numpy')
        return float(polynomial_func(x_point))

    def evaluate_function(self, x_point: float) -> float:
        function_func = sp.lambdify(self.x_symbol, self.f_symbol, 'numpy')
        return float(function_func(x_point))

    def real_error(self, x_point: float) -> float:
        return abs(self.evaluate_function(x_point) - self.evaluate_polynomial(x_point))

    def remainder_derivative(self) -> sp.Expr:
        order = self.degree + 1
        return sp.simplify(sp.diff(self.f_symbol, self.x_symbol, order))

    def estimate_M(self, interval_a: float, interval_b: float, samples: int = 2001) -> float:
        derivative_expr = self.remainder_derivative()
        derivative_func = sp.lambdify(self.x_symbol, derivative_expr, 'numpy')
        sample_points = np.linspace(interval_a, interval_b, samples)
        max_value = 0.0
        for sample_point in sample_points:
            try:
                value = float(abs(derivative_func(sample_point)))
                if np.isfinite(value) and value > max_value:
                    max_value = value
            except Exception:
                continue
        return max_value

    def remainder_bound(self, x_point: float, M: Optional[float] = None, samples_for_M: int = 2001) -> float:
        if M is None:
            interval_a = min(min(self.nodes), x_point)
            interval_b = max(max(self.nodes), x_point)
            M = self.estimate_M(interval_a, interval_b, samples_for_M)
        product_term = 1.0
        for node_value in self.nodes:
            product_term *= (x_point - node_value)
        return (M * abs(product_term)) / math.factorial(self.degree + 1)

@dataclass
class NewtonInterpolator:
    function_expression: str
    nodes: List[float]
    x_symbol: sp.Symbol = sp.symbols('x')

    def __post_init__(self):
        self.f_symbol: sp.Expr = sp.sympify(self.function_expression)
        if len(self.nodes) < 1:
            raise ValueError("Нужен хотя бы один узел интерполирования.")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Узлы должны быть различными.")
        self.nodes_rational = [sp.Rational(str(value)) for value in self.nodes]
        self.degree = len(self.nodes) - 1
        self._divided_diff_table = None
        self._polynomial_cache = None

    def divided_differences(self) -> List[List[sp.Expr]]:
        if self._divided_diff_table is not None:
            return self._divided_diff_table
        x_sym = self.x_symbol
        table = []
        first_column = [sp.simplify(self.f_symbol.subs(x_sym, xi)) for xi in self.nodes_rational]
        table.append(first_column)
        for order in range(1, self.degree + 1):
            previous_column = table[order - 1]
            current_column = []
            for i_index in range(len(previous_column) - 1):
                numerator = sp.simplify(previous_column[i_index + 1] - previous_column[i_index])
                denominator = sp.simplify(self.nodes_rational[i_index + order] - self.nodes_rational[i_index])
                current_column.append(sp.simplify(numerator / denominator))
            table.append(current_column)
        self._divided_diff_table = table
        return table

    def coefficients(self) -> List[sp.Expr]:
        table = self.divided_differences()
        return [table[k][0] for k in range(len(table))]

    def polynomial(self, force: bool = False) -> sp.Expr:
        if self._polynomial_cache is not None and not force:
            return self._polynomial_cache
        x_sym = self.x_symbol
        coeffs = self.coefficients()
        polynomial_expr = sp.Integer(0)
        for k_index, coeff in enumerate(coeffs):
            if k_index == 0:
                polynomial_expr = polynomial_expr + coeff
            else:
                product_factor = sp.prod([(x_sym - self.nodes_rational[j_index]) for j_index in range(k_index)])
                polynomial_expr = sp.simplify(polynomial_expr + coeff * product_factor)
        polynomial_expr = sp.simplify(sp.expand(polynomial_expr))
        self._polynomial_cache = polynomial_expr
        return polynomial_expr

    def evaluate_polynomial(self, x_point: float) -> float:
        polynomial_expr = self.polynomial()
        polynomial_func = sp.lambdify(self.x_symbol, polynomial_expr, 'numpy')
        return float(polynomial_func(x_point))

    def evaluate_function(self, x_point: float) -> float:
        function_func = sp.lambdify(self.x_symbol, self.f_symbol, 'numpy')
        return float(function_func(x_point))