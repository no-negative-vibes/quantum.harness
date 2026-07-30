"""Exterior-power identities for the typed positivity search.

The generic compound-matrix convention is aligned with ZiboJin's
``oracle/exterior_cone.py`` on ``shared/work/zibo/representation-cones``.
This module freezes only the shared exterior algebra and chronological
product order; it does not copy that branch's candidate grammar.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import sympy as sp


def subset_basis(
    dimension: int,
    grade: int,
) -> tuple[tuple[int, ...], ...]:
    """Return lexicographically ordered subsets for one exterior grade."""

    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError("dimension must be an integer")
    if not isinstance(grade, int) or isinstance(grade, bool):
        raise TypeError("grade must be an integer")
    if dimension < 0 or not 0 <= grade <= dimension:
        raise ValueError("grade must satisfy 0 <= grade <= dimension")
    return tuple(combinations(range(dimension), grade))


def _sympy_square(matrix: sp.MatrixBase, *, name: str) -> sp.MatrixBase:
    if matrix.rows != matrix.cols:
        raise ValueError(f"{name} must be square")
    if any(entry.is_finite is False for entry in matrix):
        raise ValueError(f"{name} must be finite")
    return matrix


def _numpy_square(matrix: object, *, name: str) -> np.ndarray:
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric entries")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def compound_matrix(matrix: object, grade: int):
    """Return ``Lambda^grade(matrix)`` in lexicographic subset order."""

    if isinstance(matrix, sp.MatrixBase):
        exact = _sympy_square(matrix, name="matrix")
        basis = subset_basis(exact.rows, grade)
        return sp.ImmutableMatrix(
            len(basis),
            len(basis),
            lambda row, column: exact.extract(
                basis[row],
                basis[column],
            ).det(),
        )

    array = _numpy_square(matrix, name="matrix")
    basis = subset_basis(array.shape[0], grade)
    result = np.empty(
        (len(basis), len(basis)),
        dtype=np.result_type(array.dtype, np.float64),
    )
    for row_index, rows in enumerate(basis):
        for column_index, columns in enumerate(basis):
            if grade == 0:
                result[row_index, column_index] = 1.0
            else:
                result[row_index, column_index] = np.linalg.det(
                    array[np.ix_(rows, columns)]
                )
    return result


def chronological_product(factors: Sequence[object]):
    """Multiply earliest-to-latest factors as ``B_L ... B_2 B_1``."""

    if not factors:
        raise ValueError("at least one factor is required")

    first = factors[0]
    if isinstance(first, sp.MatrixBase):
        checked = tuple(
            _sympy_square(factor, name="factor")
            if isinstance(factor, sp.MatrixBase)
            else None
            for factor in factors
        )
        if any(factor is None for factor in checked):
            raise TypeError("all factors must use the same matrix backend")
        dimension = first.rows
        if any(factor.rows != dimension for factor in checked if factor is not None):
            raise ValueError("all factors must have the same square shape")
        product: sp.MatrixBase = sp.eye(dimension)
        for factor in checked:
            assert factor is not None
            product = factor * product
        return sp.ImmutableMatrix(product)

    checked_numpy = tuple(
        _numpy_square(factor, name="factor") for factor in factors
    )
    dimension = checked_numpy[0].shape[0]
    if any(factor.shape != (dimension, dimension) for factor in checked_numpy):
        raise ValueError("all factors must have the same square shape")
    dtype = np.result_type(*(factor.dtype for factor in checked_numpy))
    product = np.eye(dimension, dtype=dtype)
    for factor in checked_numpy:
        product = factor @ product
    return product


def exterior_traces(product: object) -> tuple[object, ...]:
    """Return ``tr(Lambda^k(product))`` for every exterior grade."""

    if isinstance(product, sp.MatrixBase):
        exact = _sympy_square(product, name="product")
        return tuple(
            compound_matrix(exact, grade).trace()
            for grade in range(exact.rows + 1)
        )

    array = _numpy_square(product, name="product")
    return tuple(
        np.trace(compound_matrix(array, grade))
        for grade in range(array.shape[0] + 1)
    )


def exterior_character_sum(product: object):
    """Evaluate ``sum_k tr(Lambda^k(product)) = det(I + product)``."""

    traces = exterior_traces(product)
    if isinstance(product, sp.MatrixBase):
        return sum(traces, sp.Integer(0))
    return sum(traces)


def exact_determinant_weight(factors: Sequence[object]):
    """Return ``det(I + B_L ... B_1)`` in the factors' native backend."""

    product = chronological_product(factors)
    if isinstance(product, sp.MatrixBase):
        return (sp.eye(product.rows) + product).det()
    return np.linalg.det(np.eye(product.shape[0], dtype=product.dtype) + product)


__all__ = [
    "chronological_product",
    "compound_matrix",
    "exact_determinant_weight",
    "exterior_character_sum",
    "exterior_traces",
    "subset_basis",
]
