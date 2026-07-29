from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from oracle.compound_cones import (
    chronological_product,
    compound_matrix,
    exact_determinant_weight,
    exterior_character_sum,
    exterior_traces,
    subset_basis,
)


def test_subset_basis_is_lexicographic() -> None:
    assert subset_basis(4, 2) == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )


def test_numpy_compound_matrix_uses_declared_minor_convention() -> None:
    matrix = np.array(
        [[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]]
    )

    compound = compound_matrix(matrix, 2)

    assert compound[1, 2] == pytest.approx(12.0)
    np.testing.assert_allclose(
        compound,
        np.array([[3.0, 4.0, 8.0], [-10.0, 6.0, 12.0], [-15.0, -20.0, 18.0]]),
    )


def test_sympy_compound_matrix_preserves_exact_entries() -> None:
    matrix = sp.ImmutableMatrix([[1, 2, 0], [0, 3, 4], [5, 0, 6]])

    compound = compound_matrix(matrix, 2)

    assert isinstance(compound, sp.MatrixBase)
    assert compound == sp.ImmutableMatrix(
        [[3, 4, 8], [-10, 6, 12], [-15, -20, 18]]
    )


def test_exterior_character_sum_reconstructs_numpy_determinant() -> None:
    product = np.array([[0.5, 1.0], [-0.25, 2.0]])

    traces = exterior_traces(product)

    assert len(traces) == 3
    assert exterior_character_sum(product) == pytest.approx(
        np.linalg.det(np.eye(2) + product)
    )


def test_exterior_character_sum_is_exact_for_sympy() -> None:
    product = sp.ImmutableMatrix(
        [[sp.Rational(1, 2), 1], [sp.Rational(-1, 4), 2]]
    )

    traces = exterior_traces(product)

    assert traces == (sp.Integer(1), sp.Rational(5, 2), sp.Rational(5, 4))
    assert exterior_character_sum(product) == (sp.eye(2) + product).det()


def test_chronological_product_left_multiplies_numpy_factors() -> None:
    first = np.array([[1, 1], [0, 1]])
    second = np.array([[1, 0], [2, 1]])

    product = chronological_product([first, second])

    np.testing.assert_array_equal(product, second @ first)
    assert not np.array_equal(product, first @ second)


def test_chronological_product_and_weight_preserve_sympy_exactness() -> None:
    first = sp.ImmutableMatrix([[1, 1], [0, 1]])
    second = sp.ImmutableMatrix([[1, 0], [2, 1]])

    product = chronological_product([first, second])
    weight = exact_determinant_weight([first, second])

    assert isinstance(product, sp.MatrixBase)
    assert product == second * first
    assert weight == sp.Integer(6)

