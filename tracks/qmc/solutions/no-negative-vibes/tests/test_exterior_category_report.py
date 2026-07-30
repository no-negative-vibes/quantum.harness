from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

import oracle.exterior_category_pilot as pilot
import oracle.exterior_category_report as report


CLASSIFICATIONS = (
    "no_chart",
    "typed_only_calibration",
    "signed_permutation_tn_reduction",
    "provisional_front_door_survivor",
    "known_class_filter_incomplete",
    "untyped_not_separated",
    "ambiguous",
)


def _negative_witness(
    *,
    depth: int,
    exact_weight: str,
) -> dict[str, object]:
    return {
        "depth": depth,
        "word": [f"edge-{index}" for index in range(depth)],
        "exact_weight": exact_weight,
        "float_weight": float(exact_weight),
        "product_order": "B_last ... B_first",
        "primitive_real_log_impossible_by_negative_singleton": depth == 1,
    }


def _record(
    *,
    grammar: str,
    candidate_id: str,
    classification: str,
    depth: int,
    definition_marker: str,
) -> dict[str, object]:
    witness = _negative_witness(depth=depth, exact_weight=f"-{depth + 1}")
    signed_reduction = classification == "signed_permutation_tn_reduction"
    return {
        "candidate_index": 0,
        "candidate_seed": depth * 101,
        "candidate_id": candidate_id,
        "grammar": grammar,
        "definition": {
            "grammar": grammar,
            "dimension": 4,
            "marker": definition_marker,
            "edges": [],
        },
        "float_chart_screen_passed": True,
        "has_exact_grade_chart": True,
        "chart_screen": "exact_pass",
        "induced_by_diagonal_sign_charts": grammar == "coboundary_tn_control",
        "legal_closed_witness": {
            "word": ["forward-1", "return"],
            "exact_weight": "7",
            "grade_traces": ["1", "2", "3", "4", "5"],
        },
        "determinant_checks": 11,
        "exact_replays": 2,
        "ambiguous_checks": 0,
        "exact_witnesses": [witness],
        "minimum_exact_negative_word_depth": depth,
        "primitive_real_log_audit": {
            "all_edges_pass": False,
            "status": "not_run",
            "reason": "fixture",
        },
        "physical_strong_candidate": False,
        "signed_permutation_filter_complete": True if signed_reduction else None,
        "signed_permutation_tn_witness": (
            {
                "object_permutations": {
                    "a": [0, 1, 2, 3],
                    "b": [1, 0, 3, 2],
                },
                "tested_assignments": 17,
            }
            if signed_reduction
            else None
        ),
        "novelty_status": (
            "known_signed_permutation_tn_coboundary"
            if signed_reduction
            else "known_one_particle_coboundary_control"
        ),
        "legal_stress": [
            {
                "depth": 2,
                "strategy": "alternating_first",
                "word": ["return", "forward-1"],
                "exact_weight": "13",
            }
        ],
        "classification": classification,
    }


def _counts(record: dict[str, object]) -> dict[str, int]:
    counts = {
        "candidates": 1,
        "certificate_stage_checks": 2,
        "edge_grade_checks": 15,
        "determinant_checks": int(record["determinant_checks"]),
        "exact_replays": int(record["exact_replays"]),
        "exact_witnesses": len(record["exact_witnesses"]),
        "ambiguous_checks": int(record["ambiguous_checks"]),
        "physical_strong_candidate": int(record["physical_strong_candidate"]),
        "legal_stress_word_checks": len(record["legal_stress"]),
    }
    counts.update({classification: 0 for classification in CLASSIFICATIONS})
    counts[str(record["classification"])] = 1
    return counts


def _manifest(
    *,
    cell_id: str,
    grammar: str,
    record: dict[str, object],
) -> dict[str, object]:
    exact_witnesses = [
        {
            "candidate_id": record["candidate_id"],
            "candidate_seed": record["candidate_seed"],
            **witness,
        }
        for witness in record["exact_witnesses"]
    ]
    return {
        "schema_version": 1,
        "manifest_schema": "exterior-category-pilot-cell-v1",
        "completed": True,
        "compute_success": True,
        "cell_id": cell_id,
        "params": {"grammar": grammar, "dimension": 4, "seed": 9},
        "settings": {
            "candidates_per_cell": 1,
            "legal_stress_depths": [2],
        },
        "provenance": {
            "protocol": "fixture-protocol",
            "manifest_schema": "exterior-category-pilot-cell-v1",
        },
        "spec_digest": "fixture-digest",
        "runtime_seconds": 0.1,
        "counts": _counts(record),
        "candidate_records": [record],
        "exact_witnesses": exact_witnesses,
    }


def _write_fixture_run(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    settings = {
        "candidates_per_cell": 1,
        "legal_stress_depths": [2],
    }
    provenance = {
        "protocol": "fixture-protocol",
        "manifest_schema": "exterior-category-pilot-cell-v1",
    }
    cells = [
        {
            "cell_id": "cell-control",
            "params": {
                "grammar": "coboundary_tn_control",
                "dimension": 4,
                "seed": 9,
            },
        },
        {
            "cell_id": "cell-support",
            "params": {
                "grammar": "support_aware_sparse",
                "dimension": 4,
                "seed": 9,
            },
        },
    ]
    spec = {
        "run_id": "fixture-run",
        "settings": settings,
        "provenance": provenance,
        "spec_digest": "fixture-digest",
        "cells": cells,
    }
    spec_path = run_dir / "run_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    records = {
        "cell-control": _record(
            grammar="coboundary_tn_control",
            candidate_id="control-id",
            classification="typed_only_calibration",
            depth=1,
            definition_marker="control-secret",
        ),
        "cell-support": _record(
            grammar="support_aware_sparse",
            candidate_id="support-id",
            classification="signed_permutation_tn_reduction",
            depth=3,
            definition_marker="support-kept",
        ),
    }
    paths: dict[str, Path] = {}
    for cell in cells:
        cell_id = str(cell["cell_id"])
        grammar = str(cell["params"]["grammar"])
        manifest_path = run_dir / "cells" / cell_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps(
                _manifest(
                    cell_id=cell_id,
                    grammar=grammar,
                    record=records[cell_id],
                )
            ),
            encoding="utf-8",
        )
        paths[cell_id] = manifest_path
    return spec_path, paths


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_verifies_and_summarizes_without_copying_control_witnesses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, _ = _write_fixture_run(tmp_path)
    calls: list[Path] = []
    original_verify = pilot.verify_run

    def checked_verify(path: str | Path) -> dict[str, object]:
        calls.append(Path(path))
        return original_verify(path)

    monkeypatch.setattr(report.pilot, "verify_run", checked_verify)

    summary = report.build_report(spec_path)

    assert calls == [spec_path]
    assert summary["run"]["verified_cells"] == 2
    assert summary["validation"] == {
        "classification_conservation": True,
        "exact_negative_witnesses": True,
        "legal_stress_nonnegative": True,
        "provisional_gate_consistency": True,
    }
    assert summary["exact_negative_witness_depths"] == {"1": 1, "3": 1}
    groups = {
        (row["grammar"], row["dimension"]): row
        for row in summary["by_grammar_dimension"]
    }
    assert groups[("coboundary_tn_control", 4)]["counts"][
        "typed_only_calibration"
    ] == 1
    assert groups[("support_aware_sparse", 4)][
        "exact_negative_witness_depths"
    ] == {"3": 1}

    retained = summary["support_aware_signed_permutation_reductions"]
    assert len(retained) == 1
    assert retained[0]["candidate_id"] == "support-id"
    assert retained[0]["definition"]["marker"] == "support-kept"
    assert retained[0]["evidence"]["exact_negative_witnesses"][0][
        "exact_weight"
    ] == "-4"
    assert "control-secret" not in json.dumps(summary, sort_keys=True)


def test_write_report_and_cli_accept_run_spec_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path, _ = _write_fixture_run(tmp_path)
    output_path = tmp_path / "nested" / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "exterior_category_report",
            str(spec_path),
            str(output_path),
        ],
    )

    report.main()

    written = _read_json(output_path)
    assert written["run"]["run_id"] == "fixture-run"
    assert json.loads(capsys.readouterr().out) == {
        "output": str(output_path),
        "support_aware_signed_permutation_reductions": 1,
        "verified_cells": 2,
    }


def test_report_rejects_a_structurally_incomplete_run(tmp_path: Path) -> None:
    spec_path, paths = _write_fixture_run(tmp_path)
    paths["cell-support"].unlink()

    with pytest.raises(RuntimeError, match="run verification failed"):
        report.build_report(spec_path)


def test_report_rejects_broken_classification_conservation(
    tmp_path: Path,
) -> None:
    spec_path, paths = _write_fixture_run(tmp_path)
    manifest = _read_json(paths["cell-control"])
    manifest["counts"]["typed_only_calibration"] = 0
    _write_json(paths["cell-control"], manifest)

    with pytest.raises(ValueError, match="classification conservation"):
        report.build_report(spec_path)


def test_report_rejects_a_nonnegative_exact_negative_witness(
    tmp_path: Path,
) -> None:
    spec_path, paths = _write_fixture_run(tmp_path)
    manifest = _read_json(paths["cell-control"])
    manifest["candidate_records"][0]["exact_witnesses"][0][
        "exact_weight"
    ] = "1"
    manifest["exact_witnesses"][0]["exact_weight"] = "1"
    _write_json(paths["cell-control"], manifest)

    with pytest.raises(ValueError, match="exact negative witness"):
        report.build_report(spec_path)


def test_report_rejects_a_negative_legal_stress_weight(
    tmp_path: Path,
) -> None:
    spec_path, paths = _write_fixture_run(tmp_path)
    manifest = _read_json(paths["cell-control"])
    manifest["candidate_records"][0]["legal_stress"][0][
        "exact_weight"
    ] = "-1"
    _write_json(paths["cell-control"], manifest)

    with pytest.raises(ValueError, match="legal stress nonnegative"):
        report.build_report(spec_path)


def test_report_rejects_a_provisional_record_that_skips_a_gate(
    tmp_path: Path,
) -> None:
    spec_path, paths = _write_fixture_run(tmp_path)
    manifest = _read_json(paths["cell-support"])
    record = manifest["candidate_records"][0]
    record["classification"] = "provisional_front_door_survivor"
    record["novelty_status"] = "provisional_after_front_door_filters"
    manifest["counts"]["signed_permutation_tn_reduction"] = 0
    manifest["counts"]["provisional_front_door_survivor"] = 1
    _write_json(paths["cell-support"], manifest)

    with pytest.raises(ValueError, match="provisional gate"):
        report.build_report(spec_path)
