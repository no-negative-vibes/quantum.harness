"""Validate and compactly summarize a completed typed-exterior pilot run.

The report is deliberately a small machine-readable fixture.  It retains
aggregate accounting and the complete evidence for support-aware
signed-permutation reductions, but does not duplicate the many control
witnesses already stored in the per-cell manifests.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import exterior_category_pilot as pilot


REPORT_SCHEMA = "exterior-category-pilot-report-v1"
CLASSIFICATIONS = (
    "no_chart",
    "typed_only_calibration",
    "signed_permutation_tn_reduction",
    "provisional_front_door_survivor",
    "known_class_filter_incomplete",
    "untyped_not_separated",
    "ambiguous",
)
_RECORD_COUNT_FIELDS = {
    "determinant_checks": "determinant_checks",
    "exact_replays": "exact_replays",
    "ambiguous_checks": "ambiguous_checks",
}


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}: cannot read JSON object") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: expected a JSON object")
    return payload


def _run_directory(spec_path: Path, spec: Mapping[str, Any]) -> Path:
    """Mirror the pilot runner's run-directory resolution exactly."""

    declared = Path(spec.get("run_dir", spec_path.parent))
    return declared if declared.is_absolute() else spec_path.parent


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected an object")
    return value


def _sequence(value: object, *, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected a list")
    return value


def _nonnegative_integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}: expected a nonnegative integer")
    return value


def _exact_rational(value: object, *, context: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{context}: exact value must be an integer or string")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{context}: invalid exact rational {value!r}") from error


def _depth_and_word(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> tuple[int, list[Any]]:
    depth = payload.get("depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError(f"{context}: depth must be a positive integer")
    word = payload.get("word")
    if not isinstance(word, list) or len(word) != depth:
        raise ValueError(f"{context}: word length does not equal depth")
    return depth, word


def _validate_exact_negative_witness(
    witness: object,
    *,
    context: str,
) -> tuple[Mapping[str, Any], int]:
    payload = _mapping(witness, context=context)
    depth, _ = _depth_and_word(payload, context=context)
    weight = _exact_rational(
        payload.get("exact_weight"),
        context=f"{context}: exact negative witness",
    )
    if weight >= 0:
        raise ValueError(
            f"{context}: exact negative witness has nonnegative weight"
        )
    return payload, depth


def _validate_legal_closed_witness(
    witness: object,
    *,
    context: str,
) -> None:
    payload = _mapping(witness, context=context)
    word = payload.get("word")
    if not isinstance(word, list) or not word:
        raise ValueError(f"{context}: legal closed word must be nonempty")
    weight = _exact_rational(
        payload.get("exact_weight"),
        context=f"{context}: exact weight",
    )
    traces = _sequence(
        payload.get("grade_traces"),
        context=f"{context}: grade traces",
    )
    if weight < 0 or any(
        _exact_rational(value, context=f"{context}: grade trace") < 0
        for value in traces
    ):
        raise ValueError(f"{context}: legal closed evidence is negative")


def _validate_legal_stress(
    stresses: object,
    *,
    configured_depths: set[int],
    context: str,
) -> int:
    payloads = _sequence(stresses, context=context)
    for index, stress in enumerate(payloads):
        stress_context = f"{context}[{index}]"
        payload = _mapping(stress, context=stress_context)
        depth, _ = _depth_and_word(payload, context=stress_context)
        if configured_depths and depth not in configured_depths:
            raise ValueError(
                f"{stress_context}: depth is not configured for legal stress"
            )
        weight = _exact_rational(
            payload.get("exact_weight"),
            context=f"{stress_context}: exact weight",
        )
        if weight < 0:
            raise ValueError(
                f"{stress_context}: legal stress nonnegative check failed"
            )
    return len(payloads)


def _validate_provisional_gates(
    record: Mapping[str, Any],
    *,
    configured_stress_depths: set[int],
    context: str,
) -> None:
    failures: list[str] = []
    if record.get("has_exact_grade_chart") is not True:
        failures.append("no exact grade chart")
    if record.get("legal_closed_witness") is None:
        failures.append("no legal closed witness")
    if not record.get("exact_witnesses"):
        failures.append("no exact type-erasure witness")
    if record.get("grammar") == "coboundary_tn_control":
        failures.append("control grammar")
    if record.get("induced_by_diagonal_sign_charts") is not False:
        failures.append("diagonal-sign induced")
    if record.get("signed_permutation_filter_complete") is not True:
        failures.append("signed-permutation filter incomplete")
    if record.get("signed_permutation_tn_witness") is not None:
        failures.append("known signed-permutation reduction")
    if record.get("novelty_status") != "provisional_after_front_door_filters":
        failures.append("wrong novelty status")
    if record.get("physical_strong_candidate") is not False:
        failures.append("incorrectly promoted to physical-strong")
    if configured_stress_depths and not record.get("legal_stress"):
        failures.append("missing configured legal stress")
    if failures:
        raise ValueError(
            f"{context}: provisional gate consistency failed: "
            + ", ".join(failures)
        )


def _validate_signed_permutation_reduction(
    record: Mapping[str, Any],
    *,
    context: str,
) -> None:
    failures: list[str] = []
    if record.get("has_exact_grade_chart") is not True:
        failures.append("no exact grade chart")
    if record.get("legal_closed_witness") is None:
        failures.append("no legal closed witness")
    if not record.get("exact_witnesses"):
        failures.append("no exact type-erasure witness")
    if record.get("induced_by_diagonal_sign_charts") is not False:
        failures.append("diagonal-sign induced")
    if record.get("signed_permutation_filter_complete") is not True:
        failures.append("filter not complete through a hit")
    if not isinstance(record.get("signed_permutation_tn_witness"), Mapping):
        failures.append("missing reduction witness")
    if (
        record.get("novelty_status")
        != "known_signed_permutation_tn_coboundary"
    ):
        failures.append("wrong novelty status")
    if failures:
        raise ValueError(
            f"{context}: invalid signed-permutation reduction: "
            + ", ".join(failures)
        )


def _support_reduction_fixture(
    *,
    cell_id: str,
    dimension: int,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "grammar": record["grammar"],
        "dimension": dimension,
        "candidate_id": record["candidate_id"],
        "candidate_seed": record["candidate_seed"],
        "definition": deepcopy(record["definition"]),
        "evidence": {
            "has_exact_grade_chart": record["has_exact_grade_chart"],
            "induced_by_diagonal_sign_charts": record[
                "induced_by_diagonal_sign_charts"
            ],
            "legal_closed_witness": deepcopy(
                record["legal_closed_witness"]
            ),
            "legal_stress": deepcopy(record["legal_stress"]),
            "exact_negative_witnesses": deepcopy(
                record["exact_witnesses"]
            ),
            "minimum_exact_negative_word_depth": record[
                "minimum_exact_negative_word_depth"
            ],
            "signed_permutation_filter_complete": record[
                "signed_permutation_filter_complete"
            ],
            "signed_permutation_tn_witness": deepcopy(
                record["signed_permutation_tn_witness"]
            ),
            "novelty_status": record["novelty_status"],
            "primitive_real_log_audit": deepcopy(
                record["primitive_real_log_audit"]
            ),
            "physical_strong_candidate": record[
                "physical_strong_candidate"
            ],
        },
    }


def _histogram_payload(histogram: Counter[int]) -> dict[str, int]:
    return {
        str(depth): histogram[depth]
        for depth in sorted(histogram)
    }


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    cell_id: str,
    grammar: str,
    dimension: int,
    configured_stress_depths: set[int],
    seen_candidate_ids: set[str],
) -> tuple[Counter[int], list[dict[str, Any]]]:
    context = cell_id
    counts = _mapping(manifest.get("counts"), context=f"{context}: counts")
    for field, value in counts.items():
        _nonnegative_integer(value, context=f"{context}: counts.{field}")
    records = _sequence(
        manifest.get("candidate_records"),
        context=f"{context}: candidate_records",
    )
    candidate_count = _nonnegative_integer(
        counts.get("candidates"),
        context=f"{context}: counts.candidates",
    )
    classifications: Counter[str] = Counter()
    record_totals: Counter[str] = Counter()
    flattened_witnesses: list[dict[str, Any]] = []
    witness_depths: Counter[int] = Counter()
    retained: list[dict[str, Any]] = []

    for index, candidate in enumerate(records):
        record_context = f"{context}: candidate_records[{index}]"
        record = _mapping(candidate, context=record_context)
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"{record_context}: invalid candidate_id")
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"{record_context}: duplicate candidate_id")
        seen_candidate_ids.add(candidate_id)
        candidate_seed = record.get("candidate_seed")
        if isinstance(candidate_seed, bool) or not isinstance(
            candidate_seed,
            int,
        ):
            raise ValueError(f"{record_context}: invalid candidate_seed")
        if record.get("grammar") != grammar:
            raise ValueError(f"{record_context}: grammar does not match cell")
        definition = _mapping(
            record.get("definition"),
            context=f"{record_context}: definition",
        )
        if definition.get("grammar") != grammar:
            raise ValueError(
                f"{record_context}: definition grammar does not match cell"
            )
        if int(definition.get("dimension", dimension)) != dimension:
            raise ValueError(
                f"{record_context}: definition dimension does not match cell"
            )

        classification = record.get("classification")
        if classification not in CLASSIFICATIONS:
            raise ValueError(
                f"{record_context}: unknown classification {classification!r}"
            )
        classifications[str(classification)] += 1

        for count_field, record_field in _RECORD_COUNT_FIELDS.items():
            record_totals[count_field] += _nonnegative_integer(
                record.get(record_field),
                context=f"{record_context}: {record_field}",
            )
        record_totals["physical_strong_candidate"] += int(
            record.get("physical_strong_candidate") is True
        )
        record_totals["legal_stress_word_checks"] += _validate_legal_stress(
            record.get("legal_stress"),
            configured_depths=configured_stress_depths,
            context=f"{record_context}: legal_stress",
        )

        legal_closed = record.get("legal_closed_witness")
        if legal_closed is not None:
            _validate_legal_closed_witness(
                legal_closed,
                context=f"{record_context}: legal_closed_witness",
            )

        record_witnesses = _sequence(
            record.get("exact_witnesses"),
            context=f"{record_context}: exact_witnesses",
        )
        depths: list[int] = []
        for witness_index, witness in enumerate(record_witnesses):
            witness_payload, depth = _validate_exact_negative_witness(
                witness,
                context=(
                    f"{record_context}: exact negative witness "
                    f"{witness_index}"
                ),
            )
            depths.append(depth)
            witness_depths[depth] += 1
            flattened_witnesses.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_seed": candidate_seed,
                    **dict(witness_payload),
                }
            )
        record_totals["exact_witnesses"] += len(record_witnesses)
        expected_minimum = min(depths) if depths else None
        if record.get("minimum_exact_negative_word_depth") != expected_minimum:
            raise ValueError(
                f"{record_context}: exact negative witness minimum depth "
                "is inconsistent"
            )

        requires_negative_witness = classification in {
            "typed_only_calibration",
            "signed_permutation_tn_reduction",
            "provisional_front_door_survivor",
            "known_class_filter_incomplete",
        }
        if requires_negative_witness and not record_witnesses:
            raise ValueError(
                f"{record_context}: classification lacks an exact negative "
                "witness"
            )
        if classification in {
            "no_chart",
            "untyped_not_separated",
            "ambiguous",
        } and record_witnesses:
            raise ValueError(
                f"{record_context}: classification contradicts exact "
                "negative witnesses"
            )

        if classification == "provisional_front_door_survivor":
            _validate_provisional_gates(
                record,
                configured_stress_depths=configured_stress_depths,
                context=record_context,
            )
        if (
            record.get("novelty_status")
            == "provisional_after_front_door_filters"
            and classification != "provisional_front_door_survivor"
        ):
            raise ValueError(
                f"{record_context}: provisional gate status is misclassified"
            )
        if classification == "signed_permutation_tn_reduction":
            _validate_signed_permutation_reduction(
                record,
                context=record_context,
            )
            if grammar == "support_aware_sparse":
                retained.append(
                    _support_reduction_fixture(
                        cell_id=cell_id,
                        dimension=dimension,
                        record=record,
                    )
                )

    conservation_failures: list[str] = []
    if len(records) != candidate_count:
        conservation_failures.append(
            f"records={len(records)} candidates={candidate_count}"
        )
    classified_count = sum(
        _nonnegative_integer(
            counts.get(classification),
            context=f"{context}: counts.{classification}",
        )
        for classification in CLASSIFICATIONS
    )
    if classified_count != candidate_count:
        conservation_failures.append(
            f"classified={classified_count} candidates={candidate_count}"
        )
    for classification in CLASSIFICATIONS:
        declared = int(counts[classification])
        actual = classifications[classification]
        if declared != actual:
            conservation_failures.append(
                f"{classification}: declared={declared} actual={actual}"
            )
    if conservation_failures:
        raise ValueError(
            f"{context}: classification conservation failed: "
            + "; ".join(conservation_failures)
        )

    for count_field, actual in record_totals.items():
        declared = _nonnegative_integer(
            counts.get(count_field),
            context=f"{context}: counts.{count_field}",
        )
        if declared != actual:
            raise ValueError(
                f"{context}: record accounting mismatch for {count_field}: "
                f"declared={declared} actual={actual}"
            )

    manifest_witnesses = _sequence(
        manifest.get("exact_witnesses"),
        context=f"{context}: exact_witnesses",
    )
    for index, witness in enumerate(manifest_witnesses):
        _validate_exact_negative_witness(
            witness,
            context=f"{context}: manifest exact negative witness {index}",
        )
    if Counter(map(_canonical_json, manifest_witnesses)) != Counter(
        map(_canonical_json, flattened_witnesses)
    ):
        raise ValueError(
            f"{context}: manifest exact negative witness index is inconsistent"
        )

    return witness_depths, retained


def build_report(run_spec: str | Path) -> dict[str, Any]:
    """Verify a completed run and return a compact deterministic summary."""

    spec_path = Path(run_spec)
    structural = pilot.verify_run(spec_path)
    spec = _read_json_object(spec_path, label="run spec")
    run_dir = _run_directory(spec_path, spec)
    shared_settings = _mapping(
        spec.get("settings", {}),
        context="run spec settings",
    )
    shared_provenance = _mapping(
        spec.get("provenance", {}),
        context="run spec provenance",
    )
    configured_stress_depths = {
        int(depth)
        for depth in _sequence(
            shared_settings.get("legal_stress_depths", []),
            context="run spec legal_stress_depths",
        )
    }

    total_counts: Counter[str] = Counter()
    total_witness_depths: Counter[int] = Counter()
    group_counts: dict[tuple[str, int], Counter[str]] = {}
    group_cells: Counter[tuple[str, int]] = Counter()
    group_witness_depths: dict[tuple[str, int], Counter[int]] = {}
    retained_reductions: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()

    cells = _sequence(spec.get("cells"), context="run spec cells")
    for cell_index, cell_payload in enumerate(cells):
        cell = _mapping(
            cell_payload,
            context=f"run spec cells[{cell_index}]",
        )
        cell_id = str(cell.get("cell_id"))
        params = _mapping(
            cell.get("params"),
            context=f"{cell_id}: params",
        )
        grammar = str(params.get("grammar"))
        dimension_value = params.get("dimension")
        if isinstance(dimension_value, bool) or not isinstance(
            dimension_value,
            int,
        ):
            raise ValueError(f"{cell_id}: invalid dimension")
        dimension = dimension_value
        manifest = _read_json_object(
            run_dir / "cells" / cell_id / "manifest.json",
            label=f"{cell_id} manifest",
        )
        depths, reductions = _validate_manifest(
            manifest,
            cell_id=cell_id,
            grammar=grammar,
            dimension=dimension,
            configured_stress_depths=configured_stress_depths,
            seen_candidate_ids=seen_candidate_ids,
        )
        counts = _mapping(
            manifest["counts"],
            context=f"{cell_id}: counts",
        )
        key = (grammar, dimension)
        group_cells[key] += 1
        group_count = group_counts.setdefault(key, Counter())
        group_depths = group_witness_depths.setdefault(key, Counter())
        for field, value in counts.items():
            integer = _nonnegative_integer(
                value,
                context=f"{cell_id}: counts.{field}",
            )
            total_counts[field] += integer
            group_count[field] += integer
        total_witness_depths.update(depths)
        group_depths.update(depths)
        retained_reductions.extend(reductions)

    verified_counts = _mapping(
        structural.get("aggregate_counts"),
        context="verify_run aggregate_counts",
    )
    if dict(total_counts) != dict(verified_counts):
        raise ValueError(
            "aggregate accounting differs from verify_run output"
        )
    verified_cells = _nonnegative_integer(
        structural.get("verified_cells"),
        context="verify_run verified_cells",
    )
    if verified_cells != len(cells):
        raise ValueError(
            "verified cell count differs from the run specification"
        )

    grouped = [
        {
            "grammar": grammar,
            "dimension": dimension,
            "cells": group_cells[(grammar, dimension)],
            "counts": dict(sorted(group_counts[(grammar, dimension)].items())),
            "exact_negative_witness_depths": _histogram_payload(
                group_witness_depths[(grammar, dimension)]
            ),
        }
        for grammar, dimension in sorted(group_counts)
    ]
    retained_reductions.sort(
        key=lambda item: (
            item["dimension"],
            item["cell_id"],
            item["candidate_id"],
        )
    )
    return {
        "schema_version": 1,
        "report_schema": REPORT_SCHEMA,
        "run": {
            "run_id": spec.get("run_id"),
            "verified_cells": verified_cells,
            "spec_digest": structural.get("spec_digest"),
            "settings": deepcopy(dict(shared_settings)),
            "provenance": deepcopy(dict(shared_provenance)),
        },
        "validation": {
            "classification_conservation": True,
            "exact_negative_witnesses": True,
            "legal_stress_nonnegative": True,
            "provisional_gate_consistency": True,
        },
        "totals": dict(sorted(total_counts.items())),
        "exact_negative_witness_depths": _histogram_payload(
            total_witness_depths
        ),
        "by_grammar_dimension": grouped,
        "support_aware_signed_permutation_reductions": retained_reductions,
    }


def write_report(run_spec: str | Path, output: str | Path) -> dict[str, Any]:
    """Validate ``run_spec`` and atomically write its compact JSON report."""

    summary = build_report(run_spec)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_spec", help="completed pilot run_spec.json")
    parser.add_argument("output", help="compact JSON report path")
    arguments = parser.parse_args()
    summary = write_report(arguments.run_spec, arguments.output)
    print(
        json.dumps(
            {
                "output": str(Path(arguments.output)),
                "verified_cells": summary["run"]["verified_cells"],
                "support_aware_signed_permutation_reductions": len(
                    summary[
                        "support_aware_signed_permutation_reductions"
                    ]
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


__all__ = [
    "CLASSIFICATIONS",
    "REPORT_SCHEMA",
    "build_report",
    "main",
    "write_report",
]


if __name__ == "__main__":
    main()
