# Typed Exterior Positivity Pilot Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `subagent-driven-development` to implement this plan task-by-task.

**Goal:** Build and run a reproducible 50,000–100,000-check pilot that searches for finite typed matrix alphabets whose legal closed words have certified nonnegative fermion determinant weights, while forgetting the types exposes an exact negative word.

**Architecture:** Separate mathematical identities (`compound_cones.py`) from typed-graph certificates (`exterior_category_search.py`) and the resumable search runner (`exterior_category_pilot.py`). The fast path uses float64 only to screen small integer/rational matrices; every reported negative witness is replayed with SymPy exact arithmetic. A per-grade diagonal-sign chart solver supplies the first nontrivial support-aware simplicial charts, while an induced one-particle coboundary family is retained only as a positive calibration.

**Tech Stack:** Python 3, NumPy, SymPy, pytest, the repository's generic `scripts/parameter_scan.py`.

---

### Task 1: Freeze the exterior identities and product convention

**Files:**
- Create: `tracks/qmc/solutions/no-negative-vibes/oracle/compound_cones.py`
- Create: `tracks/qmc/solutions/no-negative-vibes/tests/test_compound_cones.py`

**Step 1: Write failing tests**

Add tests that:

```python
assert compound_matrix(matrix, 2)[row, column] == expected_minor
assert exterior_character_sum(product) == det(I + product)
assert chronological_product([a, b]) == b @ a
```

Use a noncommuting integer pair so reversing the order cannot pass accidentally.

**Step 2: Run the focused tests and observe the import failure**

Run:

```bash
cd tracks/qmc/solutions/no-negative-vibes
PYTHONPATH=. python3 -m pytest -q tests/test_compound_cones.py
```

Expected: FAIL because `oracle.compound_cones` does not exist.

**Step 3: Implement the minimal exact/float helpers**

Implement:

```python
def subset_basis(dimension: int, grade: int) -> tuple[tuple[int, ...], ...]: ...
def compound_matrix(matrix, grade: int): ...
def chronological_product(factors): ...
def exterior_traces(product) -> tuple[object, ...]: ...
def exterior_character_sum(product): ...
def exact_determinant_weight(factors): ...
```

Preserve SymPy matrices as exact objects and use NumPy for the screening path.

**Step 4: Re-run the focused tests**

Expected: PASS.

### Task 2: Implement typed paths and per-grade simplicial chart certificates

**Files:**
- Create: `tracks/qmc/solutions/no-negative-vibes/oracle/exterior_category_search.py`
- Create: `tracks/qmc/solutions/no-negative-vibes/tests/test_exterior_category_search.py`

**Step 1: Write failing tests**

Cover:

```python
edge = TypedEdge("e", source="a", target="b", matrix=...)
assert not graph.is_legal_word(("e", "e"))
assert graph.is_closed_word(("e", "r"))
with pytest.raises(ValueError, match="source"):
    graph.word_product(("e", "e"), require_legal=True)
```

Also freeze the exact four-dimensional calibration:

```text
S = [[-1,0,4,-2],[-2,1,4,-2],[2,-2,-3,2],[1,0,-2,1]]
C1 = [[1,1,0,0],[0,1,1,0],[0,0,1,0],[0,0,2,1]]
C2 = [[1,0,0,0],[3,3,2,4],[0,1,1,2],[0,0,0,1]]
B1 = S C1
B2 = S C2
det(I + B2 B1) = -12
```

With `B1,B2: a -> b` and `S^{-1}: b -> a`, legal closed paths telescope to products of totally nonnegative coordinate factors, but the untyped word `(B1,B2)` is exactly negative.

**Step 2: Run and observe the import failure**

Run:

```bash
PYTHONPATH=. python3 -m pytest -q tests/test_exterior_category_search.py
```

Expected: FAIL because the typed module does not exist.

**Step 3: Implement typed graph validation**

Add immutable `TypedEdge`, `GradeChartCertificate`, and `TypedExteriorGraph` objects. `word_product` must use chronological order, reject mismatched source/target labels when requested, and accept only closed paths for positivity certification.

**Step 4: Implement the GF(2) sign-chart solver**

For each grade, create one Boolean variable for every `(object, subset)` coordinate. Every nonzero compound entry imposes:

```text
x(target,row) XOR x(source,column) = sign(entry)
```

Solve connected components by parity propagation. Return diagonal `+/-1` charts only if all constraints are consistent, and verify every transformed edge compound is entrywise nonnegative. Record whether all grade charts are induced by the grade-one object gauges; induced certificates are classified `coboundary_calibration`, not discoveries.

**Step 5: Re-run focused tests**

Expected: PASS, including exact `-12` replay and legal-path exterior trace reconstruction.

### Task 3: Build the deterministic pilot runner

**Files:**
- Create: `tracks/qmc/solutions/no-negative-vibes/oracle/exterior_category_pilot.py`
- Create: `tracks/qmc/solutions/no-negative-vibes/tests/test_exterior_category_pilot.py`

**Step 1: Write failing tests**

Require a one-cell scan to:

- echo all parameters and seed;
- count every candidate and every determinant/compound check;
- distinguish `no_chart`, `typed_only_calibration`, `typed_grade_chart_survivor`, and `ambiguous`;
- save an exact integer/rational forbidden word for each survivor;
- write one manifest atomically and skip it on restart.

**Step 2: Run and observe the failure**

Run:

```bash
PYTHONPATH=. python3 -m pytest -q tests/test_exterior_category_pilot.py
```

Expected: FAIL because the pilot runner does not exist.

**Step 3: Implement two candidate grammars**

Implement:

1. `coboundary_tn_control`: integer unimodular object charts and products of nonnegative Jacobi factors; this must recover known typed-only controls and calibrate the exact witness path.
2. `support_aware_sparse`: determinant-positive sparse integer edges on two or three typed objects, with path/chord/feedback supports that are not generated from the collaborator's odd-cycle alphabet. The grade-chart solver, not construction, decides survival.

For each candidate, enumerate all untyped words through the configured short depth until the first stable negative is found. Legal closed paths are checked independently. Near-zero float64 weights are quarantined; a survivor is recorded only when SymPy confirms the forbidden word is strictly negative.

**Step 4: Implement resumable run-spec execution**

Expose:

```python
def scan_cell(...): ...
def run_spec(path: str | Path) -> dict[str, int]: ...
```

Write `cells/<cell-id>/manifest.json` via temporary-file replacement. Existing complete manifests are never recomputed.

**Step 5: Re-run focused tests**

Expected: PASS and byte-stable deterministic output for a fixed seed.

### Task 4: Freeze and execute the 50,000–100,000-check protocol

**Files:**
- Create: `tracks/qmc/solutions/no-negative-vibes/protocols/exterior-positive-category-v1/axes.json`
- Create: `tracks/qmc/solutions/no-negative-vibes/protocols/exterior-positive-category-v1/settings.json`
- Create: `tracks/qmc/solutions/no-negative-vibes/protocols/exterior-positive-category-v1/provenance.json`
- Create: `tracks/qmc/solutions/no-negative-vibes/protocols/exterior-positive-category-v1/README.md`

**Step 1: Declare the frozen pilot grid**

Use dimensions `4,5,6`, both grammars, deterministic seeds, and a short forbidden-word depth chosen so the declared total is 50,000–100,000 checks. Keep long typed stress depths `2,4,8,16,32` in settings, with exhaustive enumeration only through depth 8 when affordable.

**Step 2: Generate the run specification with the generic planner**

Run:

```bash
python3 scripts/parameter_scan.py plan \
  --axes tracks/qmc/solutions/no-negative-vibes/protocols/exterior-positive-category-v1/axes.json \
  --settings tracks/qmc/solutions/no-negative-vibes/protocols/exterior-positive-category-v1/settings.json \
  --provenance tracks/qmc/solutions/no-negative-vibes/protocols/exterior-positive-category-v1/provenance.json \
  --run-id exterior-positive-category-v1-pilot \
  --run-dir "$PWD/tracks/qmc/results/no-negative-vibes/exterior-positive-category-v1-pilot"
```

**Step 3: Run locally only after the generated plan confirms the budget**

Run:

```bash
cd tracks/qmc/solutions/no-negative-vibes
PYTHONPATH=. python3 -m oracle.exterior_category_pilot \
  ../../../../qmc/results/no-negative-vibes/exterior-positive-category-v1-pilot/run_spec.json
```

Expected: under ten minutes and 16 GB; every planned cell ends with an atomic manifest.

**Step 4: Collect and validate completeness**

Use `scripts/parameter_scan.py collect` with `completed=true` and fields for total checks, exact negative witnesses, chart survivors, and ambiguity count. Do not interpret an incomplete grid.

### Task 5: Exact replay, report, and integration

**Files:**
- Create: `tracks/qmc/solutions/no-negative-vibes/fixtures/exterior_positive_category_certificates.json`
- Create: `tracks/qmc/solutions/no-negative-vibes/docs/EXTERIOR_POSITIVE_CATEGORY_PILOT.md`
- Modify: `tracks/qmc/solutions/no-negative-vibes/docs/RESULTS_LEDGER.md`
- Modify: `tracks/qmc/solutions/no-negative-vibes/docs/PROJECT_MASTER_SUMMARY.zh-CN.md`
- Modify: `tracks/qmc/solutions/no-negative-vibes/START_HERE.md`

**Step 1: Export only minimal exact evidence**

Store integer/rational matrices, source/target labels, chart signs, forbidden word, exact determinant weight, classification level, and known-class status. Do not commit bulky cell manifests.

**Step 2: Write the result in two layers**

The technical report must distinguish:

- exact theorem for legal typed paths;
- exact failure after erasing types;
- induced coboundary calibrations;
- genuinely grade-specific finite survivors;
- finite-depth evidence versus arbitrary-depth certificates.

The Chinese summary must explain the same funnel without assuming group theory.

**Step 3: Run verification**

Run:

```bash
cd tracks/qmc/solutions/no-negative-vibes
PYTHONPATH=. python3 -m pytest -q
python3 -m py_compile oracle/compound_cones.py \
  oracle/exterior_category_search.py oracle/exterior_category_pilot.py
```

Expected: the full baseline plus new tests pass.

**Step 4: Commit and push**

Commit the plan first, then code/tests, then protocol/results. Push `work/xianzhi/exterior-positive-category-search` to the shared remote and update PR #7 with the exact pilot counts and calibrated claim boundary.
