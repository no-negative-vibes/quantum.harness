# Exterior positive category pilot v1

This protocol searches two-object typed matrix alphabets with two
`a -> b` edges and one `b -> a` edge.  A candidate advances only when exact
rational exterior-grade charts prove every legal closed history nonnegative.
Erasing the types must expose an exact negative word.

The pilot contains:

- a known TN coboundary control;
- a sparse support-aware grammar made from identity/three-cycle permutations,
  one positive rational diagonal scale, and one to three integer shears;
- dimensions `4,5,6`, four deterministic cell seeds, and 200 candidates per
  cell;
- exactly 86,400 float screening edge-grade compound matrices
  (`sum_cells candidates * 3 edges * (dimension + 1 grades)`), followed by
  exact replay and type-erasure determinants;
- legal closed-word sanity replays at depths `2,4,8,16,32`.

Float64 is only a cheap rejector.  Every theorem-level chart and every saved
negative word is rebuilt from the integer definition with SymPy.  A
signed-permutation/gauge-TN hit is a known reduction.  The 1,000-assignment cap
is exhaustive in dimension 4 (`24^2 = 576`) but not in dimensions 5 or 6; a
cap-limited miss is therefore `known_class_filter_incomplete`, never a
survivor.  A negative singleton also proves that primitive edge cannot be one
real exponential, so it is mathematical evidence only.

From the harness root:

```bash
PROTO=tracks/qmc/solutions/no-negative-vibes/protocols/exterior-positive-category-v1
RUN_DIR="$PWD/tracks/qmc/results/no-negative-vibes/exterior-positive-category-v1-pilot"

python3 scripts/parameter_scan.py plan \
  --axes "$PROTO/axes.json" \
  --settings "$PROTO/settings.json" \
  --provenance "$PROTO/provenance.json" \
  --run-id exterior-positive-category-v1-pilot \
  --run-dir "$RUN_DIR"

cd tracks/qmc/solutions/no-negative-vibes
PYTHONPATH=. python3 -m oracle.exterior_category_pilot \
  "$RUN_DIR/run_spec.json"

PYTHONPATH=. python3 -m oracle.exterior_category_pilot \
  "$RUN_DIR/run_spec.json" --verify-only
```

The same runner command resumes safely.  It reuses a cell only when
`compute_success`, cell parameters, settings, provenance, and the full
run-spec digest all match.

Collect from the harness root:

```bash
python3 scripts/parameter_scan.py collect \
  --run-spec "$RUN_DIR/run_spec.json" \
  --success-field compute_success \
  --success-value true \
  --value-field counts.edge_grade_checks \
  --value-field counts.determinant_checks \
  --value-field counts.signed_permutation_tn_reduction \
  --value-field counts.provisional_front_door_survivor \
  --value-field counts.known_class_filter_incomplete
```

The collector is descriptive.  Scientific interpretation requires every
planned manifest to be present and matching; missing or failed cells invalidate
the aggregate.

After strict verification, regenerate the compact tracked evidence from the
solution directory:

```bash
cd tracks/qmc/solutions/no-negative-vibes
PYTHONPATH=. python3 -m oracle.exterior_category_report \
  "$RUN_DIR/run_spec.json" \
  fixtures/exterior_positive_category_pilot.json
```

The reporter rechecks classification conservation, exact-negative witnesses,
legal nonnegative stress weights, and provisional-candidate gate consistency.
It retains the complete support-aware signed-permutation reduction but does not
copy thousands of known-control witnesses into Git.
