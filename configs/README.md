# Candidate configuration declarations

Each JSON file is a complete, reviewable declaration for one candidate scale. The
architecture fields mirror `EventGraphConfig`; tests prevent them from drifting from
the executable declarations in `braid.model.config`. `expected_parameters` is the
exact count for the declared default byte tokenizer, measured with a meta-device model
that allocates no weights. `target_parameters` remains the rounded research label.

The training and promotion sections are policy, not evidence that a run occurred.
`cloud_authorized` is false in every checked-in declaration. Changing it in this public
repository cannot authorize a cloud run; authorization is a separately signed private
lab artifact.

These files do not make a model claim. Current release and evidence status is reported
by `braid status` and `docs/implementation-status.md`.
