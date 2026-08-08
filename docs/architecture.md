# Braid 2 architecture

## Research thesis

Braid 2 is a schema-conditioned temporal event transducer. It receives an
as-of graph prefix and a machine-readable schema, then returns a probability
distribution over constrained future graph patches. It does not generate free
text explanations and it never commits a predicted mutation.

The plausible contribution is the combined system: open-schema structural
induction, dual-clock forecasting, arbitrary-role persistent memory, censored
quiet windows, and evidence-basis-constrained decoding. No component by itself
authorizes a novelty or frontier claim.

## Causal data path

1. Validate the complete v2 bundle before any model code sees it.
2. Construct a prefix using `observed_at <= cutoff`; remove every later node,
   node snapshot, event, evidence record, exposure, and judgment.
3. Encode schema declarations and support events without global entity,
   relation, schema, or repository embeddings.
4. Serialize visible events in observation order. Represent validity as a
   signed lag relative to observation time.
5. Update participant memories with explicit argument roles.
6. Retrieve typed candidates and report retrieval coverage independently.
7. Generate graph-patch fields through a finite-state grammar.
8. Aggregate rollouts into event marginals and calibrated uncertainty.
9. Return proposals with the immutable model-manifest ID; never mutate source
   data from inside the model package.

The prefix-invariance contract is strict: appending any future data to a bundle
must not change the bytes of the historical prefix or its deterministic model
inputs.

## Model family

The EventGraph model has four independently ablatable parts:

- bidirectional schema encoder for type, relation, role, and constraint text;
- higher-order relation-motif encoder using only causally visible events;
- causal event decoder with field-specific heads and censored-time likelihood;
- role-aware persistent node memory plus typed retrieval.

Dynamic handles are allocated within an episode and randomly renamed during
training. The decoder may point to visible nodes, evidence, types, and relations.
It must emit `SCHEMA_CHANGE` before using a novel relation declaration.

## Scale promotion

The checked-in configurations are experimental candidates, not promises to run
or claims of capability. Promotion requires valid licensed data, zero causal or
split audit failures, monotonic fixed-compute scaling, improved macro transfer,
qualified baseline coverage, and explicit cloud authorization.

No public release is described as a foundation or frontier model until a future
claim-grade controller recomputes every E/R/S/T/P endpoint from raw artifacts and
accepts two independent sealed cohorts. This engineering release's controller is
intentionally unable to authorize non-engineering vocabulary.
