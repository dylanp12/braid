# Research basis and required comparators

Braid's research thesis is tested against, not inferred from, related work.
Required comparator families include:

- [InGram](https://proceedings.mlr.press/v202/lee23c.html) and
  [ULTRA](https://arxiv.org/abs/2310.04562) for unseen entity/relation transfer;
- [POSTRA](https://arxiv.org/abs/2506.06367) for fully inductive temporal
  knowledge-graph reasoning;
- [Gamma](https://arxiv.org/abs/2512.22931) and comparable current multi-graph
  relation-transfer systems;
- [GET](https://openreview.net/forum?id=786oOfRVXO) and official Temporal Graph
  Benchmark models for event-sequence and future-link prediction;
- [TGB 2.0](https://arxiv.org/abs/2406.09639) for reproducible multi-domain
  temporal heterogeneous/knowledge-graph evaluation; and
- [GraphBFF](https://arxiv.org/abs/2602.04768) and applicable RelBench systems
  for scaling and unseen relational-data transfer.

The registry must be refreshed and frozen before each confirmatory release.
Naming a method here does not qualify it; qualification requires an exact code
commit, checkpoint, environment, allowed-input declaration, tuning budget, and
successful reproduction of its official anchor result.
