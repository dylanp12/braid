# Data governance

## Allowed global-training inputs

Global Braid training may use only affirmatively licensed public or contracted
corpora whose recorded allowed uses include machine-learning research and model
training. A public URL is not a redistribution license.

Every dataset manifest binds source, acquisition date, terms snapshot, content
and transform hashes, allowed uses, personal-data classification, schema family,
organization/repository cluster, split, retention rule, and deletion contact.

## Private product data

Alluvia notes, judgments, embeddings, proposal slates, and derived examples stay
inside their user or organization boundary. They do not enter the global corpus,
benchmark, tokenizer, retrieval index, or shared checkpoint. Future org-local
adapters must be deletable with the source organization and cannot pool updates
across organizations.

## Split and contamination policy

Split hierarchy is schema family, organization, fork/mirror/code-copy cluster,
repository, then chronological block. Alias resolution, near-duplicate text,
inverse relations, deterministic restatements, templates, and copied subgraphs
must be clustered before assignment. Tokenizers, normalizers, indexes, candidate
generators, and thresholds fit on training data only.

## Synthetic data

Synthetic records must have unique identities, dual clocks, explicit causal
mechanisms, censoring, retractions, and schema evolution. Repeated synthetic
records do not count toward scaling-token floors. Synthetic-only performance is
diagnostic and cannot establish real-world capability.
