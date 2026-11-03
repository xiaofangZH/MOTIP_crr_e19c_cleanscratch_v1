# Full CRR Roadmap

## Core paradigm

CRR treats an identity as a dynamic causal entity state generated from past
observations. Identity numbers are addresses only. The model is exactly
equivariant to identity renaming and never requires a learned ID dictionary.

## Staged architecture

### P0 — Causal pointer

Learned query/key metric with strict causal masking and max mention retrieval.

### E1 — Query-conditional causal entity state (current stage)

Reliability- and time-aware attention creates a dynamic state from multiple
mentions. A zero-initialized residual preserves P0 at initialization.

### E2 — Relational current-set reasoner

Current detections exchange information through permutation-equivariant
self-attention, then query dynamic entity states. This moves one-to-one conflict
resolution into the learned representation while retaining Hungarian only as a
final exact allocator.

### E3 — Context-conditioned NEW competition

NEW is scored relative to the best existing entities, score margin, uncertainty,
track age, and set context instead of from the current feature alone.

### E4 — Structured allocation objective and robust history curriculum

Training aligns with the augmented existing-entity/replicated-NEW assignment
matrix. Causal history corruption progresses from occlusion to identity-switch
noise and then limited closed-loop history, without allowing GT into runtime.

## Experimental discipline

- E1, E2, E3, and E4 are added sequentially and each has a matching ablation.
- A short smoke proves execution, not accuracy. A learnable stage receives at
  least three full epochs; continuing trends receive the full schedule.
- Architecture decisions use the training-side development protocol. Locked
  validation is reserved for a frozen candidate.
- Failed post-hoc B3/B6 risk gates are not carried into the decoder. Their main
  lesson is that irreversible local decisions cannot reliably predict recursive
  closed-loop utility; reliability therefore acts softly inside representation.
- Final reporting includes accuracy, parameter count, FLOPs, convergence curves,
  and seed stability against both P0 and the mature official ID Decoder.

## Experiment history

A detailed, evidence-graded log of all CRR experiments (E5-E19d, R1/R3, and the
official A6000 reproduction) is maintained in
[CRR_EXPERIMENT_LOG.md](CRR_EXPERIMENT_LOG.md).
