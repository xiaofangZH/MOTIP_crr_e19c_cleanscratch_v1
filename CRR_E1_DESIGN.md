# CRR-E1: Query-Conditional Causal Entity State

## Purpose

CRR-E1 is the first representation-level upgrade over P0. It preserves the
pointer paradigm: identity labels are dynamic addresses, never learned semantic
classes. It replaces single-mention dominance with a query-conditional state
formed from strictly past mentions.

For current query `i`, past mention `j`, and entity `e`:

`a_ij = sqrt(d) * <q_i, k_j>`

`u_ij = a_ij / tau + b_time(delta_t) + log r_j`

`alpha_ij|e = softmax_{j in e}(u_ij)`

`z_ie = normalize(sum_{j in e} alpha_ij|e * k_j)`

`s_state(i,e) = sqrt(d) * <q_i, z_ie>`

A shared residual head receives P0 max score, state score, support, entropy,
causal age, recent evidence mass, reliability, and concentration. Its final
layer is zero initialized, so CRR-E1 is exactly P0 before learning.

## Invariants

- Strictly causal history only.
- Exact identity-renaming equivariance.
- No GT enters runtime state or scoring.
- No learned ID dictionary.
- P0-exact initialization.
- Phase E1 changes history representation only; current-set reasoning and the
  context-conditioned NEW head are deferred to later isolated stages.

## Validation protocol

1. Unit tests: exact P0 initialization, permutation equivariance, causality,
   and nonzero learning gradient.
2. Forty-step train8 smoke for numerical and optimizer-path validation.
3. Three full train32 epochs from P0 checkpoint 9, evaluating each checkpoint
   on the established training-side trend split.
4. Continue to a longer schedule only if the learned branch is active and the
   learning/closed-loop trend supports it; do not reject a learnable module from
   the short smoke alone.
