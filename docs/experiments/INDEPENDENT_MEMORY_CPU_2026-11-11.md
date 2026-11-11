# AMN-4W — Independent CPU memory study (2026-11-11)

**Scope:** isolated experimental evidence, not a new production release and **not** an AMN-4W v0.4 model upgrade. The repository's existing architecture and trained checkpoints remain unchanged.

## Provenance and reproducibility

- Experiment source: [`experiments/independent_memory_20261111.py`](../../experiments/independent_memory_20261111.py).
- Archived observed output: [`results/independent-memory-20261111/observed-partial.log`](../../results/independent-memory-20261111/observed-partial.log).
- Original local test artifact used for the recorded observations: `amn4w_local_test.py` (provided in the conversation). The repository copy is a refactored, independent reproduction implementation; **its exact output has not been checked against the archived log**.
- Local execution environment reported: PyTorch **2.10.0+cpu**, CPU, single PyTorch thread. Original numeric probe used seed **42**, batch **200**, memory dimension **64** and **8** stored keys; original synthetic training used vocabulary **50**, **2** chains, **350** steps/config/seed, batch **64**, 500 evaluation samples per seed, 3 seeds planned.
- The archived log contains one numeric report and **6 observed training runs** (baseline seeds 0,1,2 and leak0.5 seeds 0,1,2). A prior conversation summarized 12 runs, but the available archived raw log does **not** substantiate all 12. Results for decorrelation and combined models must remain **unverified**, not inferred from that prior summary.
- Neither the independent proxy nor the observations are tests of `src/amn4w/v04/memory.py` or its learned frontend; no actual trained v0.4 model was tested in these experiments.
- The official v0.4 documented target thresholds were not met in its original published validation; do not reinterpret these toy-task results as overriding those failures.

## Mathematics and hypotheses

For memory state `S`, a new key `k` and the set of protected keys `K`, define the residual

\[
r=(I-P_{\mathrm{span}(K)})k,\qquad \rho=\|r\|_2,
\qquad \widetilde{k}=r/\rho^2.
\]

When \(\rho>0\) and numerical rank is correctly identified, \(\widetilde{k}^{\mathsf T}k=1\) and \(\widetilde{k}^{\mathsf T}k_j=0\) for all protected \(k_j\). Hence

\[
S_{\mathrm{write}}'=S+\eta(v-Sk)\widetilde{k}^{\mathsf T},\qquad
S_{\mathrm{erase}}'=S-(Sk_i)\widetilde{k}_i^{\mathsf T}
\]

preserve protected reads in exact arithmetic, while targeting the selected read. This is **conditional on address separability**, not an unconditional no-interference guarantee.

The implementation uses float64 SVD and a numerical-rank mask for the residual, converts results back to the incoming memory dtype, and declines nonseparable cases. Very small \(\rho\) creates large update norm:

\[
\|\Delta S\|_F=\eta\|v-Sk\|_2/\rho.
\]

Thus exact read isolation can incur large memory-state growth and sensitivity to query perturbations. No robustness guarantee follows from low read error at exact addresses.

### Existing v0.4 architecture is different

The **existing** `src/amn4w/v03/protected.py` already uses dual projection with a pseudoinverse; `src/amn4w/v04/memory.py` inherits it and adds typed learned addresses, a trained clause-level frontend and explicit tombstones. The toy proxy instead uses a token embedding, previous-token key/next-token value, explicit write masks and an optional two-hop query layer. It has no equivalent pretrained frontend or real-language question parser. Accordingly, treating the proxy results as validation of the v0.4 runtime would be incorrect.

## Actual archived observations

Numeric results from original local script, 200 random test examples (values are maxima or means as indicated):

| Measurement | Result |
|---|---:|
| Dual erase: max target residual norm | 4.8498226e-8 |
| Dual erase: max protected-read change | 1.9369644e-8 |
| Duplicate stored key: max protected-read change | 2.0692401e-8 |
| Duplicate-key deletion accepted | 100% |

Controlled new write for various key cosine similarities (standard vs dual):

| cos(new, neighbor) | Standard: neighbor cosine | Dual: neighbor cosine | Dual: maximum old-read change | Mean dual update Frobenius norm |
|---:|---:|---:|---:|---:|
| 0.3 | 0.941616 | 1.000000 | 3.8192e-8 | 1.20557 |
| 0.9 | 0.201288 | 1.000000 | 7.0302e-8 | 3.21767 |
| 0.99 | 0.025924 | 1.000000 | 2.4611e-7 | 10.30842 |
| 0.9999 | 0.014310 | 1.000000 | 2.1384e-6 | 103.56767 |

The measured isolation is limited to the sampled exact keys. The increase in state norm at high cosine is significant; this experiment does not measure long-horizon read stability.

Archived synthetic training results (these are observed **individual seeds**, not a completed full-factorial sweep):

| Config | Seed | FINAL | r2→C | Leakage | Mean maximum abs key cosine |
|---|---:|---:|---:|---:|---:|
| baseline | 0 | 1.000 | 1.000 | 0.201821 | 0.170692 |
| baseline | 1 | 1.000 | 1.000 | 0.201536 | 0.172188 |
| baseline | 2 | 1.000 | 1.000 | 0.206890 | 0.175299 |
| leak0.5 | 0 | 1.000 | 1.000 | 0.176014 | 0.149621 |
| leak0.5 | 1 | 1.000 | 1.000 | 0.176904 | 0.149807 |
| leak0.5 | 2 | 1.000 | 1.000 | 0.181268 | 0.154991 |

Observed three-seed mean leakage: baseline **0.203416**, leak0.5 **0.178062**; relative reduction about **12.46%**, with no FINAL deterioration in this tiny task. These values are exploratory: 3 seeds, small vocabulary, heavily supervised intermediate targets and shared evaluation generation. They do not support a causal claim that `key_decorrelation` is beneficial; its archived training results are missing.

## How to reproduce without altering main runtime

From the repository root, in a Python environment with PyTorch installed:

```bash
python experiments/independent_memory_20261111.py --numeric
python experiments/independent_memory_20261111.py --train --steps 350 --seeds 3
```

The script defaults to CPU and executes independent synthetic tests. These commands may generate different timings or tiny floating-point differences across hardware and PyTorch versions. **They do not rerun or validate the original v0.4 development/final datasets.** For v0.4-specific verification, use the instructions in `README_V04_TR.md`.

## Open validation work

1. Re-run the repository copy and compare its outputs with the original archived log.
2. Complete the 5-config × 5-seed × 3000-step sweep on a **versioned, fixed model**, preserving individual raw run records and experiment signatures.
3. Compare leakage, `FINAL`, 2-hop intermediate correctness, \(\rho\) and neighbor damage with paired identical eval episodes; report confidence intervals.
4. Measure noisy-key retrieval error and state norm as \(\rho\to0\), as well as gradient stability through the SVD path in *training*, not just forward evaluation.
5. Port a separately qualified candidate to the actual v0.4 runtime, then compare against the **existing** dual-projection memory without modifying prior release artifacts. Preserve baseline failures and default model unless independent targets are met.

**Status:** numeric memory probes — supported by the archived local run; small baseline and leak-penalty training — partially supported by six raw runs; full sweep, v0.4 integration, long-horizon noise resilience and Transformer parity — **not verified**.
