# AMN-4W independent synthetic sweep — completed small-scale run

**Scope:** independent proxy model from conversation, NOT GitHub AMN-4W v0.4 production model. **Not the requested full 3000-step run.**

- Environment: PyTorch 2.10.0+cpu, CPU.
- 5 configurations × 5 seeds = **25/25 completed**.
- Steps: **350** per seed; V=50, M=2, batch=64; eval=500 independent synthetic examples per seed; aux weight 1.0; 2-hop.
- All 25 seeds had FINAL=1.000 on this small synthetic task. This is a saturated metric, **not evidence of generalized language ability**.
- The 3000-step V=300 M=4 run was attempted but did not yield any completed result within runtime limits.
- Leakage is an unnormalized proxy ratio; reductions do not mean all interference is removed.

| Configuration | FINAL mean±sd | leakage mean±sd | relative leakage reduction | erase other | mean max|cos| | min ρ |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1.0000 ± 0.0000 | 0.2017 ± 0.0032 | 0.0% | 0.0181 ± 0.0011 | 0.1721 ± 0.0028 | 0.9733 ± 0.0008 |
| leak0.5 | 1.0000 ± 0.0000 | 0.1772 ± 0.0031 | 12.2% | 0.0157 ± 0.0011 | 0.1516 ± 0.0030 | 0.9793 ± 0.0006 |
| decorr5 | 1.0000 ± 0.0000 | 0.1537 ± 0.0026 | 23.8% | 0.0132 ± 0.0006 | 0.1337 ± 0.0033 | 0.9841 ± 0.0008 |
| decorr20 | 1.0000 ± 0.0000 | 0.1008 ± 0.0049 | 50.0% | 0.0087 ± 0.0009 | 0.0903 ± 0.0031 | 0.9926 ± 0.0005 |
| leak0.5+decorr5 | 1.0000 ± 0.0000 | 0.1415 ± 0.0027 | 29.8% | 0.0122 ± 0.0006 | 0.1234 ± 0.0032 | 0.9864 ± 0.0007 |

All 25 runs also showed r2→C=1.000 and q2→k(B)=1.000 at evaluation. The fraction with minρ<0.3 and the rank-deficient rate were 0 in these tests.

**Preliminary finding:** decorrelation weight 20 had the lowest leakage and smallest standard-erase disturbance in this small setup (the other 4 configurations showed higher values). Because accuracy is saturated, these values cannot select the final architecture. The effect on larger dictionaries, longer chains, noisy queries, and real v0.4 model remains unknown.

**Code:** `amn4w_sweep_real.py` reads `amn4w_local_test.py`. The experiment identity is `775c2ff13bca1ffc` for the completed small configuration.

**Reproduction:** `python amn4w_sweep_real.py --steps 350 --vocab 50 --chains 2 --seeds 5 --eval 500 --workers 1 --out amn4w_partial_350_sweep.jsonl`. The JSONL preserves all 25 per-seed raw metrics.

## Per-seed records

| Config | seed | FINAL | leakage | erase dnorm | maxcos |
|---|---:|---:|---:|---:|---:|
|baseline|0|1.000|0.20182|0.01895|0.17069|
|baseline|1|1.000|0.20153|0.01804|0.17219|
|baseline|2|1.000|0.20689|0.01935|0.17530|
|baseline|3|1.000|0.19939|0.01729|0.17396|
|baseline|4|1.000|0.19878|0.01675|0.16812|
|leak0.5|0|1.000|0.17601|0.01685|0.14962|
|leak0.5|1|1.000|0.17690|0.01507|0.14981|
|leak0.5|2|1.000|0.18128|0.01661|0.15499|
|leak0.5|3|1.000|0.17864|0.01543|0.15468|
|leak0.5|4|1.000|0.17304|0.01431|0.14870|
|decorr5|0|1.000|0.15141|0.01393|0.13103|
|decorr5|1|1.000|0.15142|0.01239|0.12942|
|decorr5|2|1.000|0.15624|0.01363|0.13553|
|decorr5|3|1.000|0.15662|0.01349|0.13667|
|decorr5|4|1.000|0.15276|0.01274|0.13605|
|decorr20|0|1.000|0.10576|0.00984|0.09425|
|decorr20|1|1.000|0.09634|0.00757|0.08584|
|decorr20|2|1.000|0.09901|0.00902|0.08944|
|decorr20|3|1.000|0.10621|0.00897|0.09208|
|decorr20|4|1.000|0.09650|0.00809|0.08985|
|leak0.5+decorr5|0|1.000|0.13856|0.01291|0.12049|
|leak0.5+decorr5|1|1.000|0.13907|0.01130|0.11946|
|leak0.5+decorr5|2|1.000|0.14296|0.01262|0.12542|
|leak0.5+decorr5|3|1.000|0.14511|0.01252|0.12557|
|leak0.5+decorr5|4|1.000|0.14178|0.01190|0.12618|