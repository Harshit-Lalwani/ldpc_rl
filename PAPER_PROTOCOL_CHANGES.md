# RELDEC-paper protocol: edited copies of the tanh-MI and RELDEC scripts

This branch adds **new files only**; none of the existing scripts are modified. The new files are minimally
edited copies of `rl_tanh_mi.m`, `reldec_test.m`, `test_rl_mi_tanh.m`, `test_reldec.m` and `ldpc_cluster.m`
(taken at commit `41d06c13`). Each edit is marked in the code with `% PAPER-PROTOCOL: <item>`.

**Why the copies exist.** Harshit's MATLAB pipeline and your scripts gave different BER for the same
methods. The copies put both codebases on one training/evaluation protocol, close to the RELDEC paper
(Habib, Beemer, Kliewer, IEEE Trans. Commun. 2023), and feed both from one common set of frames and random
numbers. With identical inputs the two codebases then produce **bit-identical Q tables and identical
decoding decisions** (see *Verification*).

| New file | Copy of | Role |
|---|---|---|
| `make_common_db.m` | new | writes the common training and evaluation databases |
| `rl_tanh_mi_paper.m` | `rl_tanh_mi.m` | tanh-MI training |
| `reldec_paper.m` | `reldec_test.m` | RELDEC training |
| `test_rl_mi_tanh_paper.m` | `test_rl_mi_tanh.m` | tanh-MI evaluation |
| `test_reldec_paper.m` | `test_reldec.m` | RELDEC evaluation |
| `ldpc_cluster_paper.m` | `ldpc_cluster.m` | check-node update with clipped numerics |

## How to run (MATLAB, from the repository root)

```matlab
make_common_db('common_db');      % ~80 s: train_db_wran.mat (15,000 frames), eval_db_wran.mat (100,000 frames), common_db.md5

[Q, out] = rl_tanh_mi_paper('common_db/train_db_wran.mat', 1:15000, 'Q_wran_tanh_mi_paper.mat');
res = test_rl_mi_tanh_paper('common_db/eval_db_wran.mat', 'Q_wran_tanh_mi_paper.mat', 1:6, 1:100000);

[Q, out] = reldec_paper('common_db/train_db_wran.mat', 1:15000, 'Q_wran_reldec_paper.mat');
res = test_reldec_paper('common_db/eval_db_wran.mat', 'Q_wran_reldec_paper.mat', 1:6, 1:100000);
```

`res.ber` / `res.fer` give one value per SNR. `res.bit_errors`, `res.iterations`, `res.messages` and
`res.actions` are per (SNR, frame), for frame-by-frame comparisons. Smaller runs: pass fewer episodes, for
example `1:1000`, or fewer frames, for example `1:200`. `make_common_db(dir, struct('n_train', N1, 'n_eval', N2))`
writes the first N1 / N2 frames of the same databases.

## The common databases (`make_common_db.m`)

WRAN code (`wran_384_256.mat`, 128 x 384), rate `R = 1 - m/n` computed in floating point (not the literal
2/3, which differs by one ulp), all-zero codeword, BPSK +1.

- **Training set:** 15,000 frames. Frame e is at Eb/N0 `[1 2 3 4 5 5.5]` dB, cycled, so 2,500 frames per
  SNR, as in the paper (K = 6, 15,000 frames in total). Noise comes from `RandStream('Threefry', 'Seed',
  20261021)` with substream e.
- **Per-step random numbers:** 3 x 50 per training episode (exploration test, exploration action,
  tie-break), from Threefry seed 20261022, substream e.
- **Evaluation set:** 100,000 noise vectors from Threefry seed 20261023, substream f. The same vectors are
  reused at every evaluation SNR (1 to 6 dB).
- **Evaluation tie-break draws:** generated per frame, not stored: Threefry seed 20261024, substream
  `(j-1)*100000 + f`, 640 values (128 clusters x 5 iterations) per frame.

In Octave there is no Threefry, so the generator falls back to `randn('state', [seed; substream])`. That is
for local testing only: the numbers differ from the MATLAB ones.

## Protocol (what the copies implement)

| | Setting |
|---|---|
| Training SNRs | 6 SNRs, 1, 2, 3, 4, 5, 5.5 dB Eb/N0 (paper), 2,500 frames each |
| Training budget | one pass over the 15,000 frames (paper) |
| Episode | `lmax` = 50 learning steps; one step schedules one check node (z = 1); any check node can be picked, repeats allowed (paper and your code) |
| Early stop | the episode ends as soon as the syndrome is zero, checked after every step (our choice; the paper always runs all 50 steps) |
| Exploration | epsilon = 0.6 (paper); exploration picks a check node uniformly at random |
| Discount | beta = 0.9 |
| Learning rate | alpha = 0.1 |
| Update | `Q(s,a) = (1-alpha) Q(s,a) + alpha (r + beta max_k Q(s'_k, k))`, maximum over all check nodes (your code) |
| Initial Q | 0 |
| Ties | broken uniformly at random among the maximising check nodes (your code took the first index) |
| tanh-MI state | mean tanh(L) over the check node's neighbours, 40 bins of width 0.05: `s = min(floor((x+1)/0.05), 39) + 1` |
| tanh-MI reward | increase in the neighbours' mean MI, `MI(L) = J(sqrt(max(2L, 0)))` |
| RELDEC state and reward | yours, unchanged: bits `L < 0` of the first 11 neighbours, MSB first; reward = fraction of neighbours with `L >= 0` |
| Check-node update | `|tanh| >= 1e-12`, leave-one-out product clipped to `±(1 - 1e-15)`, message `log((1+p)/(1-p))` (instead of `2*atanh`) |
| Evaluation | greedy, 5 iterations, every check node once per iteration in policy order, each check node's own state, stop after an iteration that ends with a zero syndrome |
| BER | over all n = 384 bits (the paper's definition: Pr[x_v != x̂_v], v = 1..n) |

## Changes, file by file

### `rl_tanh_mi_paper.m` and `reldec_paper.m` (training)

1. **Function form:** `[Q, out] = f(db_path, episodes, q_out)` instead of a script.
2. **Data from the common training set** instead of generating `L_set` in the script. In `reldec_test.m`
   this also removes the use of `R`, which the script never defined.
3. **Randomness from the database:**
   - exploration test `u_eps < epsilon`, then action `a = floor(u_act*128) + 1`;
   - greedy ties `a = ties(floor(u_tie*numel(ties)) + 1)`, instead of `[~, a] = max(vals1)`, which always
     took the first index (with Q starting at 0, check node 1 was always picked until the Q values
     separated).
4. **Exploration:** `epsilon` 0.1 -> 0.6.
5. **tanh-MI state:** 22 levels with `floor((x+1)/2*22)+1` -> 40 bins of width 0.05.
6. **tanh-MI reward:** `Jfun(x)` on the raw LLR -> `Jfun(sqrt(max(2*x, 0)))`.
7. **Check-node update:** the clipped numerics above, instead of `2*atanh(prodLq ./ temp)`, which gives
   Inf/NaN once LLRs saturate.
8. **Q value used for the next action:** after `Q(s(a),a)` is updated, `vals1(a)` is refreshed to the
   updated value. In the original, the next greedy choice still saw the value from before the update.
9. **Early stop:** the episode ends once the syndrome is zero (after that step's update).
10. **Outputs:** `out` holds per-episode actions, steps, bit errors and messages, for frame-by-frame
    comparison; Q and meta are saved to `q_out`.

### `test_rl_mi_tanh_paper.m` and `test_reldec_paper.m` (evaluation)

1. **Function form:** `out = f(db_path, q_path, snr_idx, frames)`, with the one trained table passed in
   and used at every SNR (the protocol trains one table on the SNR mix). Your RELDEC evaluation used
   `Q1{3}` for every SNR; your newest tanh-MI evaluation uses SNR-matched tables `Q1{i}`.
2. **Frames from the common evaluation set;** SNRs 1 to 6 dB (the RELDEC loop started at `i = 4`).
3. **No Communications Toolbox:** all-zero codeword instead of `ldpcEncode`; bit errors counted as
   `L < 0` over all n bits instead of `biterr` over the k information bits.
4. **Each check node's own state:** `current_state(v)`, as you have now fixed upstream in `a964f4a`.
5. **tanh-MI state:** 40 bins of width 0.05, matching training.
6. **Ties:** uniformly random, with the per-frame draws described above (was first index).
7. **Check-node update:** `ldpc_cluster_paper.m` (clipped numerics).
8. **Stopping:** after any iteration that ends with a zero syndrome (was: always all 5 iterations).
9. **Outputs:** per-frame bit errors, iterations, messages (sum of the scheduled check nodes' degrees)
   and the action sequence; per-SNR BER and FER.

### `ldpc_cluster_paper.m`

A copy of `ldpc_cluster.m` with the clipped numerics. It is otherwise identical.

## Issues found in the original scripts (as of `41d06c13`), and their status on your current `main`

| Issue | Status on `main` (`770f248`) |
|---|---|
| `test_rl_mi_tanh.m` gave every check node the state of check node `i` (the SNR-loop index): `current_state(i)` instead of `current_state(v)`, in both the initialisation and the update after every step | **fixed** in `a964f4a` |
| `test_rl_mi_tanh.m` decoded every SNR with the one table `Q_wran_0_tanh_mi` | now uses SNR-matched tables `Q1{i}` |
| tanh-MI reward fed the raw LLR to `Jfun`, which expects a sigma | now `Jfun(sqrt(max(x, 0)))`. Note the factor 2: the usual consistent-Gaussian relation is sigma = sqrt(2L), which is what the copies use |
| `2*atanh(prodLq ./ temp)` with no clipping: in our runs about half of all frames at 3 dB ended with NaN LLRs, and `NaN < 0` is false, so those bits counted as **correct** | still present |
| Training noise uses amplitude `sqrt(snr)` and sigma = 1 with no rate factor, while the evaluation uses Eb/N0 with the rate. A training "snr" therefore corresponds to Eb/N0 = 10 log10(snr) - 1.25 dB on WRAN (snr(1) = 1 is about -1.25 dB, snr(3) about +0.75 dB) | still present |
| `reldec_test.m` uses `R` without defining it | still present |
| `test_reldec.m` uses `Q1{3}` for every SNR and starts the SNR loop at `i = 4` | still present |
| Greedy ties take the first index (deterministic; with Q starting at 0 the first check node is always picked) | still present (a design choice) |
| BER over the k information bits; the paper defines it over all n bits | still present (a design choice) |

## Verification against Harshit's MATLAB pipeline

Both codebases were fed identical inputs from the common databases:

| Check | Engine | Size | Result |
|---|---|---|---|
| tanh-MI training | MATLAB R2024a | 1,000 episodes | Q tables **bit-identical** (0 of 5,120 entries differ); identical action sequences |
| tanh-MI evaluation | MATLAB R2024a | 200 frames x 6 SNRs | identical actions, bit errors, iterations and messages for every frame |
| tanh-MI and RELDEC | Octave | 300 episodes; 60 frames x 6 SNRs | identical Q tables and identical decoding |

Wall times in MATLAB for that run, edited copies vs Harshit's pipeline: training 16.9 s vs 6.5 s;
evaluation of 1,200 frames 71.5 s vs 23.8 s. The difference is mostly that the copies recompute every
check node's state after each step, where the pipeline updates only the neighbours that changed.
