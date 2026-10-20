# Changes to the scripts

Base: commit `41d06c13`. The edited copies are new files, so the original scripts are untouched. Each
edit is marked `% PAPER-PROTOCOL` in the code.

## `rl_tanh_mi.m` → `rl_tanh_mi_paper.m`, `reldec_test.m` → `reldec_paper.m`

1. Script turned into a function `[Q, out] = f(db_path, episodes, q_out)`.
2. Training frames read from `train_db_wran.mat` instead of being generated in the script (this also
   removes the undefined `R` in `reldec_test.m`).
3. Exploration test and exploration action read from the database: `u_eps < epsilon`, then
   `a = floor(u_act*128) + 1`.
4. Greedy ties broken uniformly at random using the database: `a = ties(floor(u_tie*numel(ties)) + 1)`,
   instead of `[~, a] = max(vals1)` (first index).
5. tanh-MI only: 22 state levels `floor((x+1)/2*22)+1` → 40 bins of width 0.05,
   `min(floor((x+1)/0.05), 39) + 1`.
6. tanh-MI only: reward `Jfun(x)` → `Jfun(sqrt(max(2*x, 0)))`.
7. Check-node update `2*atanh(prodLq ./ temp)` → `|tanh| >= 1e-12`, product clipped to `±(1 - 1e-15)`,
   message `log((1+p)/(1-p))`.
8. After `Q(s(a),a)` is updated, `vals1(a)` is refreshed, so the next greedy choice uses the updated value.
9. Episode ends early once the syndrome is zero (checked after each step's update).
10. Per-episode outputs added (`out`: actions, steps, bit errors, messages); Q and meta saved to `q_out`.

## `test_rl_mi_tanh.m` → `test_rl_mi_tanh_paper.m`, `test_reldec.m` → `test_reldec_paper.m`

1. Script turned into a function `out = f(db_path, q_path, snr_idx, frames)`; the trained table is passed in
   and used at every SNR (`test_reldec.m` used `Q1{3}`).
2. Frames read from `eval_db_wran.mat`; SNRs 1–6 dB (`test_reldec.m` started the SNR loop at `i = 4`).
3. All-zero codeword instead of `ldpcEncode`; bit errors counted as `L < 0` over all n bits instead of
   `biterr` over the k information bits.
4. `test_rl_mi_tanh.m` only: each check node's own state, `current_state(v)` instead of `current_state(i)`.
5. tanh-MI only: 40 state bins of width 0.05, as in training.
6. Ties broken uniformly at random with per-frame draws (Threefry seed 20261024, substream
   `(j-1)*100000 + f`), instead of the first index.
7. Check-node update via `ldpc_cluster_paper.m` (clipped numerics).
8. Decoding stops after an iteration that ends with a zero syndrome, instead of always running 5.
9. Per-frame outputs added (bit errors, iterations, messages, action sequence) and per-SNR BER/FER.

## `ldpc_cluster.m` → `ldpc_cluster_paper.m`

1. Check-node update with the clipped numerics of item 7 above; otherwise identical.

## New: `make_common_db.m`

Writes `train_db_wran.mat` (15,000 frames at 1, 2, 3, 4, 5, 5.5 dB, plus the per-step random draws) and
`eval_db_wran.mat` (100,000 noise vectors), used by the four scripts above.
