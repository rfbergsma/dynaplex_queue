# PPO for the Queue-Routing MDP — Handoff Document

*Written 2026-07-07 at the close of a three-day investigation (Claude Fable 5 + Ritsaart
Bergsma). Everything below is backed by committed code and reproducible runs; log
references are in `Log/probe_*.out` on Snellius and the session transcripts.*

---

## ADDENDUM 2026-07-12 (supersedes parts of the original below)

Full details and tables: `docs/ppo_queueing_report.tex` (binary chapter §1–4, QL
chapter §5). What changed since 2026-07-07:

1. **Critical bug found & fixed** (commit 3dfec38): a dangling-else introduced in
   535f033 silently clobbered `reward_type` to 1 for every run without
   `macro_feat=1`. **All runs between 2026-07-08 08:20 and the fix are invalid**
   (except `macro_feat=1` ones). The skip-all "catastrophe" of that morning was an
   artifact; retested clean, skip-all is safe but not beneficial (3/8 vs base 6/8),
   and it *breaks extraction, not learning* (stoch fine, argmax catastrophic).
2. **Controlled ablation matrix** (queue_ablation exe, 8 cells × 8 seeds): every
   recipe ingredient confirmed with mechanism-consistent failure signatures.
   base 6/8; dper 0/8 (→never-serve); vnorm 0/8 (exactly bimodal: never-serve /
   FIFO-clone); resets0 1/8; shape0 2/8; anneal0 3/8 with the argmax-vs-stoch gap
   made directly visible (stoch near-optimal, argmax bad).
3. **QL chapter (reward_type 1)**: binary recipe → bit-exact FIFO clone 8/8
   (heavy-tail variance, NOT extraction — the gap vanishes under QL). Working QL
   recipe: **average-reward + 128×2048 batch + robust-median/leaky-ratchet guard**
   (`avg=1 envs=128 rollout=2048 epochs=2 minibatch=4096 grobust=1 gleak=0.05`,
   1000 updates): 7/8 beat FIFO, best ≈ optimum. Discounted destabilizes at
   longer training (3/8 never-serve at 500u). rtype 3 = QL + urgency shaping
   exists, verified cost-neutral, empirically unnecessary.
4. **RVI benchmark repaired**: QL reference was 6% loose (learned policies beat
   it). `rvi_tol=0.001` probe knob; Exp2-QL ref 2492→2350 (binary refs were fine).
   Residual ~1–2% looseness remains. Also: **QL cost is NOT tick-rate invariant**
   (scales ~ν; binary is flat) — do not run cross-tick QL studies before fixing.
5. **Per-event action space implemented** (`action_mode=per_event`, probe
   `mode=pe`; commits 1f0bb69/8e91d9f/2935d48): each idle capacity unit picks a
   type or idles (valid_actions=n_jobs+1, strict masking). Equivalence gates
   PASSED: FIFO and never-serve bit-exact vs old mode; RVI g* matches on exp2/exp3
   (≤0.7%, solver tolerance). PPO smoke on binary Exp2/Exp3 in flight. This is
   open thread #1 of the original document, now motivated also by scaling
   (per-decision action space O(#types), credit concentration).
6. **Roadmap agreed with RB**: per-event PPO battery → tick-invariance fix +
   robustness grid (tick × due-date, frozen recipe, worst-cell scoring) → exp6
   scaling probe → reward_type 4 = fraction-served-on-time (≈ binary-shaped flux
   objective `c_n[1{FIL crosses d} + (λ_n/ν)1{FIL late}]`; needs reneging or an
   ε·QL hybrid against the abandonment degeneracy; design in session transcript
   2026-07-12).
7. Practicalities: GitHub push works from RB's machine again (patch-over-scp is
   fallback only); SURF rate-limits SSH (batch commands, few connections); Slurm
   time limits should be right-sized (~4h) or jobs pend overnight; DCL
   stochastic-base (ε=0.3) doubles gen-1 success under QL (4/8, three seeds
   sub-reference) but catastrophic collapses remain — DCL sampling-skew fix
   (period-anchored sampling) still open.

Superseded below: the aux-head A/B remains valid; the "Open threads" list is
replaced by item 6 above; grand tables remain valid for the original
(entropy=0.01, temp_min=0.25) recipe — the tuned recipe (entropy=0.03,
temp_min=0.1, rollout=512) achieves 7/8 within 6% and is the current default
reference, reproducible bit-exactly.

---

## Executive summary

**The headline: the optimal policy is learnable.** Starting from PPO collapsing to
"never serve" on 8/8 seeds (9.8× the optimum), we now reach RVI-optimal (±1–5%)
deterministic policies on both benchmark cells:

- **Exp3 (specialist+generalist): solved.** 7–8/8 seeds within 5% of optimal via plain
  argmax at 6000 updates (~10 min/seed on a shared genoa slice).
- **Exp2 (fully flexible, asymmetric cost): ~60–75% per-seed success** in every
  configuration tested (6000u, 30000u, ±aux-head); failures are a seed lottery, not
  specific bad seeds. Per *problem* it is effectively 100%: 2–3 seeds with
  evaluation-based selection always yields 1.01–1.05. The stochastic readout is
  near-optimal on ~7/8 seeds as a fallback.
- **DCL + shaping**: transformed from catastrophically unstable (up to 18× RVI on Exp3)
  to safe-but-conservative: lands optimal (~40–50% of seeds) or FIFO-copy, never
  catastrophic.
- **Large instance (6 types / 5 servers, RVI-intractable)**: recipe runs unmodified;
  1/3 seeds beat FIFO, 1/3 par, 1/3 collapsed. Untuned first attempt.

## The five diagnosed failure mechanisms (each with a fix in `master`)

1. **dper-clamp semi-MDP discounting bug** (`ppo.cpp`): intra-tick decisions (Δperiods=0)
   were discounted as if a period passed, subsidizing idleness most in the busiest
   states. Fix: `γ^0 = 1`. Evidence: Exp3 went from almost-all-collapse to 6/8 on this
   fix alone.
2. **Value-scale domination**: unnormalized value targets (O(100s)) let the value MSE
   crush the shared trunk. Fix: running return-std normalization.
3. **Persistent-env data poisoning**: environments were never reset; one bad excursion
   drove all 16 envs into the deep-late region, whose data starves the healthy states
   (their behavior then rots unrecoverably — the "late slide"). Fix: staggered resets
   (`env_reset_every=16`). A guard/brake cannot recover from this; only fresh-start
   data can.
4. **Argmax extraction mismatch ("re-presentation forgiveness")**: a skipped job is
   re-presented every tick, so with ~9 ticks of deadline slack a p=0.4 serve
   probability behaves like serving (P≈0.99 in time) — training never punishes
   wrong-sign near-tie logits, argmax makes them absolute. This defeated: temperature
   annealing (blind and guarded), best-sharp snapshots, and DCL-distillation (Q-label
   SNR collapses on a near-optimal base → returns an exact FIFO clone).
5. **Policy-gradient's advantage-weighted blindness**: PG updates scale with the
   advantage, so behaviorally-flat states never get their signs trained. Partial
   mitigations: more compute (30k updates: seeds that failed at 6k often succeed;
   52 min on genoa), aux advantage head (see below). Neither achieves 8/8 — the
   full fix is the action-space reformulation (see Open Threads).

## The recipe (all in `master`, defaults set accordingly)

PPO (`dp.GetPPO`) with:
- **`reward_type=2`** — potential-based shaping (urgency ramp `cost_rate·min(t,D)/D`
  per tick on the FIL, refunded on service). Provably and empirically cost-neutral
  (FIFO/RVI identical to 4 decimals under rtype 0 vs 2), so RVI stays a valid
  benchmark. **Load-bearing in every working configuration, for both PPO and DCL.**
- `env_reset_every=16` (staggered resets — default)
- `temp_anneal=1` guarded behavior-temperature annealing; guard watches the
  **per-period** reward rate (never per-decision: intra-tick zero-reward skips dilute
  that signal and reward collapse), with best-sharp snapshot restore
- `gamma=0.997`, 6000 updates for current problem sizes (0.99 plateaus at FIFO-copy)
- discounted mode (`average_reward=false`): the average-reward variant underperformed
  at 16×256 rollout sizes; likely viable only with much larger batches (Dai–Gluzman
  regime) — untested there
- readouts: argmax primary; evaluate stochastic (T=1) and serve-bias variants and
  select by evaluation (`GetReadoutPolicy(temperature, serve_bias, use_adv)`)

**Aux advantage head**: trained alongside (regression of taken-action GAE advantage).
Its own argmax readout is bad; as a training regularizer it rescued 3/3 locally but
the 8-seed A/B showed **no systematic improvement** (5/8 vs 5/8; it reshuffles the
lottery — seed 5 flipped good→bad). Keep it (harmless, occasionally decisive), but do
not rely on it. This A/B is a caution: single-seed rescues do not generalize; always
re-test at 8 seeds.

## Infrastructure (fast iteration)

- **`queue_dcl_probe`** — all knobs as `key=value` args (see header comment). One
  binary, parallel configs, no rebuild churn. Key knobs: `exp=2|3|6`, `reward`,
  `method=ppo|dcl`, `seeds=1,2,3`, `updates`, `gamma`, `bench=0|1|2` (0 skips the
  30-min RVI solve; 2 = FIFO-only for RVI-intractable cells), `salt=<n>` (**required
  for concurrent DCL jobs** — same-config DCL jobs race on the shared sample cache or
  silently share samples; salt perturbs the config hash for a private cache dir).
- **`bash/run_probe.slurm`** — 16-core shared-node runner: `sbatch bash/run_probe.slurm
  exp=2 seeds=4 ...`. Jobs backfill in minutes; 6000u ≈ 10–12 min, 30k ≈ 52 min.
  Reference values for `bench=0` runs: Exp2 RVI×Λ=40.50 / FIFO×Λ=70.46;
  Exp3 RVI×Λ=18.96 / FIFO×Λ=24.15.
- **`bash/run_exe.slurm`** — full-node runner for the older monolithic executables.
- Snellius: `ssh snellius` (see CLAUDE.md), repo at
  `~/queueing_project/dynaplex_queue`, logs in `Log/probe_<jobid>.out`.

## Scaling to larger instances (10 servers × 10 types)

**Do not carry over step counts — carry over time spans.** Two silent traps:
1. `rollout_steps` counts decisions; decisions-per-tick grows with system size, so a
   fixed rollout covers proportionally less physical time.
2. γ is per uniformization period; the effective time horizon is `1/((1−γ)·Λ)` and
   shrinks as tick_rate/Λ grow.

Recommendation: specify `horizon_time` and `rollout_time` in physical units and derive
`γ = 1 − 1/(horizon_time·Λ)` and `rollout_steps = rollout_time × E[decisions/time]`.
The overnight tick-rate experiment (see below) tests exactly this hypothesis at
tick ∈ {1.5, 6} with raw vs time-scaled hyperparameters.

Compute baseline: think in *simulated periods*, not updates. 6000 updates × 4096
decisions ≈ 25M decisions on Exp2. Budget larger instances proportionally, and prefer
3 seeds × 1× budget over 1 seed × 3× budget (the lottery dominates the scaling curve).

## Overnight runs in flight (results land in Log/probe_*.out by morning)

- **Sensitivity sweep** (Exp2@6000, 3 seeds each): γ ∈ {0.99, 0.997, 0.999},
  λ ∈ {0.90, 0.95, 0.99}, rollout ∈ {128, 256, 512}, resets ∈ {8, 16, 64},
  entropy ∈ {0.005, 0.01, 0.03}, lr ∈ {1e-4, 3e-4, 1e-3}, temp_min=0.1,
  temp_anneal=0 (ablation).
- **Tick-rate/time-scaling test** (bench=1, self-contained ratios):
  tick=1.5 {γ=.997/r256 raw, γ=.994/r128 scaled}, tick=6 {γ=.997/r256 raw,
  γ=.9985/r512 scaled}.

## Open threads, in priority order

1. **Action-space reformulation** ("which class to serve, or idle" per freed server per
   event — one decision per event, skip = real commitment). This eliminates
   re-presentation forgiveness *by construction* and is the principled fix for the
   Exp2 seed lottery. Deliberately deferred (invasive; touches the whole paper). The
   experimental case for it is now airtight.
2. **Large-instance tuning**: exp=6 works 1/3 untuned; apply the time-scaled
   hyperparameters first.
3. **Average-reward PPO at large batch sizes** (envs≥128, rollout≥2048) — the
   Dai–Gluzman regime; our small-batch failure doesn't condemn it.
4. Housekeeping: rename `is_rfq_winner`→`is_newest_winner`; Exp2 small-D RVI
   truncation issue (dsweep artifact: RVI_L > FIFO_L at D<9; recurs at tick=1.5).

## Design note: `action_sort` (correction from RB — this was NOT a bugfix)

The ascending candidate order (`action_sort="reverse_fifo"`) is a deliberate design that
pairs with the *enforced-FIFO* base policy (1 only on the FIFO winner, explicit 0
elsewhere): non-winners are presented first, the winner last.  Intent: DCL's one-step
deviations then cleanly cover both counterfactuals every tick — "serve this
cμ-alternative instead" (at each non-winner) and "serve vs fully idle" (at the winner,
isolating strategic idleness as a single decision point).  Likely failure causes:
0-label imbalance (a classifier answering "0 everywhere" ≈ never-serve attractor and is
barely punished by aggregate loss) and skip-labels being unlearnable without
queue-position features (the later `is_*_winner` labels were the missing half — worth
retrying the combination).  **Compatibility rule: never combine ascending order with
GREEDY FIFO** (greedy + ascending = accept the newest head = LIFO).  Coherent pairs:
greedy-FIFO ↔ descending (default), enforced-FIFO ↔ ascending.

## Grand result tables

Exp2, argmax NN×Λ vs RVI=40.50 (✗ = >2×):

| Seed | 6000u | 6000u+aux | 30k |
|---|---|---|---|
| 1 | 44.78 | 40.80 | 40.85 |
| 2 | ✗ | 41.99 | 41.73 |
| 3 | 44.81 | 53.27 | 60.84 |
| 4 | ✗ | 42.00 | 41.58 |
| 5 | 40.81 | ✗ | 40.78 |
| 6 | 40.92 | 41.59 | 42.37 |
| 7 | 40.90 | 40.89 | 41.89 |
| 8 | ✗ | 57.48 | ✗ |

Exp3@6000+aux, argmax NN×Λ vs RVI=18.96: 19.12, 19.88, 19.16, 18.97, 19.37, 18.95,
19.09, 21.55 (7/8 within 5%).

DCL+shaping (independent seeds, gen-1 from FIFO): Exp2 {75.1, 41.4, 41.7, 70.4, 40.9,
70.4, 44.4, 70.4}; Exp3 {24.1, 31.6, 18.9, 24.1, 19.0, 24.1, 24.1, 19.1}.

Large instance (exp=6, vs FIFO×Λ=2.36): 2.54, 42.8✗, 2.32.
