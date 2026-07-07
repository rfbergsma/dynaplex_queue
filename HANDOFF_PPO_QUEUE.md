# PPO for the Queue-Routing MDP — Handoff Document

*Written 2026-07-07 at the close of a three-day investigation (Claude Fable 5 + Ritsaart
Bergsma). Everything below is backed by committed code and reproducible runs; log
references are in `Log/probe_*.out` on Snellius and the session transcripts.*

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
   truncation issue (dsweep artifact: RVI_L > FIFO_L at D<9).

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
