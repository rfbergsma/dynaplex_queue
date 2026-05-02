"""
rvi_tutorial.py — Step-by-step Relative Value Iteration
========================================================
System: 1 server, 1 job type, FIL truncated at M.

Key idea:
  We want g* = long-run average cost per step.
  Standard value iteration diverges for average-cost problems.
  RVI fixes this by subtracting h[reference_state] after every
  Bellman update — this "anchors" the values and prevents drift.
  The amount subtracted each iteration converges to g*.
"""

import numpy as np

# ─── Parameters ───────────────────────────────────────────────────────────────
lam   = 0.3    # arrival rate  (λ)
mu    = 0.5    # service rate  (μ)
gamma = 1.0    # tick rate     (γ) — the due-time clock
c     = 1.0    # cost per tick while FIL > due_time
due   = 1      # due_time threshold
M     = 3      # FIL truncation cap

Lam   = lam + mu + gamma   # uniformization rate = 1.8

print("=" * 65)
print("SYSTEM")
print("=" * 65)
print(f"  λ={lam}  μ={mu}  γ={gamma}  Λ={Lam}")
print(f"  cost rate={c}  due_time={due}  FIL cap M={M}")

# ─── State Space ──────────────────────────────────────────────────────────────
#
#  State = (fil, busy)
#    fil  ∈ {-1, 0, 1, 2, 3}   (-1 = no jobs in queue)
#    busy ∈ {False, True}
#
#  AwaitAction: fil ≥ 0 AND NOT busy  → server is free, job is waiting → must decide
#  AwaitEvent:  fil == -1 OR busy     → waiting for the next random event
#
#  NOTE: we assume action=1 (assign) is always taken at AwaitAction states.
#        This is optimal here (never worth idling a server with a waiting job).

states = [(-1, False), (-1, True)]
for f in range(M + 1):
    states += [(f, False), (f, True)]   # AwaitAction, AwaitEvent

sidx = {s: i for i, s in enumerate(states)}
N    = len(states)

def is_action(s): return s[0] >= 0 and not s[1]
def label(s):
    f, b = s
    cat = "AwaitAction" if is_action(s) else "AwaitEvent "
    return f"(FIL={f:>2}, {'busy' if b else 'idle'}) {cat}"

# ─── Immediate costs ──────────────────────────────────────────────────────────
#
#  Cost only at AwaitEvent states where FIL > due_time.
#  Scaled by γ/Λ because in the uniformized chain, a "tick" event
#  (which triggers the cost) fires with probability γ/Λ.
#
imm = {s: (c * gamma / Lam if (not is_action(s) and s[0] > due) else 0.0)
       for s in states}

print("\n" + "=" * 65)
print("STATES  (immediate cost only at AwaitEvent where FIL > due_time)")
print("=" * 65)
print(f"  {'#':>2}  {'state + category':>35}  {'imm_cost':>10}")
print("  " + "-" * 52)
for i, s in enumerate(states):
    marker = "  ← costs here!" if imm[s] > 0 else ""
    print(f"  {i:>2}  {label(s):>35}  {imm[s]:>10.4f}{marker}")

# ─── Next-FIL distribution ────────────────────────────────────────────────────
#
#  When a job is assigned (action=1), the server starts processing.
#  The FIL of the NEXT waiting job is unknown — it depends on how long
#  that job has been waiting behind the front-of-line job.
#
#  Geometric model: P(new_FIL = j) ∝ (γ/(λ+γ))^(fil-j) × λ/(λ+γ)
#  i.e. each "slot" behind the front job is occupied with prob λ/(λ+γ).
#
def next_fil_dist(fil):
    if fil <= 0:
        return [(-1, 1.0)]    # only 1 job was waiting → queue becomes empty
    alpha = lam / (lam + gamma)
    beta  = gamma / (lam + gamma)
    dist  = [(j, beta**(fil - j) * alpha) for j in range(1, fil + 1)]
    dist += [(-1, beta**fil)]
    total = sum(p for _, p in dist)
    return [(f, p / total) for f, p in dist]

# ─── Transition matrix ────────────────────────────────────────────────────────
#
#  P[i][j] = probability of moving from state i to state j.
#
#  AwaitEvent transitions (uniformized CTMC):
#    • Arrival  (only if FIL=-1): rate λ  → FIL becomes 0
#    • Service completion (busy) : rate μ  → server becomes free, FIL unchanged
#    • Tick                      : rate γ  → FIL += 1 (capped at M); FIL=-1 → self-loop
#    • "Nothing" self-loop       : remaining probability (uniformization padding)
#
#  AwaitAction transitions (action=1 = assign):
#    • Assign job: server becomes busy, new FIL sampled from next_fil_dist(fil)
#
P = np.zeros((N, N))

for i, s in enumerate(states):
    f, b = s

    if is_action(s):
        # Action = 1: assign the waiting job
        for new_f, p in next_fil_dist(f):
            nf = min(new_f, M) if new_f >= 0 else -1
            P[i, sidx[(nf, True)]] += p

    else:  # AwaitEvent
        rate = 0.0
        # Arrival (only when queue is empty)
        if f == -1:
            P[i, sidx[(0, b)]] += lam / Lam
            rate += lam
        # Service completion
        if b:
            P[i, sidx[(f, False)]] += mu / Lam
            rate += mu
        # Tick
        new_f = min(f + 1, M) if f >= 0 else -1
        P[i, sidx[(new_f, b)]] += gamma / Lam
        rate += gamma
        # Self-loop (uniformization remainder)
        P[i, i] += max(0.0, 1.0 - rate / Lam)

# Sanity check
assert np.allclose(P.sum(axis=1), 1.0), "Rows don't sum to 1!"

# ─── RVI ──────────────────────────────────────────────────────────────────────
#
#  Bellman equation for average cost:
#
#    h(s) + g* = imm(s) + Σ_j P(j|s) · h(j)    [AwaitEvent]
#    h(s) + g* = min_a { Σ_j P(j|s,a) · h(j) }  [AwaitAction]
#
#  Rearranged as an iterative update:
#
#    h_new(s) = imm(s) + Σ_j P(j|s) · h(j)       [AwaitEvent]
#    h_new(s) = Σ_j P(j|s, a=1) · h(j)            [AwaitAction, action=1 optimal]
#
#  After each iteration:
#    g*     = h_new[ref]      ← the reference state's value IS the g* estimate
#    h_new -= g*              ← subtract to keep h bounded ("relative" trick)
#
ref = sidx[(-1, False)]   # reference state = S0 (empty, idle)
h   = np.zeros(N)

print("\n" + "=" * 65)
print("RVI ITERATIONS")
print("=" * 65)

MAX_ITER  = 300
EPS       = 1e-9
g_prev    = 0.0
g_stable  = 0

for it in range(1, MAX_ITER + 1):

    h_new = np.zeros(N)
    for i, s in enumerate(states):
        if is_action(s):
            h_new[i] = P[i] @ h          # action=1 (always assign here)
        else:
            h_new[i] = imm[s] + P[i] @ h

    g_star = h_new[ref]   # g* estimate = value of ref state BEFORE subtraction
    h_new -= g_star        # anchor: subtract g* from every state

    delta = float(np.max(np.abs(h_new - h)))
    h     = h_new

    # ── Verbose output for first 3 iterations ────────────────────────────────
    if it <= 3:
        print(f"\n{'─'*65}")
        print(f"Iteration {it}  {'(h starts at all-zeros)' if it==1 else ''}")
        print(f"{'─'*65}")
        print(f"  {'#':>2}  {'state':>35}  {'imm':>6}  {'Σ P·h_prev':>10}  {'h_new(before sub)':>18}  {'h[i]':>8}")
        print(f"  {'':>2}  {'':-<35}  {'':->6}  {'':->10}  {'':->18}  {'':->8}")
        for i, s in enumerate(states):
            expected = float(P[i] @ (h + g_star))   # h before subtraction
            raw      = imm[s] + expected if not is_action(s) else expected
            print(f"  {i:>2}  {label(s):>35}  {imm[s]:>6.3f}  {expected:>10.4f}  {raw:>18.4f}  {h[i]:>8.4f}")
        print(f"\n  g* = h_new[S{ref}] (before sub) = {g_star:.6f}  → subtracted from all entries")
        if it == 1:
            print(f"\n  ↑ Notice: since h_prev = 0, every h_new = imm_cost + 0 = imm_cost.")
            print(f"    Only states S{sidx[(-1,True)]} and S{sidx[(1,True)]}+ with FIL>due are nonzero.")
            print(f"    g* = 0 here because the reference state S{ref} has imm_cost=0.")

    elif it <= 10 or it % 25 == 0:
        print(f"  iter {it:>4}:  g* = {g_star:.8f}   delta = {delta:.2e}")

    # ── Convergence ────────────────────────────────────────────────────────
    if abs(g_star - g_prev) < EPS:
        g_stable += 1
        if g_stable >= 5:
            print(f"\n  Converged at iteration {it}")
            break
    else:
        g_stable = 0
    g_prev = g_star

# ─── Final results ────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"RESULT")
print(f"{'='*65}")
print(f"  g* = {g_star:.8f}  (optimal average cost per uniformized step)")
print(f"  g* × Λ = {g_star * Lam:.8f}  (cost per unit continuous time)")
print()
print("  Final relative values h (= how much worse is each state vs. average):")
print(f"  {'#':>2}  {'state':>35}  {'h[i]':>10}")
print("  " + "-" * 52)
for i, s in enumerate(states):
    print(f"  {i:>2}  {label(s):>35}  {h[i]:>10.4f}")
