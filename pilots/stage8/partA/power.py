"""Power for the Part A criteria under article clustering (35 clusters), simulated sampling distributions."""
import numpy as np
rng = np.random.default_rng(0)
def sim(n, muK, muV, cvK, cvV, icc=0.15, nclu=35, sims=4000):
    out = []
    for _ in range(sims):
        clu = rng.integers(0, nclu, n)
        eK = rng.normal(0, 1, nclu)[clu] * np.sqrt(icc) + rng.normal(0, 1, n) * np.sqrt(1 - icc)
        eV = rng.normal(0, 1, nclu)[clu] * np.sqrt(icc) + rng.normal(0, 1, n) * np.sqrt(1 - icc)
        K = muK + abs(muK if muK else 1) * cvK * eK if muK else 0.3 * muV * cvK * eK
        V = muV + muV * cvV * eV
        out.append(K.mean() / (K.mean() + V.mean()))
    return np.array(out)
print("A1: OPTIONS-AFTER, criterion s_hat >= 0.5 and s_hat - 1.96 SE >= 0.35 (per model; ^4 for all four)")
for s in (0.5, 0.55, 0.6, 0.7):
    for n in (120, 150, 200):
        T = 20.0; d = sim(n, s * T, (1 - s) * T, 1.0, 0.7)
        se = d.std(); p = np.mean((d >= 0.5) & (d - 1.96 * se >= 0.35))
        print(f"  true s={s:.2f} n={n}: SE={se:.3f} power={p:.2f} all-4={p**4:.2f}")
print("A2: NO-MENTION, criterion s_hat <= 0.2 and s_hat + 1.96 SE <= 0.3")
for s in (0.05, 0.1, 0.15):
    for n in (120, 150):
        T = 15.0; d = sim(n, s * T, (1 - s) * T, 3.0, 0.5)
        se = d.std(); p = np.mean((d <= 0.2) & (d + 1.96 * se <= 0.3))
        print(f"  true s={s:.2f} n={n}: SE={se:.3f} power={p:.2f} all-4={p**4:.2f}")
print("A5: MENTION-AFTER minus NO-MENTION >= 0.1 with CI excluding 0 (independent approx.)")
for s in (0.15, 0.2, 0.3):
    for n in (120, 150):
        T = 12.0; d = sim(n, s * T, (1 - s) * T, 1.5, 0.6); d0 = sim(n, 0.03 * T, 0.97 * T, 3.0, 0.5)
        diff = d[:len(d0)] - d0; se = diff.std(); p = np.mean((diff >= 0.1) & (diff - 1.96 * se > 0))
        print(f"  true s={s:.2f} n={n}: SE(diff)={se:.3f} power={p:.2f} all-4={p**4:.2f}")
print("A6: paired flip-rate difference >= 0.3 with CI excluding 0")
for d_true, disc in ((0.4, 0.5), (0.5, 0.55), (0.35, 0.45)):
    for n in (120, 150):
        # discordant fraction disc, p10 - p01 = d_true
        p10 = (disc + d_true) / 2; p01 = (disc - d_true) / 2
        x = rng.multinomial(n, [p10, p01, 1 - p10 - p01], size=20000)
        dh = (x[:, 0] - x[:, 1]) / n; se = np.sqrt((x[:, 0] + x[:, 1]) / n - dh**2) / np.sqrt(n) * 1.3
        p = np.mean((dh >= 0.3) & (dh - 1.96 * se > 0))
        print(f"  true diff={d_true:.2f} discordant={disc:.2f} n={n}: power={p:.2f} all-4={p**4:.2f}")
