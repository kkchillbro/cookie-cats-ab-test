"""Builds cookie_cats_ab_test.ipynb from the cells below (no nbformat dependency)."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md("""
# Cookie Cats: should the first gate move from level 30 to level 40?

**Real data A/B test** (90,189 players, [Kaggle dataset](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing)).
In *Cookie Cats* a gate forces players to wait or pay. The test moved the first gate from level 30 (`gate_30`, control) to level 40 (`gate_40`, treatment).

**Decision to make:** keep the gate at 30 or move it to 40.

| | |
|---|---|
| Hypothesis | A later gate gives players more uninterrupted play, so retention goes **up** |
| Primary metric | 7-day retention (`retention_7`): long-term engagement drives revenue |
| Secondary | 1-day retention, game rounds in the first 14 days |
| Guardrail / validity | sample ratio mismatch, outliers, duplicates |

Plan: data checks → SRM → retention (z-test + bootstrap) → engagement (heavy tails: Mann-Whitney + bootstrap of the median) → power / MDE → decision → limitations.
""")

code("""
import glob, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})
RNG = np.random.default_rng(42)

path = os.environ.get("COOKIE_CATS_CSV") or glob.glob("/kaggle/input/**/cookie_cats.csv", recursive=True)[0]
df = pd.read_csv(path)
print(df.shape)
df.head()
""")

md("## 1. Data checks")

code("""
print("duplicated userid:", df.userid.duplicated().sum())
print("missing values:", int(df.isna().sum().sum()))
print(df.version.value_counts())
df.groupby("version").sum_gamerounds.describe(percentiles=[.5, .9, .99, .999]).round(1)
""")

code("""
# One extreme player distorts means: look at the top of the distribution
top = df.sort_values("sum_gamerounds", ascending=False).head(5)
print(top[["userid", "version", "sum_gamerounds"]])
p999 = df.sum_gamerounds.quantile(0.999)
print(f"99.9th percentile: {p999:.0f} rounds")
""")

md("""
One player has ~50k rounds in 14 days, more than 10× the next player and impossible for a human (a bot or a logging error).
I **exclude only this single record** from the rounds analysis and keep everyone in retention, because a binary flag is not affected by the outlier.
Users with 0 rounds installed but never played; they stay in: they were randomized, removing them would bias the comparison.
""")

code("""
outlier_ids = df.loc[df.sum_gamerounds > 10 * df.sum_gamerounds.drop(df.sum_gamerounds.idxmax()).max(), "userid"]
clean = df[~df.userid.isin(outlier_ids)].copy()
print("removed outliers:", len(outlier_ids), "| zero-round players:", (df.sum_gamerounds == 0).mean().round(3))
""")

md("## 2. Sample ratio mismatch (SRM)\n\nThe split was meant to be 50/50. A chi-square test checks whether the observed split could be chance.")

code("""
n = df.version.value_counts().reindex(["gate_30", "gate_40"])
chi2, p_srm = stats.chisquare(n.values)
print(n.to_dict())
print(f"shares: {(n / n.sum()).round(4).to_dict()} | chi2 = {chi2:.2f}, p = {p_srm:.4f}")
""")

md("""
**Interpretation.** The standard SRM alarm is p < 0.001; here p is between 0.001 and 0.01: a ~0.9 pp imbalance.
It is not strong enough to invalidate the test, but in a real team I would ask engineering how users were assigned
(e.g., were some app versions or countries excluded from one arm) before trusting small effects.
""")

md("## 3. Retention: z-test, confidence interval and bootstrap")

code("""
def ztest(x_c, n_c, x_t, n_t, alpha=0.05):
    p_c, p_t = x_c / n_c, x_t / n_t
    p = (x_c + x_t) / (n_c + n_t)
    z = (p_t - p_c) / np.sqrt(p * (1 - p) * (1 / n_c + 1 / n_t))
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    q = stats.norm.ppf(1 - alpha / 2)
    d = p_t - p_c
    return dict(control=p_c, treatment=p_t, diff_pp=100 * d, rel=d / p_c,
                ci_low_pp=100 * (d - q * se), ci_high_pp=100 * (d + q * se), p_value=2 * stats.norm.sf(abs(z)))

rows = []
for metric in ["retention_1", "retention_7"]:
    g = df.groupby("version")[metric].agg(["sum", "count"])
    r = ztest(g.loc["gate_30", "sum"], g.loc["gate_30", "count"], g.loc["gate_40", "sum"], g.loc["gate_40", "count"])
    rows.append({"metric": metric, **r})
res = pd.DataFrame(rows).set_index("metric")
res.style.format({"control": "{:.2%}", "treatment": "{:.2%}", "diff_pp": "{:+.2f}", "rel": "{:+.1%}",
                  "ci_low_pp": "{:+.2f}", "ci_high_pp": "{:+.2f}", "p_value": "{:.4f}"})
""")

code("""
# Bootstrap of the difference (gate_40 - gate_30), as a robustness check of the normal approximation
def boot_diff(metric, n_boot=5000):
    a = df.loc[df.version == "gate_30", metric].to_numpy()
    b = df.loc[df.version == "gate_40", metric].to_numpy()
    ma = RNG.binomial(len(a), a.mean(), n_boot) / len(a)
    mb = RNG.binomial(len(b), b.mean(), n_boot) / len(b)
    return 100 * (mb - ma)

fig, ax = plt.subplots(figsize=(8, 3.5))
for metric, color in [("retention_1", "#60a5fa"), ("retention_7", "#dc2626")]:
    d = boot_diff(metric)
    ax.hist(d, bins=60, alpha=0.6, color=color, label=f"{metric}: P(gate_40 worse) = {(d < 0).mean():.1%}")
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("difference gate_40 − gate_30, pp")
ax.set_title("Bootstrap distribution of the retention difference")
ax.legend(frameon=False)
plt.tight_layout(); plt.show()
""")

md("## 4. Engagement: game rounds (heavy-tailed)")

code("""
g30 = clean.loc[clean.version == "gate_30", "sum_gamerounds"].to_numpy()
g40 = clean.loc[clean.version == "gate_40", "sum_gamerounds"].to_numpy()
u, p_mw = stats.mannwhitneyu(g40, g30, alternative="two-sided")

boot_med = np.array([np.median(RNG.choice(g40, len(g40))) - np.median(RNG.choice(g30, len(g30))) for _ in range(2000)])
boot_mean = np.array([RNG.choice(g40, len(g40)).mean() - RNG.choice(g30, len(g30)).mean() for _ in range(2000)])
print(f"mean rounds: gate_30 {g30.mean():.2f} | gate_40 {g40.mean():.2f}")
print(f"median rounds: gate_30 {np.median(g30):.0f} | gate_40 {np.median(g40):.0f}")
print(f"Mann-Whitney p = {p_mw:.4f}")
print("95% CI median diff:", np.percentile(boot_med, [2.5, 97.5]).round(2))
print("95% CI mean diff:  ", np.percentile(boot_mean, [2.5, 97.5]).round(2))
""")

code("""
fig, ax = plt.subplots(figsize=(8, 3.5))
bins = np.arange(0, 101, 2)
for arr, name, color in [(g30, "gate_30", "#2563eb"), (g40, "gate_40", "#dc2626")]:
    ax.hist(arr, bins=bins, histtype="step", lw=2, color=color, label=name, density=True)
ax.axvline(30, color="#2563eb", ls="--", lw=1); ax.axvline(40, color="#dc2626", ls="--", lw=1)
ax.set_xlabel("game rounds in first 14 days (truncated at 100)")
ax.set_title("Most players stop before reaching either gate")
ax.legend(frameon=False); plt.tight_layout(); plt.show()
print("share of players who never reach round 30:", (clean.sum_gamerounds < 30).mean().round(3))
""")

md("## 5. Power: what effect could this test detect?")

code("""
def mde(p, n_per_group, alpha=0.05, power=0.8):
    return (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * np.sqrt(2 * p * (1 - p) / n_per_group)

n_g = n.min()
for metric in ["retention_1", "retention_7"]:
    base = res.loc[metric, "control"]
    print(f"{metric}: MDE ≈ {100 * mde(base, n_g):.2f} pp (relative {mde(base, n_g) / base:.1%}) at 80% power")
""")

md("""
## 6. Decision

| metric | gate_30 | gate_40 | difference (95% CI) | p-value |
|---|---|---|---|---|
| **7-day retention** (primary) | 19.02% | 18.20% | **−0.82 pp** (−1.33 … −0.31) | 0.0016 |
| 1-day retention | 44.82% | 44.23% | −0.59 pp (−1.24 … +0.06) | 0.074 |
| median rounds (14 days) | 17 | 16 | −1 (−1 … 0) | 0.051 (Mann-Whitney) |

- **7-day retention is 0.82 pp (−4.3% relative) lower with the gate at level 40.** The confidence interval excludes zero, the bootstrap agrees, and the effect is larger than the test's MDE (0.74 pp), so the test had enough power to see it.
- 1-day retention points the same way but is not significant at α = 5%.
- Engagement barely moves: 63% of players never reach round 30, so most users never see either gate.

**Recommendation: keep the gate at level 30.** Moving it to 40 loses about 1 in 23 of the players who would still be playing after a week, and long-term players are the source of in-app revenue.

### Why might a later gate *hurt*? (hypotheses for the next test)
- *Hedonic adaptation*: a forced break makes players come back fresher; without it they burn out faster.
- At level 30 the break hits at the moment of peak engagement, creating a reason to return.

## 7. Limitations
- **Mild SRM** (p = 0.0086): a 49.6 / 50.4 split. Not enough to discard the test, but the assignment mechanism should be checked before trusting effects this small.
- Two retention metrics were tested. With a Bonferroni correction (α = 0.025) the 7-day result still holds (p = 0.0016).
- Retention is a flag on day 1 and day 7, not a curve; day-30 retention and revenue are not in the data, so the revenue impact is an inference.
- Only 37% of players reach level 30, so the treatment touches a minority. Restricting the analysis to players who reached the gate would bias it (selection after treatment), so it is deliberately not done.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("cookie_cats_ab_test.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("written", out)
