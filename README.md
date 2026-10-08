# Cookie Cats: should the first gate move from level 30 to 40? (real-data A/B test)

![data](https://img.shields.io/badge/data-real%20(Kaggle)-success) ![Python](https://img.shields.io/badge/Python-pandas%20%7C%20SciPy-blue) [![Kaggle](https://img.shields.io/badge/Kaggle-notebook-20BEFF)](https://www.kaggle.com/code/kkchillbro/cookie-cats-a-b-test-gate-30-vs-40)

**TL;DR.** Moving the first gate from level 30 to level 40 **lowered 7-day retention by 0.82 pp** (19.02% → 18.20%, −4.3% relative, 95% CI −1.33 … −0.31 pp, p = 0.0016). Recommendation: **keep the gate at level 30.**

90,189 real players of the mobile puzzle game *Cookie Cats* ([dataset](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing)). Unlike my other portfolio projects, this one uses **real data**, with its outliers, an imperfect split and an effect against the team's hypothesis.

## Results

| metric | gate_30 (control) | gate_40 | difference (95% CI) | p-value |
|---|---|---|---|---|
| **7-day retention** (primary) | 19.02% | 18.20% | **−0.82 pp** (−1.33 … −0.31) | 0.0016 |
| 1-day retention | 44.82% | 44.23% | −0.59 pp (−1.24 … +0.06) | 0.074 |
| median game rounds, 14 days | 17 | 16 | −1 (−1 … 0) | 0.051 (Mann-Whitney) |

The test could detect effects of 0.74 pp on 7-day retention at 80% power, so the observed −0.82 pp is not a lucky fluke of an underpowered test.

## What the analysis checks

| step | what I found | why it matters |
|---|---|---|
| Duplicates, missing values | none | |
| Outliers | one player with 49,854 rounds in 14 days (17× the next one), excluded from the rounds analysis only | one bot would distort mean engagement |
| Sample ratio mismatch | 44,700 vs 45,489 (49.6 / 50.4), χ² p = 0.0086 | below the usual alarm (p < 0.001) but worth asking engineering about |
| Retention | z-test with CI + bootstrap | normal approximation confirmed by resampling |
| Engagement | Mann-Whitney + bootstrap of the median | rounds are heavy-tailed, the t-test on means is misleading |
| Power / MDE | 0.74 pp for 7-day retention | is the test sensitive enough? |
| Multiple testing | Bonferroni α = 0.025 for 2 retention metrics | 7-day result survives |

## Limitations
- Day-30 retention and revenue are not in the data: the revenue impact is inferred from retention.
- 63% of players never reach round 30, so the change affects a minority. Restricting the analysis to players who reached the gate would introduce selection bias after treatment, so I deliberately did not do it.
- The mild SRM means the assignment mechanism should be verified before acting on effects this small.

## Files

- [`cookie_cats_ab_test.ipynb`](cookie_cats_ab_test.ipynb): the notebook (run it on [Kaggle](https://www.kaggle.com/code/kkchillbro/cookie-cats-a-b-test-gate-30-vs-40) with outputs and charts)
- [`build_notebook.py`](build_notebook.py): source of the notebook cells, so diffs are readable in git

Run locally: download `cookie_cats.csv` from Kaggle, then `COOKIE_CATS_CSV=path/to/cookie_cats.csv jupyter notebook`.
