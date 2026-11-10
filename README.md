# Walk on Cylinders vs Walk on Heat Balls

Code and experiments for the paper *<full paper title>* (<authors>, <year>).

The code compares two Monte Carlo methods for the heat equation $\partial_t u = \tfrac12 \Delta u$ in bounded domains: **Walk on Cylinders (WoC)** and **Walk on Heat Balls (WoHB)**. The $\tfrac12\Delta$ convention is used throughout (WoHB heat-ball constant $e/d$, space tolerance $\epsilon_1 = \sqrt{\epsilon_0 d / e}$ when the tolerances are coupled).

## Setup

Python 3.12.6

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name walk-on-cylinders
```

The figures use LaTeX text. If no LaTeX installation is found (or it cannot render the test string), the notebook falls back to matplotlib's own math rendering.

## Running the experiments

Open `notebooks/woc_wohb_s_ranges_domains_refactored_reproducible.ipynb`, select the `walk-on-cylinders` kernel and run all cells from the top (Restart & Run All).

- `FAST_MODE = True` (parameters cell) runs everything with tiny sample sizes and coarse grids as an end-to-end check. Its output goes to `results_fast/` and `figures_fast/` and is not for the paper.
- All settings that can change a result are in the *Global parameters* cell. Every random number is derived from `MASTER_SEED`, so a re-run reproduces the same numbers.
- Results are written to `results/` and figures to `figures/` (relative to the folder the notebook runs in, i.e. `notebooks/`). Each plotting cell reads the saved results, so figures can be regenerated without re-simulating by setting `RECOMPUTE[n] = False`. Experiment 4 resumes from its checkpoint CSV.
- `TOL_TIME_MODE = 'eps'` uses the time tolerance $\epsilon_0 = \epsilon$ (the same as WoHB); `'fixed'` uses $\epsilon_0 = 10^{-8}$ for every WoC run.

Expected runtime: <fill in after the full run, with the hardware used>.

## Experiments

| Experiment | What it shows | Saved results | Figures |
|---|---|---|---|
| 1 | Solution over time, WoC ($\theta$ sweep) vs WoHB vs exact | `exp1_results.pkl` | `experiment1_*.pdf` |
| 2 | Step distributions, space-exit proportions, steps vs $\theta$ (with the small-$\theta$ sweep) | `exp2_results.pkl` | `experiment2_*.pdf` |
| 3 | Steps vs tolerance $\epsilon$ (coupled $\epsilon_0$, $\epsilon_1$) and vs initial position | `exp3_results.pkl` | `experiment3_*.pdf` |
| 4 | Dimension scaling | `woc_wohb_dim_scaling_results.csv` | `plot_*.pdf` |
| 5 | Effect of the time tolerance $\epsilon_0$, bounded vs unbounded | `woc_time_tolerance_results.csv` | `experiment5_*.pdf` |

<Add a column or a list mapping these to the figure and table numbers in the paper.>

## Layout

```
shape_methods/
    shared/               domains (BoxDomain, HalfSphereDomain, ...), PathResult
    walk_on_cylinders/    WoC class, distribution tables, Bessel zeros, theta quantiles
    walk_on_heat_balls/   WoHB class
notebooks/                the experiment notebook
tests/                    (to be added)
paper_results/            final results and figures cited in the paper (optional)
```

## Choice of the tabulation scale $\theta$

$\theta_p(d)$ is the root of $H_\theta = p$ for $p = 0.1, 0.5, 0.9$ (expected steps $1/(1-p)$ in $\mathbb{R}^d$), computed from the Bessel series in `theta_quantiles.py` (normal approximation above $d = 200$). $\theta = e/d$ and a large $\theta$ are optional extras, switched by `INCLUDE_E_OVER_D` and `INCLUDE_THETA_LARGE`.

## Citation and license

<Citation / Zenodo DOI after the release is archived.>

<License: choose with your advisor.>
