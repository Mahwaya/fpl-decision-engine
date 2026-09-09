# FPL Decision Engine

A data-driven decision support system for Fantasy Premier League: predicts player
points, picks the optimal squad under FPL's rules, recommends a captain, and alerts
on team-news changes before the deadline.

Built as a personal project applying MSc Data Science skills to a real weekly decision.
**Zero third-party dependencies beyond numpy / pandas / scikit-learn / scipy** — no
PuLP, no XGBoost, no scraping frameworks.

---

## Headline results

Measured on the **2025-26 season**, held out entirely from training (trained on
2022-23 → 2024-25, 74,440 rows; tested on 26,320 rows the model never saw).

| Model | MAE | High-return MAE | **Top-20 picks: mean actual points** |
|---|---|---|---|
| `xP` — FPL's own expected points | 1.115 | 6.534 | 3.29 |
| `direct` | 1.024 | **5.110** | 4.29 |
| **`two_stage` (champion)** | **0.954** | 5.305 | **4.39** |

**Both models beat FPL's own published expected-points model.** On the
decision-relevant ranking metric the champion delivers **4.39 points per pick vs
FPL's 3.29 — about +1.1 per pick (~33% better)**.

### Captaincy is worth more than squad selection

Simulated across all 38 gameweeks of 2025-26, captain chosen each week from the 50
most-owned players:

| Strategy | pts/GW | over 38 GWs |
|---|---|---|
| Most-owned (the template pick) | 9.63 | 366 |
| `xP` (FPL's own) | 13.11 | 498 |
| **`two_stage` (champion)** | **13.63** | **518** |
| `p_haul` — P(points ≥ 10) | 11.58 | 440 |
| `q90` — 90th-percentile ceiling | 9.21 | 350 |

**+4.00 pts/GW over the template ≈ +152 points per season.**

> **A common belief this project tested and disproved.** FPL wisdom says captaincy is
> a *ceiling* decision — pick the explosive player, not the highest average. Both
> ceiling strategies lost, and `q90` was the worst tested. The reason is simple:
> captain points double **linearly**, so maximising `E[2X]` is just maximising `E[X]`.
> Expected value is provably the right criterion. The ceiling framing only applies
> when chasing *rank* in a mini-league, where variance is deliberately desirable.

### Minutes dominate everything

Permutation importance on the direct model:

```
minutes_last        0.1806   <- 4x the next feature
minutes_m3          0.0417
ict_last            0.0343
ict_m3              0.0329
total_points_m10    0.0325
minutes_m5          0.0316
```

Four of the top ten features are minutes-based. Across 113,582 historical rows,
**27.6% of player-fixtures are 60+ minutes and they hold 84.9% of all points.**
This is why the news/availability layer matters more than any modelling refinement.

---

## Project status

| Phase | State |
|---|---|
| **0 · Data spine** | Complete — collector automated, 113,582 historical rows loaded |
| **1 · Baseline + scoreboard** | Complete — prediction log, honest backtest, ranking metric |
| **2 · Real model** | Complete — beats FPL's `xP`; live recommendations |
| **3 · Optimiser** | Complete — MILP squad / transfers / wildcard |
| **4 · Minutes & news edge** | Partial — alerter built and scheduled; European congestion still outstanding |
| **5 · Self-improving loop** | Built but **unproven** — no gameweek has been scored live yet |

> **Important caveat.** Every result above is a **backtest**. Predictions for GW4 are
> logged but not yet graded. The system has never made a verified live prediction.

---

## Quick start

Requires Python 3.12 with `numpy`, `pandas`, `scikit-learn`, `scipy`.

```bash
# 1. Build the database (~2 minutes)
python fpl_collect.py                 # snapshot the live FPL API
python backfill_history.py --load     # 4 seasons, 113,582 rows
python predict.py --backfill          # this season's actual results

# 2. Verify everything is sound
python qa.py --all                    # 33 checks

# 3. Get this week's recommendation
python recommend.py --run             # trains, predicts, logs before the deadline
python optimise.py --transfers --free 1
```

### Rebuilding from scratch

`fpl.db` is gitignored because it is ~40 MB and fully regenerable. The three commands
in step 1 rebuild it. Note that **snapshot history cannot be recovered** — the FPL API
only ever exposes current state, so prices, ownership and injury flags from past days
are gone unless they were captured at the time. This is why collection is automated.

---

## The scripts

| Script | Purpose | Modes |
|---|---|---|
| `fpl_collect.py` | Snapshot the live API into SQLite | `--summary` |
| `backfill_history.py` | Load 4 historical seasons | `--load` `--verify` |
| `predict.py` | Baselines, prediction log, backtests | `--backfill` `--predict` `--score` `--backtest` `--rank` `--status` |
| `minutes.py` | Minutes analysis, persistence baseline | `--explore` `--persist` |
| `model.py` | Train and compare ML models | `--train` |
| `captain.py` | Season-long captaincy simulation | `--run` |
| `recommend.py` | Live recommendation + logs predictions | `--run` |
| `optimise.py` | MILP squad / transfer / wildcard | `--squad` `--transfers` `--wildcard` `--free N` `--bank X` `--budget X` `--horizon N` |
| `record_my_team.py` | Record actual squad + captaincy review | `--show` |
| `alert.py` | Diff snapshots for team-news changes | `--check [--hours N]` `--watch-wildcard` `--watchlist` |
| `qa.py` | Full QA suite (33 checks) | `--all` |

---

## Data model

SQLite, one file. Key tables:

| Table | Rows | Notes |
|---|---|---|
| `snapshots` | grows | One per collection run; the timeline spine |
| `players` | 654 × snapshots | Prices, ownership, **`news`**, `chance_next_round`, `status` |
| `fixtures` | 380 × snapshots | With difficulty ratings |
| `history` | 113,582 | Four static seasons — **one row per player PER FIXTURE** |
| `player_gw` | grows | This season's actual results, 18 columns incl. xG/ICT/BPS |
| `predictions` | grows | Every prediction, timestamped **before the deadline** |
| `my_squad` / `my_gameweeks` | small | Your real team, for benchmarking |
| `watchlist` | 15 | Players the alerter tracks closely |

> **Trap: `history` is keyed `(season, gw, element, fixture)`.** In a *double gameweek*
> a player has two fixtures in one GW, so `(season, gw, element)` is **not unique** —
> keying on it silently drops one fixture per DGW (374 cases in 2024-25 alone), and
> those are exactly the weeks that decide chips. **For per-gameweek totals, SUM over
> fixtures.**

---

## Methodology notes

These are the decisions that make the numbers trustworthy.

**Chronological validation only.** Train on earlier seasons, test on a later one.
Random k-fold on time-series data leaks the future into the past and produces
flattering, useless scores.

**No feature leakage.** Every feature is built with
`groupby(season, element).shift(1)` *before* any rolling window, so row *N* sees rows
1..*N*-1 and nothing else. Verified by a negative control: training on a **shuffled
target** gives MAE 1.551 vs 1.537 for predicting the mean — i.e. no skill, which is
the definitive proof.

**Ranking, not MAE, is the headline metric.** You never consume a prediction of "4.2
points" — you pick the best ~15 players. `predict.py --rank` and the top-K evaluation
in `model.py` measure what the model's favourites actually scored.

**High-return MAE is biased and must not be used alone.** That subset is conditioned
on players who *did* play, which systematically punishes any model that correctly
discounts for the risk of not playing. This is why `two_stage` loses on high-return
MAE while winning on ranking.

**Predictions are logged before the deadline.** `recommend.py` refuses to log after
the deadline has passed, because a prediction made afterwards is not a prediction.

---

## QA

`python qa.py --all` — **33 checks**, exits non-zero on failure so it can gate CI.

- **DATA** — row counts, duplicate keys, double-gameweek preservation, ranges, NULLs
- **LOGIC** — captain doubling, predictions logged pre-deadline
- **LEAKAGE** — manual feature recomputation, target-copy check, negative control, chronology
- **MODEL** — reproducibility, overfit ratio, beats trivial baselines, sane prediction range
- **OPTIMISER** — squad legality re-verified against the FPL rulebook independently of the solver

Two lessons from building it:

1. **A test that cannot fail is not a test.** The original feature-recomputation check
   compared `0.0000` vs `0.0000` and passed while proving nothing. It now verifies 12
   consecutive rows against the season's top scorer.
2. **Both initial "failures" were bugs in the tests, not the pipeline** — a 60%
   target-match rate that was really 50% trivial `0 == 0`, and "points without minutes"
   that turned out to be *managers* (FPL added "AM" as a position in 2024-25).

---

## Planned app (feasibility confirmed)

Measured: full retrain **43.8 s / 210 MB peak**; predicting all 654 players **0.01 s**.
Railway Hobby allows 48 GB RAM per service — this uses ~0.4% of the ceiling.

**Estimated added cost: ~$0.22/month**, inside the existing $5 credit.

### Use cases

| Actor | Use cases |
|---|---|
| **Manager (you)** | View recommendation · Run optimiser · Plan wildcard · Review model performance |
| **Scheduler (cron)** | Collect snapshot · Detect news change · Send email alert · Retrain & score |
| **FPL API** | Provides player, fixture and availability data |

### Screens

1. **Dashboard** — deadline countdown, captain, alert banner, recommended XI
2. **My Squad** — your 15 ranked by expected points, with availability flags
3. **Transfers** — optimiser output, free-transfer selector, hit arithmetic, wildcard mode
4. **Players** — searchable table of all 654: xPts, price, ownership, form, news
5. **Alerts** — the news-change log
6. **Performance** — model vs baselines vs your own gut picks over time

### Automation

| Job | Cron | Why then |
|---|---|---|
| Collect snapshot | `0 9,21 * * *` | Prices/news change daily; missed data is unrecoverable |
| Diff + alert | `15 21 * * *` | 15 min after the snapshot, so it diffs fresh data |
| **Deadline email** | `0 18 * * 4,5` | **Thu/Fri — when press conferences actually land** |
| Predict + log | `0 8 * * 6` | Before a Saturday deadline |
| Score + retrain | `0 10 * * 1` | Monday, once results are final |

### Two risks that must be handled before deploying

1. **Storage.** The `players.raw` JSON column is 1.7 MB per snapshot — **94% of all
   growth** — projecting ~3 GB/season against a 5 GB volume. Drop it for historical
   snapshots, keep only the newest. Growth falls to ~53 MB/season.
2. **Always-on cost.** A 512 MB service running 24/7 costs `0.5 × $10 = $5.00/month`,
   the entire credit. **Sleeping is the cost model, not an optimisation.**

---

## Local automation (current)

Two Windows scheduled tasks, running now:

```
FPL Snapshot          09:00, 21:00   -> fpl_collect.py
FPL Team News Alert   21:15          -> alert.py --check >> alerts.log
```

Remove with `Unregister-ScheduledTask -TaskName "<name>" -Confirm:$false`.

---

## Roadmap

- [ ] Drop the `raw` storage bloat (1 h, blocks deployment)
- [ ] Score GW4 live — the **first real validation** of everything above
- [ ] Position-specific models (OpenFPL's approach; still outstanding)
- [ ] European fixture congestion — CL/EL fixtures are not in the FPL API
- [ ] Deploy to Railway with cron + email (est. 14–17 h)
- [ ] Champion/challenger promotion: only ship a model that wins a held-out backtest

---

## Prior art

Standing on rather than reinventing:

- **[OpenFPL](https://github.com/daniegr/OpenFPL)** ([paper](https://arxiv.org/html/2508.09992v1)) —
  position-specific ensembles matching commercial services. Its documented weakness is
  **expected minutes**, which is precisely the gap this project targets.
- **[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)** —
  the historical archive, with xG/xA already merged in.
- The FPL API itself — public, no auth, and richer than most people realise
  (28 fields per player per gameweek including xG, xA, ICT, BPS).

---

## Notes

- **Windows console encoding.** Printing player names crashes with
  `UnicodeEncodeError` on cp1252 (e.g. `ć`). Use `optimise.safe()` to fold accents.
- **FPL `web_name` is not unique.** There are two "Palmer" and two "Martinez". Match on
  `(web_name, position)`; matching on name alone silently picks the wrong player.
- **`chance_of_playing` is NULL for fit players**, not 0. Normalise to 100 or "cleared
  to play" reads as missing data.
