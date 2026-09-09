"""
Phase 2 step 4 — captaincy, the highest-value decision in FPL.

WHY THIS FILE EXISTS: Percival's own GW1-3 review showed 44 points lost to
captaincy alone in three gameweeks. That is the single biggest measured leak
in the season so far, and bigger than any plausible gain from better squad
selection.

THE HYPOTHESIS THIS TESTED — AND DISPROVED: the common FPL wisdom (which this
project repeated) is that captaincy is a CEILING decision, so you should pick
the player most likely to explode rather than the one with the best expected
points. The simulation below shows that is wrong for maximising score, and the
reason is simple: captain points are doubled LINEARLY, so maximising E[2X] is
just maximising E[X]. Expected value is provably the right criterion.

The ceiling framing only applies when chasing RANK in a mini-league, where you
need differentials and variance is deliberately desirable. That is a different
objective from scoring the most points.

So this compares strategies that optimise for different things:
    most_owned    captain whoever the crowd owns most (the "template" default)
    xP            FPL's own expected points
    two_stage     our champion expected-points model
    p_haul        P(points >= 10)      -- explicitly optimising for explosions
    q90           90th-percentile outcome via quantile regression
    perfect       hindsight upper bound (not achievable, shows headroom)

Captains are chosen from a realistic pool: the most-owned players that
gameweek. You can only captain someone you own, and in practice that is a
premium/template player, not an obscure defender.

Usage:
    python captain.py --run
"""

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

from model import load, build_features, TRAIN_SEASONS, TEST_SEASON

warnings.filterwarnings("ignore")
DB_PATH = Path(__file__).parent / "fpl.db"

POOL = 50      # realistic captain candidates per gameweek, by ownership
HAUL = 10      # what counts as an explosion


def run(conn):
    df = load(conn)
    df, feats = build_features(df)

    train_df = df[df["season"].isin(TRAIN_SEASONS)].dropna(subset=["total_points"])
    test_df = df[df["season"] == TEST_SEASON].dropna(subset=["total_points"]).copy()
    print(f"  Train {len(train_df):,} rows | Test {len(test_df):,} rows ({TEST_SEASON})\n")

    Xtr, ytr = train_df[feats], train_df["total_points"].values
    Xte = test_df[feats]

    common = dict(max_iter=400, learning_rate=0.06, max_depth=6,
                  min_samples_leaf=40, random_state=42)

    # --- champion expected-points model (two-stage) -----------------------
    clf60 = HistGradientBoostingClassifier(**common)
    clf60.fit(Xtr, train_df["played_60"].values)
    p60 = clf60.predict_proba(Xte)[:, 1]

    played = train_df["played_60"] == 1
    reg = HistGradientBoostingRegressor(l2_regularization=1.0, **common)
    reg.fit(Xtr[played], ytr[played.values])
    test_df["two_stage"] = p60 * reg.predict(Xte)

    # --- explicit haul probability ---------------------------------------
    haul_clf = HistGradientBoostingClassifier(**common)
    haul_clf.fit(Xtr, (ytr >= HAUL).astype(int))
    test_df["p_haul"] = haul_clf.predict_proba(Xte)[:, 1]

    # --- ceiling via quantile regression ---------------------------------
    q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **common)
    q90.fit(Xtr, ytr)
    test_df["q90"] = q90.predict(Xte)

    test_df["xp_f"] = test_df["xp"].fillna(0)
    test_df["owned"] = test_df["selected"].fillna(0)

    strategies = ["most_owned", "xP", "two_stage", "p_haul", "q90", "perfect"]
    totals = {s: [] for s in strategies}

    for gw, grp in test_df.groupby("gw"):
        # realistic candidate pool: the most-owned players that week
        pool = grp.nlargest(min(POOL, len(grp)), "owned")
        if pool.empty:
            continue
        picks = {
            "most_owned": pool.nlargest(1, "owned"),
            "xP":         pool.nlargest(1, "xp_f"),
            "two_stage":  pool.nlargest(1, "two_stage"),
            "p_haul":     pool.nlargest(1, "p_haul"),
            "q90":        pool.nlargest(1, "q90"),
            "perfect":    pool.nlargest(1, "total_points"),
        }
        for s, row in picks.items():
            totals[s].append(float(row["total_points"].iloc[0]))

    n_gw = len(totals["perfect"])
    print(f"  Captaincy simulation over {n_gw} gameweeks of {TEST_SEASON}")
    print(f"  (captain chosen from the {POOL} most-owned players each week)\n")
    print(f"  {'strategy':<13}{'pts/GW':>9}{'season*':>10}{'hauls':>8}{'blanks':>8}")
    print("  " + "-" * 48)

    baseline = np.mean(totals["most_owned"]) * 2
    for s in strategies:
        v = np.array(totals[s])
        per_gw = v.mean() * 2                      # captain points are doubled
        hauls = int((v >= HAUL).sum())
        blanks = int((v <= 2).sum())
        tag = ""
        if s == "perfect":
            tag = "  <- ceiling, not achievable"
        elif s != "most_owned":
            diff = per_gw - baseline
            tag = f"  {diff:+.2f} vs template"
        print(f"  {s:<13}{per_gw:>9.2f}{per_gw*38:>10.0f}{hauls:>8}{blanks:>8}{tag}")

    print("\n  * season = pts/GW x 38, i.e. captaincy contribution over a full season.")
    print("  hauls = captain scored 10+.  blanks = captain scored 2 or less.")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        if "--run" in sys.argv:
            run(conn)
        else:
            print(__doc__)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
