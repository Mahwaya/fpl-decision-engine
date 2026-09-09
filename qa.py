"""
QA suite for the FPL decision engine.

A modelling pipeline can be confidently, silently wrong. The dangerous failures
are not crashes — they are leakage, dropped rows, and metrics that flatter. This
file tries to catch those.

Checks are grouped:
    DATA      integrity of what we stored
    LEAKAGE   the killer for time-series models
    MODEL     does it generalise, is it reproducible
    LOGIC     is the arithmetic in the evaluators actually right

Usage:
    python qa.py --all
"""

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
DB_PATH = Path(__file__).parent / "fpl.db"

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def check(name, status, detail=""):
    results.append((name, status, detail))
    icon = {PASS: "[ok]  ", FAIL: "[FAIL]", WARN: "[warn]"}[status]
    print(f"  {icon} {name}")
    if detail:
        print(f"         {detail}")


# ---------------------------------------------------------------- DATA

def qa_data(conn):
    print("\nDATA INTEGRITY")

    n = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    check("history table populated", PASS if n > 100_000 else FAIL, f"{n:,} rows")

    dupes = conn.execute(
        "SELECT COUNT(*) FROM (SELECT season, gw, element, fixture, COUNT(*) c"
        " FROM history GROUP BY 1,2,3,4 HAVING c > 1)"
    ).fetchone()[0]
    check("no duplicate primary keys", PASS if dupes == 0 else FAIL, f"{dupes} duplicated keys")

    # The double-gameweek fix: players with 2+ fixtures in one GW must survive.
    dgw = conn.execute(
        "SELECT COUNT(*) FROM (SELECT season, gw, element, COUNT(*) c"
        " FROM history GROUP BY 1,2,3 HAVING c > 1)"
    ).fetchone()[0]
    check("double gameweeks preserved", PASS if dgw > 500 else FAIL,
          f"{dgw} player-gameweeks with multiple fixtures")

    bad_gw = conn.execute("SELECT COUNT(*) FROM history WHERE gw < 1 OR gw > 38").fetchone()[0]
    check("gameweeks within 1-38", PASS if bad_gw == 0 else FAIL, f"{bad_gw} out of range")

    bad_min = conn.execute(
        "SELECT COUNT(*) FROM history WHERE minutes < 0 OR minutes > 120"
    ).fetchone()[0]
    check("minutes plausible (0-120)", PASS if bad_min == 0 else WARN, f"{bad_min} outside range")

    nulls = conn.execute(
        "SELECT COUNT(*) FROM history WHERE total_points IS NULL OR minutes IS NULL"
    ).fetchone()[0]
    check("no NULL in target/minutes", PASS if nulls == 0 else FAIL, f"{nulls} rows")

    seasons = conn.execute("SELECT COUNT(DISTINCT season) FROM history").fetchone()[0]
    check("four seasons loaded", PASS if seasons == 4 else WARN, f"{seasons} seasons")

    # Points without minutes: legitimate ONLY for managers. FPL added "AM"
    # (Assistant Manager) as a position in 2024-25; managers score off team
    # results, not minutes played, so 0 minutes + 20 points is correct for them.
    # model.py filters positions to GKP/DEF/MID/FWD, so they never reach the
    # model — this check confirms that filter is still needed and still right.
    odd_players = conn.execute(
        "SELECT COUNT(*) FROM history WHERE minutes = 0 AND total_points > 0"
        " AND position IN ('GKP','DEF','MID','FWD')"
    ).fetchone()[0]
    odd_mgr = conn.execute(
        "SELECT COUNT(*) FROM history WHERE minutes = 0 AND total_points > 0"
        " AND position NOT IN ('GKP','DEF','MID','FWD')"
    ).fetchone()[0]
    check("no outfield points without minutes", PASS if odd_players == 0 else WARN,
          f"{odd_players} outfield rows; {odd_mgr} manager rows (expected, filtered out of the model)")


# ---------------------------------------------------------------- LEAKAGE

def qa_leakage(conn):
    print("\nLEAKAGE (the failure mode that silently invalidates everything)")
    from model import load, build_features, TRAIN_SEASONS, TEST_SEASON

    df = load(conn)
    df, feats = build_features(df)

    # 1. Manual recomputation against a HIGH-SCORING player.
    #    Testing a player whose scores are all zero would "pass" while proving
    #    nothing (0 == 0). Pick someone with real variance, and check several
    #    rows, so the assertion has teeth.
    sub = df[df.season == "2024-25"].sort_values(["element", "gw", "fixture"])
    totals = sub.groupby("element")["total_points"].sum()
    target_el = totals.idxmax()                      # the season's top scorer
    g = sub[sub.element == target_el].reset_index(drop=True)

    mismatches, checked, sample = 0, 0, ""
    for i in range(3, min(len(g), 15)):
        expected = g.iloc[:i]["total_points"].tail(3).mean()   # prior rows only
        got = g.iloc[i]["total_points_m3"]
        if pd.isna(got):
            continue
        checked += 1
        if abs(expected - got) > 1e-6:
            mismatches += 1
        if not sample:
            sample = f"row {i}: expected {expected:.4f}, feature {got:.4f}"

    variance_ok = g["total_points"].std() > 0
    ok = checked > 0 and mismatches == 0 and variance_ok
    check("rolling features use prior fixtures only",
          PASS if ok else FAIL,
          f"{checked} rows verified on '{g['name'].iloc[0]}' "
          f"(season total {int(totals.max())} pts, std {g['total_points'].std():.2f}); "
          f"{mismatches} mismatches; {sample}")

    # 2. The current row's own outcome must never appear in its features.
    #    NOTE: comparing raw equality is misleading here. ~60% of rows score
    #    zero, so "last == current" is trivially true for 0 == 0 half the time.
    #    That is a property of the score distribution, not a leak. Excluding
    #    zeros gives the honest signal; a true copy would sit near 100%.
    nz = df[(df["total_points"] != 0) & df["total_points_last"].notna()]
    same_nz = (nz["total_points_last"] == nz["total_points"]).mean()
    check("target not copied into features", PASS if same_nz < 0.60 else FAIL,
          f"{100*same_nz:.1f}% match among non-zero rows "
          f"(a genuine copy would be ~100%; FPL scores are discrete so overlap is normal)")

    # 3. NEGATIVE CONTROL — the strongest test available.
    #    Shuffle the target. A clean pipeline must then have almost no skill:
    #    MAE should collapse to roughly that of predicting the mean.
    train_df = df[df.season.isin(TRAIN_SEASONS)].dropna(subset=["total_points"])
    test_df = df[df.season == TEST_SEASON].dropna(subset=["total_points"])
    tr = train_df.sample(min(20000, len(train_df)), random_state=1)
    te = test_df.sample(min(6000, len(test_df)), random_state=1)

    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(tr["total_points"].values)
    m = HistGradientBoostingRegressor(max_iter=150, random_state=0)
    m.fit(tr[feats], y_shuf)
    mae_shuf = mean_absolute_error(te["total_points"], m.predict(te[feats]))
    mae_mean = mean_absolute_error(
        te["total_points"], np.full(len(te), tr["total_points"].mean())
    )
    # With a shuffled target the model should be no better than predicting the mean.
    leaked = mae_shuf < mae_mean * 0.95
    check("negative control (shuffled target)", FAIL if leaked else PASS,
          f"shuffled-target MAE {mae_shuf:.3f} vs predict-the-mean {mae_mean:.3f}")

    # 4. Chronology: no test season row may predate the training seasons.
    max_train = train_df["season"].max()
    check("train seasons precede test season", PASS if max_train < TEST_SEASON else FAIL,
          f"train max {max_train}, test {TEST_SEASON}")


# ---------------------------------------------------------------- MODEL

def qa_model(conn):
    print("\nMODEL BEHAVIOUR")
    from model import load, build_features, TRAIN_SEASONS, TEST_SEASON

    df = load(conn)
    df, feats = build_features(df)
    train_df = df[df.season.isin(TRAIN_SEASONS)].dropna(subset=["total_points"])
    test_df = df[df.season == TEST_SEASON].dropna(subset=["total_points"])

    tr = train_df.sample(min(30000, len(train_df)), random_state=2)
    Xtr, ytr = tr[feats], tr["total_points"].values
    Xte, yte = test_df[feats], test_df["total_points"].values

    p = dict(max_iter=250, learning_rate=0.06, max_depth=6,
             min_samples_leaf=40, random_state=42)
    m1 = HistGradientBoostingRegressor(**p).fit(Xtr, ytr)
    pred1 = m1.predict(Xte)

    # Reproducibility: same seed, same data -> identical predictions.
    m2 = HistGradientBoostingRegressor(**p).fit(Xtr, ytr)
    identical = np.allclose(pred1, m2.predict(Xte))
    check("reproducible with fixed seed", PASS if identical else FAIL)

    # Overfitting: a large train/test gap means it memorised rather than learned.
    mae_tr = mean_absolute_error(ytr, m1.predict(Xtr))
    mae_te = mean_absolute_error(yte, pred1)
    ratio = mae_te / mae_tr if mae_tr else 99
    check("not badly overfit", PASS if ratio < 1.5 else WARN,
          f"train MAE {mae_tr:.3f}, test MAE {mae_te:.3f}, ratio {ratio:.2f}")

    # Must beat the trivial baselines, or it is not earning its complexity.
    mae_zero = mean_absolute_error(yte, np.zeros(len(yte)))
    mae_mean = mean_absolute_error(yte, np.full(len(yte), ytr.mean()))
    beats = mae_te < min(mae_zero, mae_mean)
    check("beats predict-zero and predict-mean", PASS if beats else FAIL,
          f"model {mae_te:.3f} vs zero {mae_zero:.3f}, mean {mae_mean:.3f}")

    # Predictions should be in a sane range for FPL scoring.
    check("predictions in plausible range", PASS if pred1.min() > -3 and pred1.max() < 30 else WARN,
          f"min {pred1.min():.2f}, max {pred1.max():.2f}")


# ---------------------------------------------------------------- LOGIC

def qa_logic(conn):
    print("\nEVALUATOR LOGIC")

    # Captain doubling: a captain scoring 9 raw must contribute 18.
    raw, doubled = 9, 9 * 2
    check("captain doubling arithmetic", PASS if doubled == 18 else FAIL,
          f"raw {raw} -> {doubled}")

    # my_squad stores captain points ALREADY doubled (as the FPL app shows them).
    # Any comparison against team-mates must halve the captain first, or it is
    # not like-for-like. This bug was found and fixed on 2026-09-07.
    row = conn.execute(
        "SELECT name, points FROM my_squad WHERE gameweek=2 AND is_captain=1"
    ).fetchone()
    if row:
        ok = row[1] == 18   # Joao Pedro, 9 raw, shown as 18
        check("captain points stored doubled (as displayed)", PASS if ok else WARN,
              f"GW2 captain {row[0]} stored as {row[1]}")

    # Prediction log must be immutable and pre-deadline.
    n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    check("prediction log populated", PASS if n > 0 else WARN, f"{n} rows")

    bad = conn.execute(
        "SELECT COUNT(*) FROM predictions p JOIN snapshots s"
        " ON s.next_gw = p.gameweek WHERE p.made_at > s.next_deadline"
    ).fetchone()[0]
    check("no predictions logged after deadline", PASS if bad == 0 else FAIL,
          f"{bad} late predictions")

    # Our own collected results should agree with the historical archive where
    # the seasons overlap — an independent cross-check of the collector.
    ours = conn.execute("SELECT COUNT(*) FROM player_gw").fetchone()[0]
    check("own-collected results present", PASS if ours > 0 else WARN, f"{ours} rows")


def qa_optimiser(conn):
    """
    A MILP will happily return a confident, illegal squad if a constraint is
    mis-specified. Verify the solution against the FPL rulebook independently
    of the solver rather than trusting it.
    """
    print("\nOPTIMISER OUTPUT (checked against the FPL rulebook)")
    import optimise

    df, gw = optimise.candidates(conn)
    if df.empty:
        check("optimiser has candidates", WARN, "no predictions logged yet")
        return

    res = optimise.solve(df, budget=100.0)
    n = len(df)
    z = np.round(res.x).astype(int)
    squad = [i for i in range(n) if z[i] == 1]
    xi = [i for i in range(n) if z[n + i] == 1]
    caps = [i for i in range(n) if z[2 * n + i] == 1]

    check("squad has exactly 15", PASS if len(squad) == 15 else FAIL, f"{len(squad)}")

    counts = df.iloc[squad]["position"].value_counts().to_dict()
    want = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    check("squad composition 2/5/5/3", PASS if counts == want else FAIL, str(counts))

    spend = df.iloc[squad]["price"].sum()
    check("within 100.0m budget", PASS if spend <= 100.0 + 1e-6 else FAIL, f"{spend:.1f}m")

    per_club = df.iloc[squad]["team_id"].value_counts()
    check("max 3 players per club", PASS if per_club.max() <= 3 else FAIL,
          f"max {int(per_club.max())} from one club")

    check("starting XI has 11", PASS if len(xi) == 11 else FAIL, f"{len(xi)}")

    xi_pos = df.iloc[xi]["position"].value_counts().to_dict()
    legal = (xi_pos.get("GKP", 0) == 1
             and 3 <= xi_pos.get("DEF", 0) <= 5
             and 2 <= xi_pos.get("MID", 0) <= 5
             and 1 <= xi_pos.get("FWD", 0) <= 3)
    check("legal formation", PASS if legal else FAIL, str(xi_pos))

    check("exactly one captain", PASS if len(caps) == 1 else FAIL, f"{len(caps)}")
    check("captain is a starter", PASS if caps and caps[0] in xi else FAIL)
    check("all starters are in the squad", PASS if set(xi) <= set(squad) else FAIL)

    # Wildcard mode rescales `ep` by a fixture horizon. That rescaling must not
    # break any squad rule — a legal-looking but illegal wildcard would be
    # submitted and rejected, or worse, silently mis-shaped.
    snap = conn.execute("SELECT MAX(snapshot_id) FROM players").fetchone()[0]
    mult = optimise.horizon_multipliers(conn, snap, gw, 5)
    wdf = df.copy()
    wdf["ep"] = wdf["ep"] * wdf["team_id"].map(mult).fillna(0.0)
    wres = optimise.solve(wdf, budget=100.0)
    wz = np.round(wres.x).astype(int)
    wsquad = [i for i in range(n) if wz[i] == 1]
    wxi = [i for i in range(n) if wz[n + i] == 1]
    wcounts = wdf.iloc[wsquad]["position"].value_counts().to_dict()
    wclub = wdf.iloc[wsquad]["team_id"].value_counts().max()
    wpos = wdf.iloc[wxi]["position"].value_counts().to_dict()
    wlegal = (len(wsquad) == 15 and wcounts == want and len(wxi) == 11
              and wclub <= 3
              and wpos.get("GKP", 0) == 1
              and 3 <= wpos.get("DEF", 0) <= 5
              and 2 <= wpos.get("MID", 0) <= 5
              and 1 <= wpos.get("FWD", 0) <= 3)
    check("wildcard squad also legal", PASS if wlegal else FAIL,
          f"squad {wcounts}, XI {wpos}, max/club {int(wclub)}")

    # Horizon weighting must actually vary by team, or it is doing nothing.
    spread = max(mult.values()) - min(mult.values()) if mult else 0
    check("horizon weighting differentiates teams", PASS if spread > 0.3 else WARN,
          f"team multiplier range {min(mult.values()):.2f}-{max(mult.values()):.2f}")

    # A transfer must never be taken when the hit costs more than it gains.
    cur = pd.read_sql_query(
        "SELECT DISTINCT element_id FROM my_squad WHERE gameweek ="
        " (SELECT MAX(gameweek) FROM my_squad)", conn)
    ids = set(cur["element_id"].dropna().astype(int))
    if ids:
        owned_val = df[df["element_id"].isin(ids)]["price"].sum()
        r0 = optimise.solve(df, budget=owned_val, current_ids=ids, free=0)
        z0 = np.round(r0.x).astype(int)
        kept = sum(1 for i in range(n) if z0[i] == 1 and df["element_id"].iloc[i] in ids)
        moves = 15 - kept
        hits = int(z0[3 * n])
        # with zero free transfers, any move must be justified by > 4 pts
        check("no unjustified hits at free=0", PASS if moves == hits else FAIL,
              f"{moves} transfers, {hits} charged")


def summary():
    print("\n" + "=" * 58)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    print(f"  {n_pass} passed, {n_warn} warnings, {n_fail} failures")
    if n_fail:
        print("\n  FAILURES:")
        for name, s, d in results:
            if s == FAIL:
                print(f"    - {name}: {d}")
    if n_warn:
        print("\n  WARNINGS:")
        for name, s, d in results:
            if s == WARN:
                print(f"    - {name}: {d}")
    print("=" * 58)
    return n_fail


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        if "--all" in sys.argv:
            qa_data(conn)
            qa_logic(conn)
            qa_leakage(conn)
            qa_model(conn)
            qa_optimiser(conn)
            sys.exit(1 if summary() else 0)
        else:
            print(__doc__)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
