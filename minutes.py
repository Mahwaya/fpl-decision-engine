"""
Phase 2, step 1 — understand the minutes problem before modelling it.

The Phase 1 backtest showed aggregate MAE barely beats "always predict zero",
because most players don't play. So the real first question is not "how many
points will this player score" but "will this player be on the pitch at all".

FPL appearance points make this concrete:
    0 minutes        -> 0 points
    1-59 minutes     -> 1 point
    60+ minutes      -> 2 points  (and only 60+ players realistically haul)

So the target that matters most is P(60+ minutes).

This file only LOOKS at the data. No model yet — measure first, build second.

Usage:
    python minutes.py --explore     # what the minutes data actually looks like
    python minutes.py --persist     # how predictable is playing time?
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "fpl.db"


def explore(conn):
    gws = [r[0] for r in conn.execute(
        "SELECT DISTINCT gameweek FROM player_gw ORDER BY gameweek")]
    if not gws:
        print("  No results yet. Run: python predict.py --backfill")
        return

    print("  How playing time is distributed, by gameweek\n")
    print(f"  {'GW':<5}{'in data':>9}{'0 min':>9}{'1-59':>8}{'60+':>8}{'60+ %':>8}")
    for gw in gws:
        rows = conn.execute(
            "SELECT minutes FROM player_gw WHERE gameweek=?", (gw,)
        ).fetchall()
        total = len(rows)
        zero = sum(1 for (m,) in rows if not m)
        part = sum(1 for (m,) in rows if m and m < 60)
        full = sum(1 for (m,) in rows if m and m >= 60)
        print(f"  {gw:<5}{total:>9}{zero:>9}{part:>8}{full:>8}{100*full/total:>7.1f}%")

    print("\n  Where the points actually come from\n")
    print(f"  {'Bucket':<12}{'players':>9}{'mean pts':>10}{'>=5 pts':>9}{'share of all pts':>18}")
    all_pts = conn.execute("SELECT SUM(total_points) FROM player_gw").fetchone()[0] or 1
    for label, cond in (
        ("0 min",   "minutes IS NULL OR minutes = 0"),
        ("1-59 min", "minutes > 0 AND minutes < 60"),
        ("60+ min",  "minutes >= 60"),
    ):
        rows = conn.execute(
            f"SELECT total_points FROM player_gw WHERE {cond}"
        ).fetchall()
        n = len(rows)
        if not n:
            continue
        pts = [r[0] or 0 for r in rows]
        hauls = sum(1 for p in pts if p >= 5)
        print(f"  {label:<12}{n:>9}{sum(pts)/n:>10.2f}{hauls:>9}{100*sum(pts)/all_pts:>17.1f}%")

    print("\n  Read: if you can predict who gets 60+ minutes, you have narrowed")
    print("  the field to where essentially all the points live.")


def persist(conn):
    """
    How well does recent playing time predict next-gameweek playing time?
    This is the baseline any minutes model has to beat.
    """
    gws = [r[0] for r in conn.execute(
        "SELECT DISTINCT gameweek FROM player_gw ORDER BY gameweek")]
    testable = [g for g in gws if g > min(gws)]
    if not testable:
        print("  Need at least two gameweeks of results.")
        return

    print("  Baseline: P(60+ mins) = share of previous gameweeks with 60+ mins\n")
    print(f"  {'GW':<5}{'n':>7}{'Brier':>9}{'accuracy':>11}{'always-no':>12}")

    for gw in testable:
        rows = []
        for eid, mins in conn.execute(
            "SELECT element_id, minutes FROM player_gw WHERE gameweek=?", (gw,)
        ):
            hist = conn.execute(
                "SELECT minutes FROM player_gw WHERE element_id=? AND gameweek<?"
                " ORDER BY gameweek DESC LIMIT 3", (eid, gw)
            ).fetchall()
            if not hist:
                continue
            p = sum(1 for (m,) in hist if m and m >= 60) / len(hist)
            actual = 1 if (mins and mins >= 60) else 0
            rows.append((p, actual))

        if not rows:
            continue
        n = len(rows)
        brier = sum((p - a) ** 2 for p, a in rows) / n       # lower is better
        acc = sum(1 for p, a in rows if (p >= 0.5) == bool(a)) / n
        # "nobody plays 60+" — the trivial comparison
        always_no = sum(1 for _, a in rows if a == 0) / n
        print(f"  {gw:<5}{n:>7}{brier:>9.3f}{100*acc:>10.1f}%{100*always_no:>11.1f}%")

    print("\n  Brier score: 0 = perfect, 0.25 = coin flip. 'always-no' is the")
    print("  accuracy you would get by predicting nobody starts — beat that,")
    print("  or the model is adding nothing.")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        if "--explore" in sys.argv:
            explore(conn)
        elif "--persist" in sys.argv:
            persist(conn)
        else:
            print(__doc__)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
