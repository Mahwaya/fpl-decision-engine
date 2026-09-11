"""
Engine versus gut — the scoreboard this whole project exists to keep.

WHY THIS IS THE POINT. A model that produces confident numbers is easy. A model
that is demonstrably better than the person using it is the only thing worth
trusting with a decision. Percival picked GW1-3 on instinct and scored 180
points; the engine has to beat that, not beat a naive baseline.

So every gameweek this asks three questions, in increasing order of honesty:

  1. BEFORE the deadline — where do we disagree, and by how much?
     Expected points only. Cheap talk, but it records the disagreement in
     writing so it cannot be rationalised afterwards.

  2. AFTER the gameweek — who was actually right?
     Real points. This is the only number that counts.

  3. OVER the season — is either of us reliably better?
     One gameweek is noise. A season is evidence.

THE COMPARISON IS DELIBERATELY NARROW. It holds the 15-man squad fixed and
compares only the decisions actually available once the squad exists: which 11
start, and who is captain. Comparing against a differently-picked squad would
be comparing against a counterfactual nobody could have acted on.

CAPTAINCY IS WEIGHTED DOUBLE ON PURPOSE, because it is. A captain's points are
doubled, so a captaincy disagreement is worth exactly twice the raw gap between
the two players — which is why record_my_team.py's review found 44 points lost
to captaincy alone across three gameweeks.

Usage:
    python compare.py --gw 4        # expected points, before the deadline
    python compare.py --score       # real points, after the gameweek
    python compare.py --season      # every scored gameweek, cumulative
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "fpl.db"
MODEL = "two_stage_ml"

# THE HUMAN OVERLAY, APPLIED HERE BECAUSE THE MODEL CANNOT SEE IT.
#
# opponent.py tested fixture difficulty as a model FEATURE and it was rejected:
# paired bootstrap over 38 gameweeks gave -0.125 ranking, 95% CI [-0.287,
# +0.036], P(better) = 0.06. But the underlying effect is real and large —
# players with 60+ minutes average 2.73 points against the toughest fifth of
# opponents and 4.33 against the leakiest, a 1.59-point swing, 45% of the
# average return. The conclusion recorded there was explicit: apply fixture
# difficulty as a HUMAN OVERLAY on the model's output, especially for
# captaincy, where one extreme fixture cannot be averaged away.
#
# This is that overlay. Same multipliers as optimise.py and predict.py.
FDR_MULT = {1: 1.15, 2: 1.08, 3: 1.00, 4: 0.92, 5: 0.85}

# Legal FPL formations: exactly one keeper, then any split of the outfield ten
# inside these bounds.
DEF_RANGE = range(3, 6)
MID_RANGE = range(2, 6)
FWD_RANGE = range(1, 4)


def safe(s):
    """Windows consoles default to cp1252 and choke on Horníček / Rúben."""
    if s is None:
        return ""
    try:
        s.encode(sys.stdout.encoding or "utf-8")
        return s
    except (UnicodeEncodeError, LookupError):
        import unicodedata
        return (unicodedata.normalize("NFKD", s)
                .encode("ascii", "ignore").decode("ascii") or "?")


def fixture_map(conn, gw):
    """team_id -> (opponent short name, H/A, difficulty) for this gameweek."""
    snap = conn.execute("SELECT MAX(id) FROM snapshots").fetchone()[0]
    names = {r[0]: r[1] for r in conn.execute(
        "SELECT team_id, short_name FROM teams WHERE snapshot_id=?", (snap,))}
    out = {}
    for th, ta, dh, da in conn.execute(
        "SELECT team_h, team_a, difficulty_h, difficulty_a"
        " FROM fixtures WHERE snapshot_id=? AND event=?", (snap, gw)
    ):
        out[th] = (names.get(ta), "H", dh)
        out[ta] = (names.get(th), "A", da)
    return out


def squad_rows(conn, gw, value_col, fdr=False):
    """
    The 15, with whichever value we are comparing on attached.

    value_col is 'ep' (expected, before the deadline) or 'actual' (real points,
    after). Everything downstream is identical, which is the point — the same
    arithmetic answers "who should have been right" and "who was right".
    """
    if value_col == "ep":
        val = ("(SELECT predicted FROM predictions pr WHERE pr.element_id = s.element_id"
               f" AND pr.gameweek = s.gameweek AND pr.model = '{MODEL}')")
    else:
        val = ("(SELECT a.total_points FROM player_gw a WHERE a.element_id = s.element_id"
               " AND a.gameweek = s.gameweek)")

    rows = conn.execute(
        f"""SELECT s.name, s.element_id, p.position, s.started, s.is_captain,
                   s.is_vice, {val} AS value, p.team_id
            FROM my_squad s
            LEFT JOIN players p
              ON p.element_id = s.element_id
             AND p.snapshot_id = (SELECT MAX(id) FROM snapshots)
            WHERE s.gameweek = ?""",
        (gw,),
    ).fetchall()

    fx = fixture_map(conn, gw) if fdr else {}
    squad = []
    for (n, e, pos, st, c, v, val_, team) in rows:
        raw = 0.0 if val_ is None else float(val_)
        opp, home, diff = fx.get(team, (None, None, None))
        # The overlay only ever adjusts EXPECTED points. Actual points are what
        # happened; scaling them by a fixture rating would be nonsense.
        adj = raw * FDR_MULT.get(diff, 1.0) if (fdr and value_col == "ep") else raw
        squad.append({
            "name": n, "element_id": e, "position": pos, "started": bool(st),
            "is_captain": bool(c), "is_vice": bool(v),
            "value": adj, "raw": raw, "missing": val_ is None,
            "opponent": opp, "home": home, "fdr": diff,
        })
    return squad


def best_xi(squad):
    """
    The highest-scoring legal XI from these 15.

    Greedy within each position is exact once the position counts are fixed —
    you always want the best N defenders, never the 2nd-best — so brute-forcing
    the handful of legal shapes gives the true optimum, not an approximation.
    """
    by_pos = {}
    for p in squad:
        by_pos.setdefault(p["position"], []).append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: -p["value"])

    if not by_pos.get("GKP"):
        return None, None
    keeper = by_pos["GKP"][0]

    best, best_total = None, float("-inf")
    for d in DEF_RANGE:
        for m in MID_RANGE:
            for f in FWD_RANGE:
                if d + m + f != 10:
                    continue
                if (len(by_pos.get("DEF", [])) < d or len(by_pos.get("MID", [])) < m
                        or len(by_pos.get("FWD", [])) < f):
                    continue
                xi = ([keeper] + by_pos["DEF"][:d]
                      + by_pos["MID"][:m] + by_pos["FWD"][:f])
                total = sum(p["value"] for p in xi)
                if total > best_total:
                    best, best_total = xi, total
    return best, best_total


def analyse(conn, gw, mode, fdr=False):
    label = "EXPECTED POINTS" if mode == "ep" else "ACTUAL POINTS"
    if fdr and mode == "ep":
        label += ", FIXTURE-ADJUSTED"
    squad = squad_rows(conn, gw, mode, fdr)
    if not squad:
        print(f"  Nothing recorded for GW{gw}.")
        return None

    missing = [p["name"] for p in squad if p["missing"]]
    if len(missing) == len(squad):
        print(f"  No {label.lower()} available for GW{gw} yet.")
        if mode == "actual":
            print("  Run:  python predict.py --backfill")
        return None

    mine = [p for p in squad if p["started"]]
    if len(mine) != 11:
        print(f"  WARNING: {len(mine)} players marked as starting, expected 11.")

    my_total = sum(p["value"] for p in mine)
    engine_xi, engine_total = best_xi(squad)

    # --- captaincy ---
    my_cap = next((p for p in squad if p["is_captain"]), None)
    eligible = [p for p in squad if not p["missing"] or mode == "actual"]
    engine_cap = max(eligible, key=lambda p: p["value"]) if eligible else None

    print(f"\n{'=' * 66}")
    print(f"  GW{gw} — ENGINE vs GUT   ({label})")
    print(f"{'=' * 66}")

    # ---------------------------------------------------------- starting XI
    print(f"\n  STARTING XI")
    my_ids = {p["element_id"] for p in mine}
    eng_ids = {p["element_id"] for p in engine_xi}

    print(f"    your XI      {my_total:>7.2f}")
    print(f"    engine XI    {engine_total:>7.2f}")
    gap = engine_total - my_total
    verdict = "engine ahead" if gap > 0.005 else ("identical" if abs(gap) <= 0.005 else "YOU ahead")
    print(f"    difference   {gap:>+7.2f}   ({verdict})")

    if fdr and mode == "ep":
        print()
        print(f"    {'player':<14}{'opp':<10}{'FDR':<5}{'model':>8}{'adjusted':>10}")
        for p_ in sorted(mine, key=lambda x: -x["value"]):
            opp = f"{p_['opponent']} ({p_['home']})" if p_["opponent"] else "?"
            print(f"    {safe(p_['name']):<14}{opp:<10}{p_['fdr'] or 0:<5}"
                  f"{p_['raw']:>8.2f}{p_['value']:>10.2f}")

    if my_ids != eng_ids:
        benched = [p for p in mine if p["element_id"] not in eng_ids]
        started = [p for p in engine_xi if p["element_id"] not in my_ids]
        print(f"\n    the engine would swap:")
        for out_, in_ in zip(sorted(benched, key=lambda p: p["value"]),
                             sorted(started, key=lambda p: -p["value"])):
            print(f"      OUT {safe(out_['name']):<14}{out_['position']} {out_['value']:>6.2f}"
                  f"   ->   IN {safe(in_['name']):<14}{in_['position']} {in_['value']:>6.2f}")
    else:
        print("\n    same eleven — no disagreement on selection.")

    # ------------------------------------------------------------- captain
    print(f"\n  CAPTAIN")
    if my_cap and engine_cap:
        print(f"    yours        {safe(my_cap['name']):<14}{my_cap['value']:>7.2f}"
              f"   (doubles to {2*my_cap['value']:.2f})")
        print(f"    engine       {safe(engine_cap['name']):<14}{engine_cap['value']:>7.2f}"
              f"   (doubles to {2*engine_cap['value']:.2f})")
        if my_cap["element_id"] == engine_cap["element_id"]:
            print(f"    AGREED.")
            cap_gap = 0.0
        else:
            # Doubled, because that is what a captaincy decision is worth.
            cap_gap = 2 * (engine_cap["value"] - my_cap["value"])
            who = "engine ahead" if cap_gap > 0 else "YOU ahead"
            print(f"    difference   {cap_gap:>+7.2f}   ({who}, counted double)")
    else:
        cap_gap = 0.0
        print("    no captain recorded.")

    # -------------------------------------------------------------- totals
    total_gap = gap + cap_gap
    print(f"\n  COMBINED")
    print(f"    engine minus you   {total_gap:>+7.2f} points")
    if mode == "ep":
        print(f"\n    This is a PREDICTION of the disagreement, logged before the")
        print(f"    deadline. It settles nothing. Run --score after the gameweek.")
    else:
        if total_gap > 0.005:
            print(f"\n    The engine was right this week, by {total_gap:.0f} points.")
        elif total_gap < -0.005:
            print(f"\n    YOU were right this week, by {abs(total_gap):.0f} points.")
        else:
            print(f"\n    Dead heat.")

    if missing and mode == "ep":
        print(f"\n  NOTE: no prediction for {', '.join(safe(m) for m in missing)}"
              f" — treated as 0.00.")

    return {"gw": gw, "xi_gap": gap, "cap_gap": cap_gap, "total": total_gap,
            "my_total": my_total, "engine_total": engine_total}


def season(conn):
    """Every gameweek that has been played and scored. One week is noise."""
    gws = [r[0] for r in conn.execute(
        "SELECT DISTINCT gameweek FROM my_squad WHERE gameweek IN"
        " (SELECT DISTINCT gameweek FROM player_gw) ORDER BY gameweek")]
    if not gws:
        print("  No gameweek has both a recorded squad and finished results yet.")
        return

    print(f"\n  {'GW':<5}{'your XI':>10}{'engine XI':>12}{'XI diff':>10}"
          f"{'cap diff':>11}{'total':>9}")
    print("  " + "-" * 57)
    running = 0.0
    for gw in gws:
        squad = squad_rows(conn, gw, "actual")
        mine = [p for p in squad if p["started"]]
        if not mine:
            continue
        my_total = sum(p["value"] for p in mine)
        _, eng_total = best_xi(squad)
        my_cap = next((p for p in squad if p["is_captain"]), None)
        eng_cap = max(squad, key=lambda p: p["value"])
        cap_gap = 0.0 if (my_cap and my_cap["element_id"] == eng_cap["element_id"]) \
            else 2 * (eng_cap["value"] - (my_cap["value"] if my_cap else 0))
        gap = eng_total - my_total
        running += gap + cap_gap
        print(f"  {gw:<5}{my_total:>10.0f}{eng_total:>12.0f}{gap:>+10.0f}"
              f"{cap_gap:>+11.0f}{gap+cap_gap:>+9.0f}")

    print("  " + "-" * 57)
    print(f"  cumulative engine advantage: {running:+.0f} points over {len(gws)} gameweeks")
    if running > 0:
        print(f"  The engine is ahead. Every point there is one your instinct left behind.")
    elif running < 0:
        print(f"  YOU are ahead. The model is not yet worth deferring to.")


def main():
    fdr = "--fdr" in sys.argv
    conn = sqlite3.connect(DB_PATH)
    try:
        if "--season" in sys.argv:
            season(conn)
        elif "--score" in sys.argv:
            gw = (int(sys.argv[sys.argv.index("--gw") + 1]) if "--gw" in sys.argv
                  else conn.execute("SELECT MAX(gameweek) FROM my_squad").fetchone()[0])
            analyse(conn, gw, "actual", fdr)
        elif "--gw" in sys.argv:
            analyse(conn, int(sys.argv[sys.argv.index("--gw") + 1]), "ep", fdr)
        else:
            gw = conn.execute("SELECT MAX(gameweek) FROM my_squad").fetchone()[0]
            analyse(conn, gw, "ep", fdr)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
