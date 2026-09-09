"""
Phase 3 — squad and transfer optimisation.

The model tells you what each player is worth. This turns that into a legal
squad, which is a constrained optimisation problem, not a sorting problem:
you cannot simply take the 15 highest-scoring players.

FPL rules encoded here:
  squad        exactly 15: 2 GKP, 5 DEF, 5 MID, 3 FWD
  budget       squad price must fit the bank
  clubs        at most 3 players from any one club
  XI           exactly 11 starters, formation 1 GK / 3-5 DEF / 2-5 MID / 1-3 FWD
  captain      exactly one, must be a starter, scores double
  transfers    each transfer beyond your free ones costs 4 points

Solved as a mixed-integer linear program with scipy.optimize.milp (HiGHS).
PuLP is the usual choice but scipy is already installed and does the same job.

Variables, per candidate player i:
    x_i  in squad         y_i  in starting XI        c_i  is captain
plus one integer h = number of point-costing transfers.

Usage:
    python optimise.py --squad              best 15 from scratch (wildcard view)
    python optimise.py --transfers          best moves from the current squad
    python optimise.py --transfers --free 2 --bank 0.5
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

DB_PATH = Path(__file__).parent / "fpl.db"

def safe(s):
    """
    Fold accents to ASCII for console output.

    The default Windows console codepage (cp1252) cannot encode characters like
    'c' with a hacek, and printing one raises UnicodeEncodeError mid-report.
    Squad lists are full of such names, so fold rather than crash.
    """
    import unicodedata
    # NFKD splits accents off their base letter, but some characters are not
    # decomposable at all -- 'O with stroke' (Odegaard) is a distinct letter,
    # not an accented O, so NFKD leaves it and 'ignore' then DELETES it. That
    # silently turned "Odegaard" into "degaard" and broke name matching.
    s = str(s)
    for a, b in (("Ø", "O"), ("ø", "o"), ("Æ", "AE"), ("æ", "ae"),
                 ("Đ", "D"), ("đ", "d"), ("Þ", "Th"), ("þ", "th"),
                 ("Ł", "L"), ("ł", "l")):
        s = s.replace(a, b)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


SQUAD = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
BENCH_WEIGHT = 0.10      # bench players occasionally play; worth a little, not much
HIT = 4.0                # points cost per transfer beyond the free allowance


# Fixture Difficulty Rating -> scoring multiplier. FDR 1 is a kind fixture.
FDR_MULT = {1: 1.15, 2: 1.08, 3: 1.00, 4: 0.92, 5: 0.85}


def horizon_multipliers(conn, snap, start_gw, horizon):
    """
    Total fixture-adjusted weight for each team across the next `horizon`
    gameweeks.

    WHY THIS MATTERS FOR A WILDCARD: the model scores a player's current form,
    not their upcoming schedule. Optimising a wildcard on a single gameweek
    would buy players with one kind fixture and a punishing run behind it.
    Summing over a horizon also handles the two cases that decide wildcards:
    a BLANK gameweek contributes nothing, and a DOUBLE contributes twice.
    """
    mult = {}
    for event, th, ta, dh, da in conn.execute(
        "SELECT event, team_h, team_a, difficulty_h, difficulty_a FROM fixtures"
        " WHERE snapshot_id=? AND event >= ? AND event < ?",
        (snap, start_gw, start_gw + horizon),
    ):
        if dh is not None:
            mult[th] = mult.get(th, 0.0) + FDR_MULT.get(dh, 1.0)
        if da is not None:
            mult[ta] = mult.get(ta, 0.0) + FDR_MULT.get(da, 1.0)
    return mult


def candidates(conn):
    """Players with a logged model prediction for the upcoming gameweek."""
    snap = conn.execute("SELECT MAX(snapshot_id) FROM players").fetchone()[0]
    gw = conn.execute("SELECT next_gw FROM snapshots WHERE id=?", (snap,)).fetchone()[0]
    df = pd.read_sql_query(
        """SELECT p.element_id, p.web_name AS name, p.position, p.price,
                  p.team_id, p.selected_by_percent AS owned, pr.predicted AS ep
           FROM players p
           JOIN predictions pr ON pr.element_id = p.element_id
                AND pr.gameweek = ? AND pr.model = 'two_stage_ml'
           WHERE p.snapshot_id = ?""",
        conn, params=(gw, snap),
    )
    return df.reset_index(drop=True), gw


def solve(df, budget, current_ids=None, free=1):
    """
    Build and solve the MILP.

    scipy.optimize.milp MINIMISES, so the objective is negated. Every
    constraint is expressed as lo <= A @ z <= hi.
    """
    n = len(df)
    ep = df["ep"].values
    price = df["price"].values
    pos = df["position"].values
    team = df["team_id"].values

    # z = [ x (n) | y (n) | c (n) | h (1) ]
    N = 3 * n + 1
    X = lambda i: i
    Y = lambda i: n + i
    C = lambda i: 2 * n + i
    H = 3 * n

    # ---- objective (maximise, hence negated) -------------------------------
    obj = np.zeros(N)
    obj[0:n] = BENCH_WEIGHT * ep            # value of merely owning a player
    obj[n:2 * n] = ep - BENCH_WEIGHT * ep   # extra value of starting them
    obj[2 * n:3 * n] = ep                   # captain scores twice
    obj[H] = -HIT
    obj = -obj

    rows, cons = [], []

    def add(coef_map, lo, hi):
        v = np.zeros(N)
        for k, val in coef_map.items():
            v[k] = val
        rows.append(v)
        cons.append((lo, hi))

    # squad size
    add({X(i): 1 for i in range(n)}, 15, 15)

    # squad composition by position
    for p, need in SQUAD.items():
        add({X(i): 1 for i in range(n) if pos[i] == p}, need, need)

    # budget
    add({X(i): price[i] for i in range(n)}, 0, budget)

    # at most three per club
    for t in np.unique(team):
        add({X(i): 1 for i in range(n) if team[i] == t}, 0, 3)

    # starting XI size
    add({Y(i): 1 for i in range(n)}, 11, 11)

    # formation limits
    for p in SQUAD:
        add({Y(i): 1 for i in range(n) if pos[i] == p}, XI_MIN[p], XI_MAX[p])

    # exactly one captain
    add({C(i): 1 for i in range(n)}, 1, 1)

    # y_i <= x_i  and  c_i <= y_i
    for i in range(n):
        add({Y(i): 1, X(i): -1}, -np.inf, 0)
        add({C(i): 1, Y(i): -1}, -np.inf, 0)

    # transfer penalty: h >= (15 - kept) - free
    if current_ids is not None:
        keep = {X(i): 1 for i in range(n) if df["element_id"].iloc[i] in current_ids}
        keep[H] = 1
        add(keep, 15 - free, np.inf)

    A = sparse.csr_matrix(np.array(rows))
    lo = np.array([c[0] for c in cons])
    hi = np.array([c[1] for c in cons])

    integrality = np.ones(N)                      # all variables integer
    bounds = Bounds(lb=np.zeros(N), ub=np.concatenate([np.ones(3 * n), [15]]))

    res = milp(c=obj, constraints=LinearConstraint(A, lo, hi),
               integrality=integrality, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"solver failed: {res.message}")
    return res


def show(df, res, gw, current_ids=None, horizon=None):
    n = len(df)
    z = np.round(res.x).astype(int)
    in_squad = [i for i in range(n) if z[i] == 1]
    starting = {i for i in range(n) if z[n + i] == 1}
    cap = [i for i in range(n) if z[2 * n + i] == 1][0]

    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    in_squad.sort(key=lambda i: (order[df["position"].iloc[i]], -df["ep"].iloc[i]))

    has_gw = "gw_ep" in df.columns
    label = f"GW{gw}-{gw+horizon-1} horizon" if horizon else f"GW{gw}"
    print(f"\n  OPTIMAL SQUAD — {label}")
    hdr = f"  {'':<3}{'player':<18}{'pos':<5}{'club':>5}{'price':>7}{'horiz':>7}"
    print(hdr + (f"{'GW'+str(gw):>7}" if has_gw else ""))
    print("  " + "-" * (len(hdr) + (7 if has_gw else 0)))
    # In horizon (wildcard) mode the squad is chosen for the whole run, but
    # CAPTAINCY IS A WEEKLY DECISION. Marking the solver's horizon-captain here
    # would contradict the "CAPTAIN for GWn" line below and show two different
    # captains without explaining why. Mark this week's captain instead.
    cap_shown = (max(starting, key=lambda i: df["gw_ep"].iloc[i])
                 if "gw_ep" in df.columns else cap)

    spend = 0.0
    for i in in_squad:
        r = df.iloc[i]
        spend += r["price"]
        mark = "C" if i == cap_shown else ("*" if i in starting else " ")
        line = (f"  {mark:<3}{safe(r['name'])[:17]:<18}{r['position']:<5}"
                f"{int(r['team_id']):>5}{r['price']:>7.1f}{r['ep']:>7.2f}")
        if has_gw:
            line += f"{r['gw_ep']:>7.2f}"
        print(line)

    xi_pts = sum(df["ep"].iloc[i] for i in starting) + df["ep"].iloc[cap]
    hits = int(z[3 * n])
    print(f"\n  * = starter, C = captain"
          + ("  (squad picked for the horizon; captain picked for this week)" if has_gw else ""))
    print(f"  spend {spend:.1f}m   |   XI value over {label} {xi_pts:.2f}"
          + (f"   |   transfer hits {hits} (-{hits*HIT:.0f})" if current_ids is not None else ""))
    if has_gw:
        # Captaincy is an immediate decision, so pick it on this gameweek's
        # expected points, not on horizon value.
        best_cap = max(starting, key=lambda i: df["gw_ep"].iloc[i])
        print(f"  CAPTAIN for GW{gw}: {safe(df['name'].iloc[best_cap])} "
              f"({df['gw_ep'].iloc[best_cap]:.2f} xPts this week)")

    if current_ids is not None:
        new_ids = set(df["element_id"].iloc[i] for i in in_squad)
        out = current_ids - new_ids
        inn = new_ids - current_ids
        name_of = dict(zip(df["element_id"], df["name"]))
        print(f"\n  TRANSFERS ({len(inn)}):")
        if not inn:
            print("    none — current squad is already optimal under these constraints")
        for o, i2 in zip(sorted(out), sorted(inn)):
            print(f"    OUT  {safe(name_of.get(o, o))[:18]:<20} ->  IN  {safe(name_of.get(i2, i2))}")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        df, gw = candidates(conn)
        if df.empty:
            print("  No predictions found. Run: python recommend.py --run")
            return

        budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 100.0

        if "--wildcard" in sys.argv:
            H = int(sys.argv[sys.argv.index("--horizon") + 1]) if "--horizon" in sys.argv else 5
            snap = conn.execute("SELECT MAX(snapshot_id) FROM players").fetchone()[0]
            mult = horizon_multipliers(conn, snap, gw, H)

            df["gw_ep"] = df["ep"]                              # immediate gameweek
            df["ep"] = df["ep"] * df["team_id"].map(mult).fillna(0.0)   # horizon value

            print(f"  WILDCARD — optimising over GW{gw}-{gw+H-1} ({H} gameweeks), budget {budget:.1f}m")
            print(f"  Player value = model form score x summed fixture difficulty over the horizon.")
            res = solve(df, budget=budget)
            show(df, res, gw, horizon=H)

        elif "--squad" in sys.argv:
            print(f"  Optimising a fresh 15 from {len(df)} candidates (budget {budget:.1f}m)")
            res = solve(df, budget=budget)
            show(df, res, gw)

        elif "--transfers" in sys.argv:
            free = int(sys.argv[sys.argv.index("--free") + 1]) if "--free" in sys.argv else 1
            bank = float(sys.argv[sys.argv.index("--bank") + 1]) if "--bank" in sys.argv else 0.0

            cur = pd.read_sql_query(
                "SELECT DISTINCT element_id FROM my_squad WHERE gameweek ="
                " (SELECT MAX(gameweek) FROM my_squad)", conn)
            current_ids = set(cur["element_id"].dropna().astype(int))
            owned = df[df["element_id"].isin(current_ids)]
            budget = owned["price"].sum() + bank
            print(f"  Current squad: {len(current_ids)} players, value {owned['price'].sum():.1f}m"
                  f" + bank {bank:.1f}m = {budget:.1f}m")
            print(f"  Free transfers: {free}   (each extra costs {HIT:.0f} pts)")
            res = solve(df, budget=budget, current_ids=current_ids, free=free)
            show(df, res, gw, current_ids)
        else:
            print(__doc__)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
