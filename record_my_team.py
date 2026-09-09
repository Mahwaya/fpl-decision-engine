"""
Record Percival's actual FPL squad and results, gameweek by gameweek.

WHY THIS MATTERS: this is the benchmark. Any model we build later has to beat
these decisions, not just beat a naive baseline. Without a record of what you
actually picked — and what it scored — there's no way to know whether the
system is helping or just producing confident-looking numbers.

Team: "Progeny"
Source: FPL app screenshots, transcribed 2026-09-07.

Usage:
    python record_my_team.py            # write the gameweeks below into fpl.db
    python record_my_team.py --show     # print what's recorded
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "fpl.db"

# --- The data -------------------------------------------------------------
# Each entry: (name, points, started?, is_captain, is_vice)
# points of None = did not play / no score shown.

GAMEWEEKS = {
    1: {
        "total_points": 53,
        "transfers": 0,
        "formation": "3-5-2",
        "squad": [
            ("Martinez",     None, True,  False, False),
            ("Virgil",          2, True,  False, False),
            ("Cash",           -1, True,  False, False),
            ("Shaw",            1, True,  False, False),
            ("B.Fernandes",     2, True,  False, False),
            ("Palmer",         13, True,  False, True),
            ("Szoboszlai",      8, True,  False, False),
            ("Mbeumo",          2, True,  False, False),
            ("Rice",            3, True,  False, False),
            ("Wood",            1, True,  False, False),
            ("João Pedro",     22, True,  True,  False),
            ("Phillips",     None, False, False, False),
            ("Palestra",     None, False, False, False),
            ("Gyökeres",     None, False, False, False),
            ("Mykolenko",       6, False, False, False),
        ],
    },
    2: {
        "total_points": 79,
        "transfers": 1,
        "formation": "3-5-2",
        "squad": [
            ("Martinez",        2, True,  False, False),
            ("Virgil",          1, True,  False, False),
            ("Shaw",            2, True,  False, False),
            ("Mykolenko",       4, True,  False, False),
            ("B.Fernandes",    23, True,  False, True),
            ("Palmer",          7, True,  False, False),
            ("Szoboszlai",      4, True,  False, False),
            ("Mbeumo",         11, True,  False, False),
            ("Rice",            5, True,  False, False),
            ("Havertz",         2, True,  False, False),
            ("João Pedro",     18, True,  True,  False),
            ("Phillips",     None, False, False, False),
            ("Wood",         None, False, False, False),
            ("Cash",            2, False, False, False),
            ("Palestra",     None, False, False, False),
        ],
    },
    3: {
        "total_points": 48,
        "transfers": 1,
        "formation": "4-4-2",
        "squad": [
            ("Martinez",        3, True,  False, False),
            ("Virgil",          6, True,  False, False),
            ("Cash",            6, True,  False, False),
            ("Shaw",            4, True,  False, False),
            ("Hall",            4, True,  False, False),
            ("B.Fernandes",     4, True,  True,  False),
            ("Palmer",          1, True,  False, False),
            ("Mbeumo",          8, True,  False, False),
            ("Szoboszlai",      3, True,  False, False),
            ("João Pedro",      1, True,  False, True),
            ("Havertz",         8, True,  False, False),
            ("Phillips",     None, False, False, False),
            ("Wood",            1, False, False, False),
            ("Rice",            5, False, False, False),
            ("Mykolenko",       1, False, False, False),
        ],
    },
}


def create_schema(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS my_gameweeks (
            gameweek     INTEGER PRIMARY KEY,
            total_points INTEGER,
            transfers    INTEGER,
            formation    TEXT,
            decided_by   TEXT DEFAULT 'gut'   -- 'gut' or 'model', so we can compare later
        );

        CREATE TABLE IF NOT EXISTS my_squad (
            gameweek   INTEGER NOT NULL,
            name       TEXT NOT NULL,
            element_id INTEGER,               -- matched to the FPL API where possible
            points     INTEGER,               -- NULL = did not play
            started    INTEGER,
            is_captain INTEGER,
            is_vice    INTEGER,
            PRIMARY KEY (gameweek, name)
        );
        """
    )


# FPL web_names are NOT unique. There are two "Palmer" (Cole Palmer MID, and a
# goalkeeper) and two "Martinez" (Emiliano Martinez GKP, and a defender).
# Matching on name alone silently picked the wrong player. Position is what
# disambiguates them, so it is recorded explicitly here.
POSITIONS = {
    "Martinez": "GKP", "Phillips": "GKP",
    "Virgil": "DEF", "Cash": "DEF", "Shaw": "DEF", "Hall": "DEF",
    "Mykolenko": "DEF", "Palestra": "DEF",
    "B.Fernandes": "MID", "Palmer": "MID", "Szoboszlai": "MID",
    "Mbeumo": "MID", "Rice": "MID",
    "Wood": "FWD", "João Pedro": "FWD", "Havertz": "FWD", "Gyökeres": "FWD",
}


def match_element_ids(conn):
    """
    Link squad names to FPL element_ids.

    Must match on (web_name, position). Name alone is ambiguous — see POSITIONS
    above. If a name is still ambiguous after adding position, we refuse to
    guess and report it, because a silent wrong match corrupts every downstream
    join without ever looking broken.
    """
    row = conn.execute("SELECT MAX(snapshot_id) FROM players").fetchone()
    if not row or row[0] is None:
        print("  (no snapshot yet — run fpl_collect.py first to enable ID matching)")
        return
    snap = row[0]

    unmatched, ambiguous, fixed = [], [], 0
    for (name,) in conn.execute("SELECT DISTINCT name FROM my_squad").fetchall():
        pos = POSITIONS.get(name)
        if pos:
            cands = conn.execute(
                "SELECT element_id, price FROM players"
                " WHERE snapshot_id=? AND web_name=? AND position=?",
                (snap, name, pos),
            ).fetchall()
        else:
            cands = conn.execute(
                "SELECT element_id, price FROM players WHERE snapshot_id=? AND web_name=?",
                (snap, name),
            ).fetchall()

        if len(cands) == 1:
            conn.execute("UPDATE my_squad SET element_id=? WHERE name=?", (cands[0][0], name))
            fixed += 1
        elif len(cands) > 1:
            ambiguous.append(f"{name} ({len(cands)} candidates)")
        else:
            unmatched.append(name)

    conn.commit()
    print(f"  Matched {fixed} players by (name, position).")
    if ambiguous:
        print(f"  AMBIGUOUS, not guessed: {', '.join(ambiguous)}")
    if unmatched:
        print(f"  Could not match: {', '.join(unmatched)}")


def record(conn):
    for gw, data in GAMEWEEKS.items():
        conn.execute(
            "INSERT OR REPLACE INTO my_gameweeks (gameweek, total_points, transfers, formation, decided_by)"
            " VALUES (?,?,?,?,'gut')",
            (gw, data["total_points"], data["transfers"], data["formation"]),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO my_squad (gameweek, name, points, started, is_captain, is_vice)"
            " VALUES (?,?,?,?,?,?)",
            [(gw, n, p, int(s), int(c), int(v)) for (n, p, s, c, v) in data["squad"]],
        )
    conn.commit()
    print(f"  Recorded {len(GAMEWEEKS)} gameweeks.")
    match_element_ids(conn)


def show(conn):
    rows = conn.execute(
        "SELECT gameweek, total_points, transfers, formation FROM my_gameweeks ORDER BY gameweek"
    ).fetchall()
    if not rows:
        print("Nothing recorded yet. Run: python record_my_team.py")
        return

    total = sum(r[1] for r in rows)
    print("PROGENY - season so far (all decisions made on gut feel)\n")
    print(f"  {'GW':<4}{'Points':>8}{'Transfers':>11}  Formation   Captain (pts)")
    for gw, pts, tr, form in rows:
        cap = conn.execute(
            "SELECT name, points FROM my_squad WHERE gameweek=? AND is_captain=1", (gw,)
        ).fetchone()
        cap_s = f"{cap[0]} ({cap[1]})" if cap else "-"
        print(f"  {gw:<4}{pts:>8}{tr:>11}  {form:<11} {cap_s}")

    print(f"\n  Total: {total} pts over {len(rows)} GWs   |   Average: {total/len(rows):.1f} per GW")

    print("\n  Captaincy review (the biggest single lever in FPL):")
    total_lost = 0
    for gw, pts, _, _ in rows:
        # Compare RAW scores. The captain's stored points are already doubled,
        # so halve them; everyone else's are already raw. Comparing a halved
        # captain against other players' displayed scores is not like-for-like.
        squad = conn.execute(
            "SELECT name, points, is_captain FROM my_squad"
            " WHERE gameweek=? AND started=1 AND points IS NOT NULL", (gw,)
        ).fetchall()
        if not squad:
            continue
        raws = [(n, (p / 2 if c else p)) for n, p, c in squad]
        cap = next(((n, r) for (n, r), (_, _, c) in zip(raws, squad) if c), None)
        if not cap:
            continue
        best_name, best_raw = max(raws, key=lambda x: x[1])
        lost = 2 * (best_raw - cap[1])
        total_lost += lost
        if lost <= 0:
            note = "optimal choice"
        else:
            note = f"{best_name} raw {best_raw:.0f} would have given +{lost:.0f}"
        print(f"    GW{gw}: {cap[0]:<14} raw {cap[1]:>4.0f}  ->  {note}")
    print(f"\n    Points left on the table by captaincy alone: {total_lost:.0f}")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        create_schema(conn)
        if "--show" in sys.argv:
            show(conn)
        else:
            record(conn)
            print()
            show(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
