"""
Phase 4 — team-news alerter.

WHAT PROBLEM THIS SOLVES: the model predicts from past minutes. It cannot know
that a player picked up a knock on Wednesday, or that his manager ruled him out
in Friday's press conference. That information IS already in the data we
collect — every snapshot stores `news`, `news_added`, `chance_of_playing_next_
round` and `status` — but nothing ever compared one snapshot to the next, so a
change could land and go unnoticed until after the deadline.

This diffs snapshots and reports what changed, ranked by how much it should
matter to you.

WHY TIMING MATTERS: for a Saturday deadline, pre-match press conferences happen
Thursday and Friday. Run this Thursday evening and Friday evening — running it
on a Tuesday tells you nothing, because the news does not exist yet.

Priorities:
    [SQUAD]  a player you own                       -- act on these
    [WATCH]  a player on your watchlist             -- e.g. a planned wildcard
    [OWNED]  widely owned, so it moves the field
    [-]      everyone else

Usage:
    python alert.py --check                 changes since the previous snapshot
    python alert.py --check --hours 48      changes over the last 48 hours
    python alert.py --watch-wildcard        save the current optimiser squad to watch
    python alert.py --watchlist             show the watchlist
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "fpl.db"

try:
    from optimise import safe
except Exception:                                   # keep the alerter standalone
    def safe(s):
        import unicodedata
        return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def create_schema(conn):
    conn.execute(
        """CREATE TABLE IF NOT EXISTS watchlist (
               element_id INTEGER PRIMARY KEY,
               label      TEXT,
               added_at   TEXT DEFAULT CURRENT_TIMESTAMP
           )"""
    )


def snapshot_pair(conn, hours=None):
    """The snapshot to compare against, and the latest one."""
    latest = conn.execute("SELECT MAX(id) FROM snapshots").fetchone()[0]
    if hours:
        row = conn.execute(
            "SELECT id FROM snapshots WHERE taken_at <= datetime('now', ?)"
            " ORDER BY taken_at DESC LIMIT 1", (f"-{hours} hours",)
        ).fetchone()
        prev = row[0] if row else None
        if prev is None:      # not enough history; fall back to the earliest
            prev = conn.execute("SELECT MIN(id) FROM snapshots").fetchone()[0]
    else:
        row = conn.execute(
            "SELECT id FROM snapshots WHERE id < ? ORDER BY id DESC LIMIT 1", (latest,)
        ).fetchone()
        prev = row[0] if row else None
    return prev, latest


def classify(old_news, new_news, old_chance, new_chance, old_status, new_status):
    """
    Turn a raw change into a severity and a human sentence.

    chance is NULL when a player is fully fit, which is why NULL is normalised
    to 100 rather than treated as missing — otherwise 'cleared to play' looks
    like data loss.
    """
    oc = 100 if old_chance is None else old_chance
    nc = 100 if new_chance is None else new_chance
    on, nn = (old_news or "").strip(), (new_news or "").strip()

    if nc == 0 and oc > 0:
        return "CRITICAL", f"RULED OUT ({nn or new_status})"
    if on and not nn and nc >= 100:
        return "GOOD", "cleared - news removed, fit again"
    if nc > oc:
        return "GOOD", f"improved {oc}% -> {nc}%"
    if nc < oc:
        return "WARNING", f"worsened {oc}% -> {nc}% ({nn})"
    if nn and nn != on:
        return "WARNING", f"new news: {nn}"
    if old_status != new_status:
        return "INFO", f"status {old_status} -> {new_status}"
    return None, None


SEV_RANK = {"CRITICAL": 0, "WARNING": 1, "INFO": 2, "GOOD": 3}
TAG_RANK = {"SQUAD": 0, "WATCH": 1, "OWNED": 2, "-": 3}


def table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def collect(conn, hours=None):
    """
    Diff two snapshots and return the result as data, not printed text.

    WHY THIS IS SEPARATE FROM check(): the alerter, the JSON publisher and the
    email notifier all need the same answer. When each formats its own version
    of "what changed" they drift apart silently — the email says one thing, the
    web page another, and neither is obviously wrong. One function computes it,
    three callers render it.

    Returns a dict, or None when there is nothing to compare against.
    """
    prev, latest = snapshot_pair(conn, hours)
    if prev is None:
        return None

    t_prev, t_latest = conn.execute(
        "SELECT (SELECT taken_at FROM snapshots WHERE id=?),"
        "       (SELECT taken_at FROM snapshots WHERE id=?)", (prev, latest)
    ).fetchone()

    # A database rebuilt from scratch — the cloud runner does exactly this when
    # the stored copy is missing — has no my_squad until record_my_team.py has
    # run. Degrade to "no squad known" and SAY SO, rather than crashing the
    # scheduled run or, worse, quietly reporting zero squad alerts as if all
    # were well.
    if not table_exists(conn, "my_squad"):
        print("  NOTE: my_squad does not exist — no [SQUAD] tagging this run."
              " Run record_my_team.py to populate it.")
        squad = set()
    else:
        squad = {r[0] for r in conn.execute(
            "SELECT DISTINCT element_id FROM my_squad WHERE gameweek ="
            " (SELECT MAX(gameweek) FROM my_squad) AND element_id IS NOT NULL")}
    watch = {r[0]: r[1] for r in conn.execute("SELECT element_id, label FROM watchlist")}

    rows = conn.execute(
        """SELECT c.element_id, c.web_name, c.position, c.price, c.selected_by_percent,
                  p.news, c.news, p.chance_next_round, c.chance_next_round,
                  p.status, c.status, p.price
           FROM players c JOIN players p
             ON p.element_id = c.element_id AND p.snapshot_id = ?
           WHERE c.snapshot_id = ?""",
        (prev, latest),
    ).fetchall()

    alerts, price_moves = [], []
    for (eid, name, pos, price, owned, on_, nn_, oc, nc, os_, ns_, oprice) in rows:
        sev, msg = classify(on_, nn_, oc, nc, os_, ns_)
        if sev:
            if eid in squad:
                tag = "SQUAD"
            elif eid in watch:
                tag = "WATCH"
            elif (owned or 0) >= 8:
                tag = "OWNED"
            else:
                tag = "-"
            alerts.append({
                "element_id": eid, "name": name, "position": pos,
                "price": price, "owned": owned or 0.0,
                "severity": sev, "tag": tag, "message": msg,
                "_sort": (TAG_RANK[tag], SEV_RANK[sev], -(owned or 0)),
            })
        if oprice is not None and price is not None and abs(price - oprice) >= 0.05:
            price_moves.append({
                "name": name, "old": oprice, "new": price,
                "owned": owned or 0.0, "direction": "up" if price > oprice else "down",
            })

    alerts.sort(key=lambda a: a["_sort"])
    for a in alerts:
        a.pop("_sort")
    price_moves.sort(key=lambda p: -p["owned"])

    urgent = [a for a in alerts
              if a["tag"] in ("SQUAD", "WATCH") and a["severity"] in ("CRITICAL", "WARNING")]

    return {
        "prev_snapshot": prev, "latest_snapshot": latest,
        "prev_taken_at": t_prev, "latest_taken_at": t_latest,
        "alerts": alerts, "price_moves": price_moves, "urgent": urgent,
    }


def check(conn, hours=None):
    data = collect(conn, hours)
    if data is None:
        print("  Only one snapshot exists — nothing to compare yet.")
        return

    print(f"  Comparing snapshot #{data['prev_snapshot']} ({data['prev_taken_at'][:16]})"
          f"  ->  #{data['latest_snapshot']} ({data['latest_taken_at'][:16]})")

    alerts = data["alerts"]
    if not alerts:
        print("\n  No availability changes.")
    else:
        print(f"\n  {len(alerts)} availability change(s), most important first:\n")
        print(f"  {'':<8}{'sev':<10}{'player':<16}{'pos':<5}{'own%':>6}   what changed")
        print("  " + "-" * 74)
        for a in alerts:
            t = f"[{a['tag']}]" if a["tag"] != "-" else ""
            print(f"  {t:<8}{a['severity']:<10}{safe(a['name'])[:15]:<16}"
                  f"{a['position']:<5}{a['owned']:>6.1f}   {safe(a['message'])[:40]}")

    if data["price_moves"]:
        print(f"\n  Price changes ({len(data['price_moves'])}):")
        for p in data["price_moves"][:8]:
            print(f"    {safe(p['name'])[:15]:<16}{p['old']:.1f} -> {p['new']:.1f}"
                  f"  ({p['direction']})  {p['owned']:.1f}% owned")

    if data["urgent"]:
        print(f"\n  >>> {len(data['urgent'])} change(s) affect players you own or are "
              f"watching. Review before the deadline.")


def watch_wildcard(conn):
    """Save the current optimiser wildcard squad so the alerter tracks it."""
    import optimise
    df, gw = optimise.candidates(conn)
    snap = conn.execute("SELECT MAX(snapshot_id) FROM players").fetchone()[0]
    mult = optimise.horizon_multipliers(conn, snap, gw, 4)
    df["gw_ep"] = df["ep"]
    df["ep"] = df["ep"] * df["team_id"].map(mult).fillna(0.0)
    res = optimise.solve(df, budget=99.6)
    n = len(df)
    import numpy as np
    z = np.round(res.x).astype(int)
    picked = [int(df["element_id"].iloc[i]) for i in range(n) if z[i] == 1]
    conn.execute("DELETE FROM watchlist WHERE label='wildcard'")
    conn.executemany(
        "INSERT OR REPLACE INTO watchlist (element_id, label) VALUES (?, 'wildcard')",
        [(e,) for e in picked],
    )
    conn.commit()
    print(f"  Watching {len(picked)} players from the recommended wildcard squad.")
    show_watchlist(conn)


def show_watchlist(conn):
    snap = conn.execute("SELECT MAX(snapshot_id) FROM players").fetchone()[0]
    rows = conn.execute(
        """SELECT w.label, p.web_name, p.position, p.price, p.selected_by_percent,
                  p.status, p.chance_next_round, p.news
           FROM watchlist w JOIN players p
             ON p.element_id = w.element_id AND p.snapshot_id = ?
           ORDER BY w.label, p.position, p.price DESC""", (snap,)
    ).fetchall()
    if not rows:
        print("  Watchlist is empty.")
        return
    print(f"\n  {'label':<10}{'player':<16}{'pos':<5}{'price':>7}   status")
    print("  " + "-" * 56)
    for label, name, pos, price, owned, status, chance, news in rows:
        note = safe(news)[:30] if news else "ok"
        print(f"  {label:<10}{safe(name)[:15]:<16}{pos:<5}{price:>6.1f}m   {note}")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        create_schema(conn)
        if "--check" in sys.argv:
            hours = None
            if "--hours" in sys.argv:
                hours = int(sys.argv[sys.argv.index("--hours") + 1])
            check(conn, hours)
        elif "--watch-wildcard" in sys.argv:
            watch_wildcard(conn)
        elif "--watchlist" in sys.argv:
            show_watchlist(conn)
        else:
            print(__doc__)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
