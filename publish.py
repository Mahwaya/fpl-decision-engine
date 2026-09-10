"""
Turn the database into small JSON files the web app can read.

WHY THIS EXISTS — the split that makes free hosting work
========================================================
Streamlit Community Cloud cannot run this project. Its filesystem is ephemeral,
so `fpl.db` is destroyed on every reboot; it has no scheduler, so nothing runs
unless a browser is open; and it sleeps after 12 quiet hours. A decision support
system that only thinks while you are looking at it is useless.

So the work is split:

    GitHub Actions  = the clock.  Collects, predicts, alerts, emails.
    Streamlit Cloud = the window. Renders what Actions already decided.

This script is the seam between them. It reads the 23 MB database and writes a
few hundred KB of JSON, which is committed to the repo. The web app then needs
no database, no scikit-learn and no model training — it reads JSON and draws.
That is what keeps the app inside 1 GB of RAM with a cold start measured in
seconds rather than minutes.

DESIGN RULE: this script computes nothing that the pipeline already computed.
It reads the `predictions` table that recommend.py wrote before the deadline,
rather than re-running the model — re-running it here would produce numbers
that disagree with the ones that were logged, and the logged ones are the
honest record.

Usage:
    python publish.py              # write site/data/*.json
    python publish.py --print      # write, and show a summary
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import alert

ROOT = Path(__file__).parent
DB_PATH = ROOT / "fpl.db"
OUT_DIR = ROOT / "site" / "data"

MODEL_NAME = "two_stage_ml"

# Email when a squad/watchlist player changes badly, or when the deadline is
# close enough that you still have time to act but might forget to look.
DEADLINE_WARN_HOURS = 26


def utcnow():
    return datetime.now(timezone.utc)


def parse_iso(s):
    """FPL deadlines come back as '...Z', which fromisoformat rejects."""
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def write(name, payload):
    """
    Write one JSON file.

    encoding is pinned to UTF-8 deliberately: the default on Windows is cp1252,
    which cannot represent the accented names in this data (Horníček, Rúben,
    João Pedro) and raises UnicodeEncodeError mid-write, leaving a truncated
    file behind. ensure_ascii=False keeps them readable in the committed diff.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False, default=str)
    return path


# --------------------------------------------------------------- sections


def build_meta(conn):
    snap, taken_at, cur_gw, next_gw, deadline = conn.execute(
        "SELECT id, taken_at, current_gw, next_gw, next_deadline"
        " FROM snapshots ORDER BY id DESC LIMIT 1"
    ).fetchone()

    dl = parse_iso(deadline)
    hours = (dl - utcnow()).total_seconds() / 3600 if dl else None

    counts = {}
    for table in ("history", "player_gw", "predictions", "players",
                  "snapshots", "my_squad", "watchlist"):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    return {
        "generated_at": utcnow().isoformat(),
        "snapshot_id": snap,
        "snapshot_taken_at": taken_at,
        "current_gw": cur_gw,
        "next_gw": next_gw,
        "deadline": deadline,
        "hours_to_deadline": round(hours, 2) if hours is not None else None,
        "deadline_passed": bool(dl and utcnow() > dl),
        "row_counts": counts,
    }


def build_squad(conn, next_gw):
    """
    The squad currently owned, with the model's expected points attached.

    Reads `my_squad` at its highest gameweek. That table is the single source
    of truth for ownership; when it fell out of date the alerter spent a week
    warning about players that had already been sold.
    """
    rows = conn.execute(
        """SELECT s.name, s.element_id, s.is_captain, s.is_vice, s.started,
                  p.position, p.price, p.selected_by_percent, p.status,
                  p.chance_next_round, p.news, t.short_name,
                  (SELECT predicted FROM predictions
                    WHERE element_id = s.element_id AND gameweek = ?
                      AND model = ?) AS ep
           FROM my_squad s
           LEFT JOIN players p
             ON p.element_id = s.element_id
            AND p.snapshot_id = (SELECT MAX(id) FROM snapshots)
           LEFT JOIN teams t
             ON t.team_id = p.team_id
            AND t.snapshot_id = p.snapshot_id
           WHERE s.gameweek = (SELECT MAX(gameweek) FROM my_squad)
           ORDER BY CASE p.position
                      WHEN 'GKP' THEN 1 WHEN 'DEF' THEN 2
                      WHEN 'MID' THEN 3 ELSE 4 END,
                    ep DESC""",
        (next_gw, MODEL_NAME),
    ).fetchall()

    squad = []
    for (name, eid, cap, vice, started, pos, price, owned, status,
         chance, news, club, ep) in rows:
        squad.append({
            "name": name, "element_id": eid, "position": pos,
            "club": club, "price": price, "owned": owned,
            "status": status, "chance": chance, "news": news,
            "is_captain": bool(cap), "is_vice": bool(vice),
            "started": started,
            "ep": round(ep, 2) if ep is not None else None,
            # A flagged player is a flagged player whatever the model thinks.
            "flagged": bool(news) or (status not in (None, "a")),
        })

    gw = conn.execute("SELECT MAX(gameweek) FROM my_squad").fetchone()[0]
    value = sum(p["price"] for p in squad if p["price"] is not None)
    return {
        "gameweek": gw,
        "squad": squad,
        "squad_value": round(value, 1),
        "count": len(squad),
    }


def build_recommendations(conn, next_gw, limit=40):
    """Top players by the model's expected points for the upcoming gameweek."""
    rows = conn.execute(
        """SELECT pr.element_id, pr.predicted, pr.made_at,
                  p.web_name, p.position, p.price, p.selected_by_percent,
                  p.status, p.chance_next_round, p.news, t.short_name,
                  EXISTS(SELECT 1 FROM my_squad m
                          WHERE m.element_id = pr.element_id
                            AND m.gameweek = (SELECT MAX(gameweek) FROM my_squad))
           FROM predictions pr
           JOIN players p
             ON p.element_id = pr.element_id
            AND p.snapshot_id = (SELECT MAX(id) FROM snapshots)
           LEFT JOIN teams t
             ON t.team_id = p.team_id AND t.snapshot_id = p.snapshot_id
           WHERE pr.gameweek = ? AND pr.model = ?
           ORDER BY pr.predicted DESC
           LIMIT ?""",
        (next_gw, MODEL_NAME, limit),
    ).fetchall()

    picks = [{
        "element_id": eid, "ep": round(ep, 2), "made_at": made_at,
        "name": name, "position": pos, "club": club, "price": price,
        "owned": owned, "status": status, "chance": chance, "news": news,
        "owned_by_me": bool(mine),
    } for (eid, ep, made_at, name, pos, price, owned,
           status, chance, news, club, mine) in rows]

    return {"gameweek": next_gw, "model": MODEL_NAME, "picks": picks}


def build_captain(conn, next_gw):
    """
    The captaincy call: highest expected points among players actually owned.

    Not highest ceiling. captain.py simulated a full season and ceiling
    strategies lost, because captain points double linearly — maximising E[2X]
    is exactly maximising E[X].
    """
    rows = conn.execute(
        """SELECT s.name, s.element_id, p.position, p.news, p.status,
                  (SELECT predicted FROM predictions
                    WHERE element_id = s.element_id AND gameweek = ?
                      AND model = ?) AS ep
           FROM my_squad s
           LEFT JOIN players p
             ON p.element_id = s.element_id
            AND p.snapshot_id = (SELECT MAX(id) FROM snapshots)
           WHERE s.gameweek = (SELECT MAX(gameweek) FROM my_squad)""",
        (next_gw, MODEL_NAME),
    ).fetchall()

    ranked = sorted(
        [{"name": n, "element_id": e, "position": pos, "ep": round(ep, 2),
          "flagged": bool(news) or (st not in (None, "a"))}
         for (n, e, pos, news, st, ep) in rows if ep is not None],
        key=lambda r: -r["ep"],
    )
    recorded = conn.execute(
        "SELECT name FROM my_squad WHERE gameweek = (SELECT MAX(gameweek) FROM my_squad)"
        " AND is_captain = 1"
    ).fetchone()

    return {
        "gameweek": next_gw,
        "model_pick": ranked[0] if ranked else None,
        "your_pick": recorded[0] if recorded else None,
        "agrees": bool(ranked and recorded and ranked[0]["name"] == recorded[0]),
        "ranked": ranked[:8],
    }


def build_accuracy(conn):
    """
    Every model's error on every gameweek that has both predictions and
    results. This is the part that keeps the system honest: predictions are
    written before a deadline and never touched afterwards, so the scoring
    below cannot be flattered after the fact.
    """
    models = [r[0] for r in conn.execute(
        "SELECT DISTINCT model FROM predictions ORDER BY model")]
    gws = [r[0] for r in conn.execute(
        "SELECT DISTINCT gameweek FROM predictions"
        " WHERE gameweek IN (SELECT DISTINCT gameweek FROM player_gw)"
        " ORDER BY gameweek")]

    scored = []
    for gw in gws:
        for m in models:
            rows = conn.execute(
                """SELECT pr.predicted, a.total_points
                   FROM predictions pr
                   JOIN player_gw a
                     ON a.element_id = pr.element_id AND a.gameweek = pr.gameweek
                   WHERE pr.gameweek = ? AND pr.model = ?""",
                (gw, m),
            ).fetchall()
            if not rows:
                continue
            errs = [abs(p - (a or 0)) for p, a in rows]
            hi = [abs(p - (a or 0)) for p, a in rows if (a or 0) >= 5]
            scored.append({
                "gameweek": gw, "model": m, "n": len(rows),
                "mae": round(sum(errs) / len(errs), 3),
                "mae_high_return": round(sum(hi) / len(hi), 3) if hi else None,
            })

    return {"models": models, "scored": scored,
            "pending": [g for g in
                        [r[0] for r in conn.execute(
                            "SELECT DISTINCT gameweek FROM predictions ORDER BY gameweek")]
                        if g not in gws]}


def build_season(conn):
    """Percival's own results — the benchmark the model has to beat."""
    rows = conn.execute(
        "SELECT gameweek, total_points, transfers, formation, decided_by"
        " FROM my_gameweeks ORDER BY gameweek"
    ).fetchall()

    gws = []
    for gw, pts, tr, form, how in rows:
        cap = conn.execute(
            "SELECT name, points FROM my_squad WHERE gameweek=? AND is_captain=1", (gw,)
        ).fetchone()
        gws.append({
            "gameweek": gw, "points": pts, "transfers": tr,
            "formation": form, "decided_by": how,
            "captain": cap[0] if cap else None,
            "captain_points": cap[1] if cap else None,
            "played": pts is not None,
        })

    played = [g for g in gws if g["played"]]
    return {
        "gameweeks": gws,
        "total_points": sum(g["points"] for g in played),
        "played": len(played),
        "average": round(sum(g["points"] for g in played) / len(played), 1) if played else None,
    }


def build_alerts(conn, hours=24):
    data = alert.collect(conn, hours=hours)
    if data is None:
        return {"available": False, "alerts": [], "price_moves": [], "urgent": []}
    data["available"] = True
    data["window_hours"] = hours
    # Price moves are long and low-value on a phone; the biggest movers suffice.
    data["price_moves"] = data["price_moves"][:12]
    return data


def build_notify(meta, alerts, squad, captain):
    """
    Decide whether this run should email, and what it should say.

    The workflow reads `should_email` and skips the send step when it is false,
    so a quiet day costs nothing and does not train you to ignore the inbox.
    """
    reasons, lines = [], []

    urgent = [a for a in alerts.get("urgent", [])]
    if urgent:
        reasons.append(f"{len(urgent)} squad/watchlist change(s)")
        lines.append("AVAILABILITY — players you own or watch:")
        for a in urgent:
            lines.append(f"  [{a['tag']}] {a['severity']}  {a['name']} ({a['position']})"
                         f"  {a['message']}")
        lines.append("")

    hrs = meta.get("hours_to_deadline")
    if hrs is not None and 0 < hrs <= DEADLINE_WARN_HOURS:
        reasons.append(f"GW{meta['next_gw']} deadline in {hrs:.0f}h")
        lines.append(f"DEADLINE — GW{meta['next_gw']} in {hrs:.1f} hours "
                     f"({meta['deadline']}).")
        lines.append("")

    flagged = [p for p in squad["squad"] if p["flagged"]]
    if flagged:
        lines.append("FLAGGED IN YOUR SQUAD:")
        for p in flagged:
            ch = "n/a" if p["chance"] is None else f"{p['chance']}%"
            lines.append(f"  {p['name']} ({p['position']}) chance {ch} — {p['news']}")
        lines.append("")

    if captain and captain.get("model_pick"):
        mp = captain["model_pick"]
        lines.append(f"CAPTAIN — model says {mp['name']} ({mp['ep']:.2f} xPts).")
        if captain.get("your_pick") and not captain["agrees"]:
            lines.append(f"  You have {captain['your_pick']} recorded. Model disagrees.")
        lines.append("")

    lines.append(f"Snapshot #{meta['snapshot_id']} at {meta['snapshot_taken_at']}")

    subject = "FPL: " + "; ".join(reasons) if reasons else "FPL: nothing urgent"
    return {
        "should_email": bool(reasons),
        "reasons": reasons,
        "subject": subject[:180],
        "body": "\n".join(lines),
    }


# ------------------------------------------------------------------- main


def run(conn, verbose=False):
    meta = build_meta(conn)
    next_gw = meta["next_gw"]

    squad = build_squad(conn, next_gw)
    recs = build_recommendations(conn, next_gw)
    captain = build_captain(conn, next_gw)
    accuracy = build_accuracy(conn)
    season = build_season(conn)
    alerts = build_alerts(conn)
    notify = build_notify(meta, alerts, squad, captain)

    written = []
    for name, payload in [
        ("meta.json", meta),
        ("squad.json", squad),
        ("recommendations.json", recs),
        ("captain.json", captain),
        ("accuracy.json", accuracy),
        ("season.json", season),
        ("alerts.json", alerts),
        ("notify.json", notify),
    ]:
        written.append(write(name, payload))

    total = sum(p.stat().st_size for p in written)
    print(f"  Wrote {len(written)} files to {OUT_DIR.relative_to(ROOT)} "
          f"({total/1024:.0f} KB total)")
    if verbose:
        for p in written:
            print(f"    {p.name:<24}{p.stat().st_size/1024:>7.1f} KB")
        print(f"\n  GW{next_gw}  deadline {meta['deadline']}  "
              f"({meta['hours_to_deadline']}h)")
        print(f"  squad {squad['count']} players, value {squad['squad_value']}m")
        print(f"  {len(recs['picks'])} recommendations, "
              f"{len(alerts.get('alerts', []))} alerts, "
              f"{len(alerts.get('urgent', []))} urgent")
        print(f"  email: {notify['should_email']}  ({notify['subject']})")
    return written


def main():
    if not DB_PATH.exists():
        print(f"  No database at {DB_PATH}. Run fpl_collect.py first.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    try:
        run(conn, verbose="--print" in sys.argv)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
