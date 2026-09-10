"""
FPL Decision Engine — the web front end.

READ THIS BEFORE ADDING ANYTHING HEAVY HERE
===========================================
This app deliberately does no work. It opens no database, trains no model and
imports neither scikit-learn nor scipy. It reads a handful of small JSON files
that GitHub Actions produced on a schedule, and draws them.

That is not laziness, it is the constraint that makes free hosting viable:

  * Streamlit Community Cloud's filesystem is ephemeral, so a 23 MB fpl.db
    committed here would be wiped on every reboot anyway.
  * The free tier caps memory at 1 GB. Training the two-stage model needs far
    more headroom than that leaves.
  * Apps sleep after 12 quiet hours, so anything that only happens "when the
    page loads" happens roughly never.

Every decision this page shows was already made by the pipeline, before the
deadline, and logged. The page is a window onto that record, not a calculator.
If you find yourself wanting to compute something here, compute it in
publish.py instead and read the answer.

Run locally:   streamlit run app.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "site" / "data"

st.set_page_config(
    page_title="FPL Decision Engine",
    page_icon="⚽",
    layout="wide",
)


# ------------------------------------------------------------------ data

@st.cache_data(ttl=300)
def load(name):
    """Read one published JSON file. Missing files are not fatal."""
    path = DATA / f"{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_iso(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def humanise(hours):
    if hours is None:
        return "unknown"
    if hours < 0:
        return "passed"
    d, h = divmod(hours, 24)
    if d >= 1:
        return f"{int(d)}d {int(h)}h"
    m = (hours - int(hours)) * 60
    return f"{int(hours)}h {int(m)}m"


meta = load("meta")

if meta is None:
    st.title("⚽ FPL Decision Engine")
    st.warning(
        "No published data yet. The GitHub Actions workflow writes "
        "`site/data/*.json` on its schedule — once it has run at least once, "
        "this page fills in."
    )
    st.stop()


# ---------------------------------------------------------------- header

st.title("⚽ FPL Decision Engine")

# The countdown is recomputed from the deadline on every page load rather than
# read from the JSON: `hours_to_deadline` was true when the pipeline ran, which
# may have been hours ago, and a stale countdown is worse than none.
deadline = parse_iso(meta.get("deadline"))
live_hours = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600 if deadline else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Next gameweek", f"GW{meta.get('next_gw', '?')}")
c2.metric("Deadline in", humanise(live_hours))
c3.metric("Snapshot", f"#{meta.get('snapshot_id', '?')}")

generated = parse_iso(meta.get("generated_at"))
age = (datetime.now(timezone.utc) - generated).total_seconds() / 3600 if generated else None
c4.metric("Data age", humanise(age) if age is not None else "unknown")

if live_hours is not None and live_hours < 0:
    st.info(f"The GW{meta.get('next_gw')} deadline has passed. "
            "Results are scored once the gameweek finishes.")
elif live_hours is not None and live_hours < 26:
    st.warning(f"**GW{meta.get('next_gw')} deadline in {humanise(live_hours)}** — "
               f"{meta.get('deadline')}")

if age is not None and age > 14:
    st.error(
        f"This data is {humanise(age)} old. The scheduled workflow runs twice "
        "daily, so anything beyond about 14 hours means a run failed — check "
        "the Actions tab before trusting what is below."
    )


tab_squad, tab_picks, tab_news, tab_model, tab_season = st.tabs(
    ["Squad & captain", "Recommendations", "Team news", "Model accuracy", "My season"]
)


# ----------------------------------------------------------- squad tab

with tab_squad:
    squad = load("squad")
    captain = load("captain")

    if not squad or not squad.get("squad"):
        st.info("No squad recorded yet.")
    else:
        left, right = st.columns([3, 2])

        with left:
            st.subheader(f"Squad — GW{squad.get('gameweek')}")
            df = pd.DataFrame(squad["squad"])
            df["player"] = df.apply(
                lambda r: f"{r['name']}"
                          + (" (C)" if r["is_captain"] else "")
                          + (" (V)" if r["is_vice"] else "")
                          + ("  ⚠" if r["flagged"] else ""),
                axis=1,
            )
            view = df[["player", "position", "club", "price", "owned", "ep"]].rename(
                columns={"position": "pos", "price": "£m",
                         "owned": "own %", "ep": "xPts"}
            )
            st.dataframe(
                view, hide_index=True, width="stretch",
                column_config={
                    "xPts": st.column_config.NumberColumn(format="%.2f"),
                    "£m": st.column_config.NumberColumn(format="%.1f"),
                    "own %": st.column_config.NumberColumn(format="%.1f"),
                },
            )
            st.caption(f"Squad value £{squad.get('squad_value')}m · "
                       f"{squad.get('count')} players · "
                       "xPts is the two-stage model's expected points, logged before the deadline")

        with right:
            st.subheader("Captain")
            if not captain or not captain.get("model_pick"):
                st.info("No captaincy prediction available.")
            else:
                mp = captain["model_pick"]
                st.metric(f"Model picks {mp['name']}",
                          f"{mp['ep']:.2f} xPts",
                          delta=f"doubles to {2*mp['ep']:.2f}")

                if captain.get("your_pick"):
                    if captain.get("agrees"):
                        st.success(f"You have **{captain['your_pick']}** — the model agrees.")
                    else:
                        st.warning(
                            f"You have **{captain['your_pick']}** recorded; "
                            f"the model prefers **{mp['name']}**."
                        )

                st.caption(
                    "Chosen by highest expected points, not highest ceiling. "
                    "A full-season simulation (captain.py) found ceiling "
                    "strategies lose — captain points double linearly, so "
                    "maximising E[2X] is just maximising E[X]."
                )
                cdf = pd.DataFrame(captain.get("ranked", []))
                if not cdf.empty:
                    cdf["name"] = cdf.apply(
                        lambda r: r["name"] + ("  ⚠" if r["flagged"] else ""), axis=1)
                    st.dataframe(
                        cdf[["name", "position", "ep"]].rename(
                            columns={"position": "pos", "ep": "xPts"}),
                        hide_index=True, width="stretch",
                        column_config={"xPts": st.column_config.NumberColumn(format="%.2f")},
                    )

        flagged = [p for p in squad["squad"] if p["flagged"]]
        if flagged:
            st.subheader("Flagged in your squad")
            for p in flagged:
                ch = "unknown" if p["chance"] is None else f"{p['chance']}%"
                st.error(f"**{p['name']}** ({p['position']}) — chance of playing {ch} — {p['news']}")


# ------------------------------------------------------ recommendations

with tab_picks:
    recs = load("recommendations")
    if not recs or not recs.get("picks"):
        st.info("No recommendations published yet.")
    else:
        st.subheader(f"Top {len(recs['picks'])} by expected points — GW{recs.get('gameweek')}")

        df = pd.DataFrame(recs["picks"])
        cols = st.columns(4)
        pos_filter = cols[0].multiselect(
            "Position", ["GKP", "DEF", "MID", "FWD"], default=["GKP", "DEF", "MID", "FWD"])
        max_price = cols[1].slider(
            "Max price (£m)", 3.5, 16.0,
            float(df["price"].max()) if df["price"].notna().any() else 16.0, 0.1)
        hide_flagged = cols[2].checkbox("Hide flagged players", value=True)
        mine_only = cols[3].checkbox("Only players I own", value=False)

        f = df[df["position"].isin(pos_filter) & (df["price"] <= max_price)]
        if hide_flagged:
            f = f[f["news"].isna() | (f["news"] == "")]
        if mine_only:
            f = f[f["owned_by_me"]]

        f = f.copy()
        f["player"] = f.apply(
            lambda r: f"{r['name']}" + ("  ✓" if r["owned_by_me"] else ""), axis=1)
        f["value"] = (f["ep"] / f["price"]).round(3)

        st.dataframe(
            f[["player", "position", "club", "price", "owned", "ep", "value", "news"]].rename(
                columns={"position": "pos", "price": "£m", "owned": "own %",
                         "ep": "xPts", "value": "xPts/£m", "news": "status"}),
            hide_index=True, width="stretch", height=520,
            column_config={
                "xPts": st.column_config.NumberColumn(format="%.2f"),
                "£m": st.column_config.NumberColumn(format="%.1f"),
                "own %": st.column_config.NumberColumn(format="%.1f"),
                "xPts/£m": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption("✓ = already in your squad. Predictions were written to the "
                   "database before the deadline and are never edited afterwards.")


# -------------------------------------------------------------- news tab

with tab_news:
    alerts = load("alerts")
    if not alerts or not alerts.get("available"):
        st.info("Not enough snapshots to compare yet.")
    else:
        st.caption(
            f"Comparing snapshot #{alerts.get('prev_snapshot')} → "
            f"#{alerts.get('latest_snapshot')} "
            f"(last {alerts.get('window_hours')}h)"
        )

        urgent = alerts.get("urgent", [])
        if urgent:
            st.subheader(f"{len(urgent)} change(s) affecting you")
            for a in urgent:
                st.warning(f"**[{a['tag']}] {a['severity']}** — {a['name']} "
                           f"({a['position']}, {a['owned']:.1f}% owned) — {a['message']}")
        else:
            st.success("No changes to players you own or watch.")

        rows = alerts.get("alerts", [])
        if rows:
            st.subheader(f"All availability changes ({len(rows)})")
            adf = pd.DataFrame(rows)
            adf["tag"] = adf["tag"].replace("-", "")
            st.dataframe(
                adf[["tag", "severity", "name", "position", "owned", "message"]].rename(
                    columns={"position": "pos", "owned": "own %", "message": "what changed"}),
                hide_index=True, width="stretch", height=420,
                column_config={"own %": st.column_config.NumberColumn(format="%.1f")},
            )

        moves = alerts.get("price_moves", [])
        if moves:
            st.subheader("Biggest price moves")
            mdf = pd.DataFrame(moves)
            mdf["change"] = (mdf["new"] - mdf["old"]).round(1)
            st.dataframe(
                mdf[["name", "old", "new", "change", "owned"]].rename(
                    columns={"owned": "own %"}),
                hide_index=True, width="stretch",
                column_config={
                    "old": st.column_config.NumberColumn(format="%.1f"),
                    "new": st.column_config.NumberColumn(format="%.1f"),
                    "change": st.column_config.NumberColumn(format="%+.1f"),
                    "own %": st.column_config.NumberColumn(format="%.1f"),
                },
            )


# ------------------------------------------------------------- model tab

with tab_model:
    acc = load("accuracy")
    st.subheader("Is the model actually any good?")
    st.markdown(
        "Predictions are written **before** each deadline and never edited "
        "afterwards, so these errors cannot be flattered in hindsight. "
        "Lower is better; **MAE on high-return players** (5+ points) matters "
        "more than overall MAE, because those are the players who decide ranks."
    )

    if not acc or not acc.get("scored"):
        pending = (acc or {}).get("pending", [])
        st.info(
            "Nothing scored yet — a gameweek needs both a logged prediction and "
            "a finished result."
            + (f" Waiting on GW{', GW'.join(str(g) for g in pending)}." if pending else "")
        )
    else:
        sdf = pd.DataFrame(acc["scored"])
        st.dataframe(
            sdf.rename(columns={"gameweek": "GW", "mae": "MAE",
                                "mae_high_return": "MAE (5+ pts)"}),
            hide_index=True, width="stretch",
            column_config={
                "MAE": st.column_config.NumberColumn(format="%.3f"),
                "MAE (5+ pts)": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        pivot = sdf.pivot_table(index="gameweek", columns="model", values="mae")
        st.line_chart(pivot)

    with st.expander("How the model works, and what was rejected"):
        st.markdown("""
**Two-stage prediction.** Expected points are `P(plays 60+ mins) × E[points | played 60+]`.
Predicting points directly makes the model hedge every rotation risk into every
score; splitting the question keeps the availability problem and the
performance problem apart.

**Validation is chronological, never random k-fold.** Three seasons train,
the following season tests. Rolling features are shifted with
`groupby(...).shift(1)` before any window is applied, so a row can only see
matches that finished before it.

**Leakage was tested, not assumed.** A negative control — the same pipeline
with the target shuffled — scored 1.551 MAE against 1.537 for simply predicting
the mean. A leaking pipeline would have beaten the mean comfortably.

**A rejected feature is kept in the repo.** `opponent.py` adds opponent
strength. The underlying effect is real and large: 2.73 points against the
toughest fifth of opponents versus 4.33 against the leakiest, a 1.59-point
swing. It still did not improve ranking — paired bootstrap over 38 gameweeks
gave −0.125, 95% CI [−0.287, +0.036], P(better) = 0.06. It is applied as a
human overlay on captaincy instead, and the negative result is documented
rather than deleted.
        """)


# ------------------------------------------------------------ season tab

with tab_season:
    season = load("season")
    if not season or not season.get("gameweeks"):
        st.info("No gameweeks recorded yet.")
    else:
        a, b, c = st.columns(3)
        a.metric("Total points", season.get("total_points", 0))
        b.metric("Gameweeks played", season.get("played", 0))
        c.metric("Average", f"{season['average']:.1f}" if season.get("average") else "—")

        sdf = pd.DataFrame(season["gameweeks"])
        st.dataframe(
            sdf[["gameweek", "points", "transfers", "formation",
                 "captain", "captain_points", "decided_by"]].rename(
                columns={"gameweek": "GW", "captain_points": "cap pts",
                         "decided_by": "decided by"}),
            hide_index=True, width="stretch",
        )

        played = sdf[sdf["played"]]
        if len(played) > 1:
            st.line_chart(played.set_index("gameweek")["points"])

        st.caption(
            "This is the benchmark. GW1–3 were picked on instinct with no model "
            "involved; anything the engine produces has to beat these, not just "
            "beat a naive baseline."
        )


st.divider()
st.caption(
    f"Generated {meta.get('generated_at')} · snapshot #{meta.get('snapshot_id')} · "
    f"{meta['row_counts']['history']:,} historical rows · "
    "built by GitHub Actions, rendered by Streamlit"
)
