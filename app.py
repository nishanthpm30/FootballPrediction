import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, log_loss

import model as M

st.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="centered")

CURRENT = M.SEASONS[-1]
SEASON_LABEL = f"20{CURRENT[:2]}/{CURRENT[2:]}"
TEST_MATCHES = 300  # most recent matches held out to measure the model honestly


# ------------------------------
# Data + model
# ------------------------------
@st.cache_data(ttl=3600, show_spinner="Downloading latest results...")
def load_data() -> pd.DataFrame:
    return M.load_matches()


@st.cache_resource(show_spinner="Training model...")
def train(matches: pd.DataFrame):
    X, y, state = M.build_dataset(matches)
    warmup = 100  # first matches have no history to learn from

    # Honest score: train on the past, test on the most recent matches.
    X_tr, y_tr = X[warmup:-TEST_MATCHES], y[warmup:-TEST_MATCHES]
    X_te, y_te = X[-TEST_MATCHES:], y[-TEST_MATCHES:]
    check = M.make_model().fit(X_tr, y_tr)
    stats = {
        "accuracy": accuracy_score(y_te, check.predict(X_te)),
        "baseline": float((y_te == "H").mean()),  # always guessing a home win
        "logloss": log_loss(y_te, check.predict_proba(X_te), labels=check.classes_),
    }

    final = M.make_model().fit(X[warmup:], y[warmup:])
    return final, state, stats


def league_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for r in df.itertuples():
        for team, gf, ga in ((r.HomeTeam, r.FTHG, r.FTAG), (r.AwayTeam, r.FTAG, r.FTHG)):
            t = rows.setdefault(team, dict(P=0, W=0, D=0, L=0, GF=0, GA=0))
            t["P"] += 1
            t["GF"] += gf
            t["GA"] += ga
            if gf > ga:
                t["W"] += 1
            elif gf == ga:
                t["D"] += 1
            else:
                t["L"] += 1
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["GD"] = table["GF"] - table["GA"]
    table["Pts"] = table["W"] * 3 + table["D"]
    table = table.sort_values(["Pts", "GD", "GF"], ascending=False)
    table.insert(0, "Pos", range(1, len(table) + 1))
    table.index.name = "Team"
    return table.reset_index()


def recent_results(df: pd.DataFrame, team: str, n: int = 5) -> pd.DataFrame:
    games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(n).iloc[::-1]
    out = []
    for r in games.itertuples():
        home = r.HomeTeam == team
        gf, ga = (r.FTHG, r.FTAG) if home else (r.FTAG, r.FTHG)
        out.append(
            {
                "Date": r.Date.strftime("%d %b"),
                "Match": f"{r.HomeTeam} {r.FTHG}-{r.FTAG} {r.AwayTeam}",
                "Result": "W" if gf > ga else "D" if gf == ga else "L",
            }
        )
    return pd.DataFrame(out)


# ------------------------------
# UI
# ------------------------------
st.title("⚽ Premier League Predictor")
st.caption(f"Season {SEASON_LABEL} · trained on {len(M.SEASONS)} seasons · data from football-data.co.uk")

try:
    matches = load_data()
except Exception as exc:  # network or parsing failure
    st.error(f"Couldn't load match data: {exc}")
    st.stop()

season = matches[matches["Season"] == CURRENT]
if season.empty:
    st.warning("No matches have been played this season yet.")
    st.stop()

model, state, stats = train(matches)
teams = sorted(set(season["HomeTeam"]) | set(season["AwayTeam"]))

with st.sidebar:
    st.header("About")
    st.metric("Matches this season", len(season))
    st.metric("Latest match", season["Date"].max().strftime("%d %b %Y"))
    st.metric(
        "Accuracy on recent matches",
        f"{stats['accuracy'] * 100:.0f}%",
        help=f"Tested on the {TEST_MATCHES} most recent matches, which the model had not seen.",
    )
    st.caption(f"Always picking a home win scores {stats['baseline'] * 100:.0f}% on the same matches.")
    st.caption(
        "Football is hard to predict. Even a good model is right about half the time, "
        "so treat this as a rough guide, not a tip."
    )
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

tab_predict, tab_table = st.tabs(["Predict", "League table"])

with tab_predict:
    col1, col2 = st.columns(2)
    home = col1.selectbox("Home team", teams, index=0)
    away = col2.selectbox("Away team", teams, index=min(1, len(teams) - 1))

    if st.button("Predict", type="primary", width="stretch"):
        if home == away:
            st.error("Home and away teams must be different.")
        else:
            feats = state.features(home, away)
            probs = dict(zip(model.classes_, model.predict_proba(pd.DataFrame([feats]))[0]))
            p_home, p_draw, p_away = (probs.get(k, 0.0) for k in ("H", "D", "A"))

            best = max(("H", p_home), ("D", p_draw), ("A", p_away), key=lambda x: x[1])[0]
            verdict = {"H": f"{home} win", "A": f"{away} win", "D": "Draw"}[best]
            st.subheader(f"Most likely: {verdict}")

            c1, c2, c3 = st.columns(3)
            for col, label, p in ((c1, f"{home} win", p_home), (c2, "Draw", p_draw), (c3, f"{away} win", p_away)):
                col.metric(label, f"{p * 100:.0f}%")
                col.progress(float(p))

            with st.expander("What the model looked at"):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "": ["Rating (Elo)", "Points per game, last 5", "Goals scored, last 5", "Goals conceded, last 5"],
                            home: [round(state.elo[home]), round(feats["form_home"], 2), round(feats["gf_home"], 1), round(feats["ga_home"], 1)],
                            away: [round(state.elo[away]), round(feats["form_away"], 2), round(feats["gf_away"], 1), round(feats["ga_away"], 1)],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )

    if home != away:
        st.divider()
        left, right = st.columns(2)
        for col, team in ((left, home), (right, away)):
            col.markdown(f"**{team}: last 5**")
            form = recent_results(matches, team)
            if form.empty:
                col.caption("No matches yet.")
            else:
                col.dataframe(form, hide_index=True, width="stretch")

with tab_table:
    st.dataframe(league_table(season), hide_index=True, width="stretch", height=740)
