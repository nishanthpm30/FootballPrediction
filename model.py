"""Data loading, feature building and model training for the predictor.

Every feature for a match is computed only from matches played BEFORE it,
so training never leaks the result it is trying to predict.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]  # last = current season
URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"
WINDOW = 5
ELO_START, ELO_K, ELO_HOME = 1500.0, 20.0, 60.0
FEATURES = [
    "elo_diff", "form_home", "form_away", "gf_home", "gf_away",
    "ga_home", "ga_away", "venue_home", "venue_away",
]


def load_matches(seasons=SEASONS) -> pd.DataFrame:
    frames = []
    for s in seasons:
        try:
            df = pd.read_csv(URL.format(s))
        except Exception:
            continue  # season file missing/unreachable: skip it
        df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].dropna()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
        df["Season"] = s
        frames.append(df)
    if not frames:
        raise RuntimeError("Could not download any season data.")
    out = pd.concat(frames, ignore_index=True)
    out[["FTHG", "FTAG"]] = out[["FTHG", "FTAG"]].astype(int)
    return out.sort_values("Date", kind="stable").reset_index(drop=True)


class TeamState:
    """Running per-team stats, updated one match at a time."""

    def __init__(self):
        self.elo = defaultdict(lambda: ELO_START)
        self.pts = defaultdict(lambda: deque(maxlen=WINDOW))
        self.gf = defaultdict(lambda: deque(maxlen=WINDOW))
        self.ga = defaultdict(lambda: deque(maxlen=WINDOW))
        self.home_pts = defaultdict(lambda: deque(maxlen=WINDOW))
        self.away_pts = defaultdict(lambda: deque(maxlen=WINDOW))

    @staticmethod
    def _avg(d, default):
        return float(np.mean(d)) if d else default

    def features(self, home: str, away: str) -> dict:
        return {
            "elo_diff": self.elo[home] + ELO_HOME - self.elo[away],
            "form_home": self._avg(self.pts[home], 1.4),
            "form_away": self._avg(self.pts[away], 1.4),
            "gf_home": self._avg(self.gf[home], 1.4),
            "gf_away": self._avg(self.gf[away], 1.4),
            "ga_home": self._avg(self.ga[home], 1.4),
            "ga_away": self._avg(self.ga[away], 1.4),
            "venue_home": self._avg(self.home_pts[home], 1.6),
            "venue_away": self._avg(self.away_pts[away], 1.1),
        }

    def update(self, home, away, hg, ag):
        ph, pa = (3, 0) if hg > ag else (1, 1) if hg == ag else (0, 3)
        exp_home = 1 / (1 + 10 ** (-(self.elo[home] + ELO_HOME - self.elo[away]) / 400))
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        margin = 1 + 0.5 * np.log1p(abs(hg - ag))
        delta = ELO_K * margin * (score - exp_home)
        self.elo[home] += delta
        self.elo[away] -= delta
        self.pts[home].append(ph); self.pts[away].append(pa)
        self.gf[home].append(hg); self.ga[home].append(ag)
        self.gf[away].append(ag); self.ga[away].append(hg)
        self.home_pts[home].append(ph); self.away_pts[away].append(pa)


def build_dataset(matches: pd.DataFrame):
    """Return (feature table, TeamState after the final match)."""
    state = TeamState()
    rows = []
    for r in matches.itertuples():
        rows.append(state.features(r.HomeTeam, r.AwayTeam))
        state.update(r.HomeTeam, r.AwayTeam, r.FTHG, r.FTAG)
    X = pd.DataFrame(rows, columns=FEATURES)
    return X, matches["FTR"].reset_index(drop=True), state


def make_model(kind: str = "logreg"):
    if kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42
        )
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
