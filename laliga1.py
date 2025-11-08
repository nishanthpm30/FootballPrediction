import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("⚽ LaLiga Match Winner Predictor")

# Load data
url = "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"
data = pd.read_csv(url)[['HomeTeam', 'AwayTeam', 'FTR']].dropna()

# Encode teams
le_team = LabelEncoder()
all_teams = pd.concat([data['HomeTeam'], data['AwayTeam']])
le_team.fit(all_teams)
data['HomeTeam'] = le_team.transform(data['HomeTeam'])
data['AwayTeam'] = le_team.transform(data['AwayTeam'])

# Prepare features and labels
X = data[['HomeTeam', 'AwayTeam']]
y = data['FTR']

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Accuracy
accuracy = accuracy_score(y, model.predict(X))
st.sidebar.info(f"Model Accuracy: {round(accuracy*100, 2)}%")

# User input
teams = sorted(le_team.classes_)
home = st.selectbox("Select Home Team", teams)
away = st.selectbox("Select Away Team", teams)

if st.button("Predict Winner"):
    if home == away:
        st.error("Home and Away teams must be different!")
    else:
        pred = model.predict([[le_team.transform([home])[0], le_team.transform([away])[0]]])[0]
        if pred == 'H':
            st.success(f"🏠 Predicted Winner: {home} (Home Win)")
        elif pred == 'A':
            st.success(f"🚗 Predicted Winner: {away} (Away Win)")
        else:
            st.info("🤝 Predicted Result: Draw")
