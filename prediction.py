import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk, messagebox

# ------------------------------
# ⚙️ Machine Learning Preparation
# ------------------------------
# Load dataset
url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
data = pd.read_csv(url)
print(data.head())

# Keep only needed columns
data = data[['HomeTeam', 'AwayTeam', 'FTR']].dropna()

# Create encoders for teams
le_team = LabelEncoder()
all_teams = pd.concat([data['HomeTeam'], data['AwayTeam']])
le_team.fit(all_teams)

# Encode team names
data['HomeTeam'] = le_team.transform(data['HomeTeam'])
data['AwayTeam'] = le_team.transform(data['AwayTeam'])

# Define features and labels
X = data[['HomeTeam', 'AwayTeam']]
y = data['FTR']   # FTR = H (Home Win), D (Draw), A (Away Win)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Accuracy:", round(accuracy, 3))

# ------------------------------
# 🖥️ Tkinter GUI Setup
# ------------------------------
root = tk.Tk()
root.title("⚽ Premier League Match Predictor")
root.geometry("400x400")
root.config(bg="#222831")

# Header Label
title_label = tk.Label(
    root,
    text="Football Match Winner Predictor",
    font=("Arial", 16, "bold"),
    bg="#222831",
    fg="#FFD369",
)
title_label.pack(pady=20)

# Dropdowns for teams
teams = sorted(le_team.classes_)

tk.Label(root, text="Select Home Team:", bg="#222831", fg="white").pack()
home_team_var = tk.StringVar()
home_dropdown = ttk.Combobox(root, textvariable=home_team_var, values=teams, width=30)
home_dropdown.pack(pady=5)

tk.Label(root, text="Select Away Team:", bg="#222831", fg="white").pack()
away_team_var = tk.StringVar()
away_dropdown = ttk.Combobox(root, textvariable=away_team_var, values=teams, width=30)
away_dropdown.pack(pady=5)

# Function to predict
def predict_match():
    home = home_team_var.get()
    away = away_team_var.get()

    if home == "" or away == "":
        messagebox.showwarning("Warning", "Please select both teams!")
        return
    if home == away:
        messagebox.showerror("Error", "Home and Away teams must be different!")
        return

    try:
        home_encoded = le_team.transform([home])[0]
        away_encoded = le_team.transform([away])[0]
        pred = model.predict([[home_encoded, away_encoded]])[0]

        if pred == 'H':
            result = f"🏠 Predicted Winner: {home} (Home Win)"
        elif pred == 'A':
            result = f"🚗 Predicted Winner: {away} (Away Win)"
        else:
            result = "🤝 Predicted Result: Draw"

        messagebox.showinfo("Prediction Result", result)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Predict button
predict_button = tk.Button(
    root,
    text="Predict Winner",
    font=("Arial", 12, "bold"),
    bg="#FFD369",
    fg="#222831",
    command=predict_match
)
predict_button.pack(pady=20)

# Accuracy display
acc_label = tk.Label(
    root,
    text=f"Model Accuracy: {round(accuracy*100,2)}%",
    bg="#222831",
    fg="#00ADB5",
    font=("Arial", 10)
)
acc_label.pack(pady=10)

root.mainloop()
