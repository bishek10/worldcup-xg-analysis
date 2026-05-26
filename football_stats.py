import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ── Load and prepare data ──────────────────────────────────────────
df = pd.read_csv("group_stats.csv")
df = df.drop(columns="Unnamed: 0")

# ── Analysis: goals scored vs expected goals (xG) ──────────────────
df["overperformance"] = df["goals_scored"] - df["expected_goal_scored"]
print(df[["team", "goals_scored", "expected_goal_scored", "overperformance"]]
      .sort_values("overperformance", ascending=False))

# Chart: who over/underperformed their xG
df_sorted = df.sort_values("overperformance")
colors = ["green" if x > 0 else "red" for x in df_sorted["overperformance"]]
plt.figure(figsize=(10, 12))
plt.barh(df_sorted["team"], df_sorted["overperformance"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("World Cup 2022: Who Beat Their Expected Goals?")
plt.xlabel("Goals scored minus expected goals (xG)")
plt.tight_layout()
plt.savefig("xg_chart.png", dpi=150)   # save the chart instead of popping a window
plt.close()

# ── Machine learning: predict who advances from the group stage ────
# Target: did the team finish top-2 in its group?
df["advanced"] = (df["rank"] <= 2).astype(int)

# Features = performance stats only.
# Exclude rank/points/goal_difference — they directly define advancement (leakage).
features = ["goals_scored", "goals_against", "expected_goal_scored",
            "exp_goal_conceded", "wins", "draws", "losses"]
X = df[features]
y = df["advanced"]

# Train/test split: evaluate on teams the model never saw
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("\nModel accuracy:", accuracy)

# Which stats did the model rely on most?
importances = pd.Series(model.feature_importances_, index=features)
print("\nFeature importance:")
print(importances.sort_values(ascending=False))

# ── Experiment: predict using ONLY performance quality (xG) ────────
# Removes "wins" (closest to the outcome) to test pure performance signal.
xg_features = ["expected_goal_scored", "exp_goal_conceded"]
X2 = df[xg_features]
y2 = df["advanced"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42)

model2 = DecisionTreeClassifier(max_depth=3, random_state=42)
model2.fit(X2_train, y2_train)

accuracy2 = accuracy_score(y2_test, model2.predict(X2_test))
print("\nxG-only accuracy:", accuracy2)