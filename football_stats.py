import pandas as pd
import matplotlib.pyplot as plt
df= pd.read_csv("group_stats.csv")
#
# df["g+a"]= df["goals"] +df["assists"]
# print(df.sort_values("g+a", ascending=False))
df = df.drop(columns="Unnamed: 0")
df["overperformance"] = df["goals_scored"]- df["expected_goal_scored"]
print(df[["team", "goals_scored", "expected_goal_scored", "overperformance"]].sort_values("overperformance", ascending=False))
df_sorted= df.sort_values("overperformance")

colors=["green" if x > 0 else "red" for x in df_sorted["overperformance"]]
plt.figure(figsize=(10,12))
plt.barh(df_sorted["team"], df_sorted["overperformance"], color=colors)

plt.axvline(0, color="black", linewidth=0.8)
plt.title("World Cup 2022: Who Beat Their Expected Goals?")
plt.xlabel("Goals scored minus expected goals (xG)")
plt.tight_layout()
plt.show()
df["advanced"]= (df["rank"]<=2).astype(int)
print(df[["team", "group", "rank", "points", "advanced"]].sort_values(["group", "rank"]))
# features = the stats we predict FROM (no rank/points/goal_difference — those leak the answer)
features = ["goals_scored", "goals_against", "expected_goal_scored",
            "exp_goal_conceded", "wins", "draws", "losses"]

X = df[features]      # X = the input stats (capital X is the ML convention)
y = df["advanced"]    # y = the answer we want to predict

print("Features shape:", X.shape)
print("Target shape:", y.shape)
from sklearn.model_selection import train_test_split

# split into training (model studies these) and test (model is quizzed on these)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Training teams:", len(X_train))
print("Test teams:", len(X_test))
from sklearn.tree import DecisionTreeClassifier

# create the model
model = DecisionTreeClassifier(random_state=42)

# train it: show it the training clues AND their answers so it learns the pattern
model.fit(X_train, y_train)

print("Model trained!")
from sklearn.metrics import accuracy_score

# the model guesses on the test teams — ones it NEVER saw during training
predictions = model.predict(X_test)

# compare its guesses to the real answers
accuracy = accuracy_score(y_test, predictions)

print("Predictions:", predictions)
print("Actual answers:", y_test.values)
print("Accuracy:", accuracy)