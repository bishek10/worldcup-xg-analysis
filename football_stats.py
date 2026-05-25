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