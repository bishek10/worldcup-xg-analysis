# World Cup 2022: Who Beat Their Expected Goals?

A data analysis project exploring which World Cup 2022 teams overperformed or 
underperformed their expected goals (xG) — finding the stories the standings don't tell.

## The Question
Expected goals (xG) measures the quality of chances a team creates. By comparing 
a team's *actual* goals to their *expected* goals, we can see who finished clinically 
and who wasted strong chances — insight a casual glance at the results table misses.

## The Data
World Cup 2022 group-stage team stats (32 teams), sourced from Kaggle, including 
goals scored and expected goals per team.

## The Method
Using Python and pandas, I calculated an `overperformance` metric 
(goals scored − expected goals) for each team, then visualized the results 
with matplotlib as a color-coded horizontal bar chart.

## The Finding
- **Clinical finishers:** England and Spain led the field, each scoring 3.8 goals 
  above expectation.
- **The hidden story — Germany:** Germany created the *most* chances of any team 
  (10.1 xG) but scored only 6, underperforming by 4.1 — the worst in the tournament. 
  They generated enough quality to advance, but their finishing let them down, and 
  they exited in the group stage.
- **xG isn't everything:** Argentina underperformed their xG (−1.0) and still won 
  the tournament — a reminder that finishing quality is one piece of a bigger picture.

## How to Run
1. Install requirements: `pip install pandas matplotlib`
2. Run: `python football_stats.py`
3. The script prints the ranked table and displays the bar chart.

## Tools
Python, pandas, matplotlib