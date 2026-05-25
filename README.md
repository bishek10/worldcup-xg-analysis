## Part 2: Predicting Who Advances (Machine Learning)

Building on the analysis, I trained a model to predict whether a team would 
advance from the group stage (top 2 finish) based only on their performance stats.

**Approach:**
- **Target:** whether a team advanced (yes/no), derived from group rank.
- **Features:** goals scored, goals against, expected goals (for & against), 
  wins, draws, losses.
- **Avoiding data leakage:** I deliberately excluded `rank`, `points`, and 
  `goal difference`, since those directly determine advancement and would let 
  the model "cheat" rather than learn from performance.
- **Model:** a Decision Tree classifier, chosen for interpretability.
- **Validation:** an 80/20 train/test split, so the model is evaluated on 
  teams it never saw during training.

**Result & honest interpretation:**
The model reached ~57% accuracy on the test set. A larger dataset spanning multiple 
tournaments would likely improve its performance.

I tried to demonstrate full ML workflow — feature engineering, leakage 
prevention, train/test validation, and honest evaluation — rather than 
overclaiming predictive power.