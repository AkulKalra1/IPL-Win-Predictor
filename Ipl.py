import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

matches = matches[['id', 'team1', 'team2', 'winner']]

df = deliveries.merge(matches, left_on='match_id', right_on='id')

df = df[df['inning'] == 2]

total_score = df.groupby('match_id')['total_runs'].sum().reset_index()
total_score.rename(columns={'total_runs': 'target'}, inplace=True)

df = df.merge(total_score, on='match_id')

df['runs_left'] = df['target'] - df['total_runs']
df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])
df['wickets'] = df.groupby('match_id')['is_wicket'].cumsum()
df['wickets_left'] = 10 - df['wickets']
df['crr'] = df['total_runs'] / (df['over'] + df['ball']/6)
df['rrr'] = (df['runs_left'] * 6) / df['balls_left']

df['result'] = (df['batting_team'] == df['winner']).astype(int)

final_df = df[['batting_team', 'bowling_team', 'runs_left', 
               'balls_left', 'wickets_left', 'crr', 'rrr', 'result']]

final_df = final_df.replace([np.inf, -np.inf], np.nan)
final_df = final_df.dropna()

final_df = pd.get_dummies(final_df, columns=['batting_team', 'bowling_team'])

X = final_df.drop('result', axis=1)
y = final_df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

sample = X_test.iloc[0:1]
prob = model.predict_proba(sample)

print("Win Probability:", prob[0][1] * 100, "%")
import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))