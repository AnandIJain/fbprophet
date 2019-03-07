import helpers as h
import pandas as pd
import numpy as np
from fbprophet import Prophet

def get_games(fn):
    # takes in fn and returns python dict of pd dfs 
    raw = csv(fn)
    # df = raw.astype(np.float32)
    games = chunk(raw, 'game_id')
    return games


def chunk(df, col):
    # returns a python dict of pandas dfs, splitting the df arg by unique col value
    # df type pd df, col type string
    games = {key: val for key, val in df.groupby(col)}
    return games


def csv(fn):
    # takes in file name, returns pandas dataframe
    # fn is type string
    df = pd.read_csv(fn, dtype='unicode')
    df.dropna()
    return df.copy()


def prophet(game, fn):
	m = Prophet()
	m.fit(game)
	future = m.make_future_dataframe(periods=150)
	forecast = m.predict(future)
	forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
	csv = forecast.to_csv(fn, index=None, header=True)


def main():
	games = get_games('nba2.csv')
	keys = list(games.keys())
	for key in keys:
		print(key)
		prophet(games[key], str(key) + '.csv')

main()
