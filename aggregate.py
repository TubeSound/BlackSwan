import os
import pandas as pd


def aggregate(dirpath):
    dfs = []
    for holder in ['b', 'c', 'd', 'e']:
        df = pd.read_csv(os.path.join(dirpath, holder, 'optimize.csv'))
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.sort_values('profit', ascending=False)
    return df


if __name__ == '__main__':
    dirpath = './optimize/breakout_tick/NIKKEI'
    df = aggregate(dirpath)
    df.to_csv(os.path.join(dirpath, 'optimize.csv'), index=False)