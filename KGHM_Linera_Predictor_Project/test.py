import pandas as pd
archive = pd.read_csv("https://stooq.pl/q/d/l/?s=jsw&i=d")
print(archive.info())
today = pd.read_csv("https://stooq.pl/q/l/?s=jsw&f=sd2t2ohlcv&h&e=csv")
print(today.info())