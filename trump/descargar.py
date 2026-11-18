import yfinance as yf, pandas as pd
tk=[t.strip() for t in open('trump/tickers.txt') if t.strip()]
partes=[]
for i in range(0,len(tk),200):
    d=yf.download(tk[i:i+200],start='2026-04-01',auto_adjust=True,progress=False,threads=True,group_by='column')
    d=d[['Open','Close']].stack(level=1,future_stack=True).reset_index()
    d.columns=['fecha','ticker','open','close']
    partes.append(d.dropna())
p=pd.concat(partes)
p.to_csv('trump/precios.csv.gz',index=False)
print(p.ticker.nunique(),'tickers',p.fecha.min(),p.fecha.max())
