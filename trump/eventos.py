import json, pandas as pd, re
d=json.load(open('/home/claude/lon3000/trump-tracker-pipeline/state/dataset.json'))
rows=[]
for tk,v in d['stocks'].items():
    for t in v['transactions']:
        dt=t['date']
        if not re.match(r'\d{4}-\d{2}-\d{2}$',dt):
            try: dt=pd.to_datetime(dt).strftime('%Y-%m-%d')
            except: continue
        rows.append(dict(ticker=tk,sector=v.get('sector'),fecha=dt,tipo=t['type'],low=t['low'],high=t['high'],mid=t['mid'],desc=t.get('rawDescription','')))
df=pd.DataFrame(rows)
df['fecha']=pd.to_datetime(df.fecha)
# publicacion por mes de la operacion
pub={1:'2026-05-14',2:'2026-05-14',3:'2026-05-14',4:'2026-06-30',5:'2026-06-30',6:'2026-08-24',7:'2026-09-22',8:'2026-10-08'}
df=df[(df.fecha.dt.year==2026)]
df['publicacion']=pd.to_datetime(df.fecha.dt.month.map(pub))
df=df[~df.ticker.str.startswith('_')]
print(df.groupby(['publicacion','tipo']).size().unstack())
print(df.sector.value_counts().head(15))
df.to_csv('operaciones.csv',index=False)
tk=sorted(set(df.ticker))
# yahoo usa guion en clases
tk=[t.replace('.','-') for t in tk]
open('tickers.txt','w').write('\n'.join(tk+['SPY','RSP','VOO']))
print(len(tk))
