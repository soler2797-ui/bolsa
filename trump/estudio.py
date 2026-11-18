import pandas as pd, numpy as np
op=pd.read_csv('operaciones.csv',parse_dates=['fecha','publicacion'])
op['ticker']=op.ticker.str.replace('.','-',regex=False)
op=op[~op.sector.fillna('').str.startswith(('Money','ETF'))]
pr=pd.read_csv('precios.csv.gz',parse_dates=['fecha']).drop_duplicates(['fecha','ticker'])
O=pr.pivot(index='fecha',columns='ticker',values='open'); C=pr.pivot(index='fecha',columns='ticker',values='close')
dias=C.index; fin=dias[-1]
pubs=sorted(op.publicacion.unique())
H=[5,20,40,60]
def entrada(p):  # primera sesion estrictamente posterior a la publicacion: compra en apertura
    return dias[dias>p][0]
filas=[]; porfil=[]
for p in pubs:
    e=entrada(p); ie=dias.get_loc(e)
    g=op[op.publicacion==p]
    neto=g.assign(s=np.where(g.tipo=='purchase',g.mid,-g.mid)).groupby('ticker').s.sum()
    compras=g[g.tipo=='purchase'].groupby('ticker').mid.sum()
    grandes=g[(g.tipo=='purchase')&(g.low>=1_000_001)].groupby('ticker').mid.sum()
    cestas={'todas_compras':compras,'compra_neta':neto[neto>0],'compras_1M+':grandes}
    for h in H+['hasta_hoy']:
        ix=len(dias)-1 if h=='hasta_hoy' else ie+h-1
        if ix>=len(dias): continue
        s=dias[ix]
        r=C.loc[s]/O.loc[e]-1
        for b in ['SPY','RSP']:
            pass
        bench={b:r[b] for b in ['SPY','RSP']}
        for nom,w in cestas.items():
            w=w[w.index.isin(r.dropna().index)]
            if len(w)==0: continue
            req=r[w.index].mean(); rpond=(r[w.index]*w).sum()/w.sum()
            filas.append(dict(publicacion=p.date(),entrada=e.date(),salida=s.date(),plazo=h,cesta=nom,n=len(w),
              igual=req,ponderada=rpond,SPY=bench['SPY'],RSP=bench['RSP'],
              exc_SPY_igual=req-bench['SPY'],exc_RSP_igual=req-bench['RSP'],exc_SPY_pond=rpond-bench['SPY'],
              acierto_vs_SPY=(r[w.index]>bench['SPY']).mean()))
res=pd.DataFrame(filas); res.to_csv('resultados.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
fmt=res.copy()
for c in ['igual','ponderada','SPY','RSP','exc_SPY_igual','exc_RSP_igual','exc_SPY_pond','acierto_vs_SPY']: fmt[c]=(fmt[c]*100).round(1)
print(fmt.to_string(index=False))
# cartera continua: en cada publicacion compra la cesta y la mantiene hasta la siguiente publicacion
print('\nCartera continua (rota en cada publicacion, comision 0,1% por lado):')
for nom in ['todas_compras','compra_neta','compras_1M+']:
    for modo in ['igual','pond']:
        v=1.0; vs={'SPY':1.0,'RSP':1.0}
        for i,p in enumerate(pubs):
            e=entrada(p); s=entrada(pubs[i+1]) if i+1<len(pubs) else None
            g=op[op.publicacion==p]
            neto=g.assign(s=np.where(g.tipo=='purchase',g.mid,-g.mid)).groupby('ticker').s.sum()
            w={'todas_compras':g[g.tipo=='purchase'].groupby('ticker').mid.sum(),'compra_neta':neto[neto>0],
               'compras_1M+':g[(g.tipo=='purchase')&(g.low>=1_000_001)].groupby('ticker').mid.sum()}[nom]
            px_s=O.loc[s] if s is not None else C.loc[fin]
            r=px_s/O.loc[e]-1
            w=w[w.index.isin(r.dropna().index)]
            w=w*0+1 if modo=='igual' else w
            v*=(1+(r[w.index]*w).sum()/w.sum())*(1-0.001)**2
            for b in vs: vs[b]*=1+r[b]
        print(f'  {nom:14s} {modo:5s} {100*(v-1):6.1f} %   SPY {100*(vs["SPY"]-1):6.1f} %   RSP {100*(vs["RSP"]-1):6.1f} %   ({entrada(pubs[0]).date()} a {fin.date()})')
