import pandas as pd, numpy as np
exec(open('estudio.py').read().split("filas=[]")[0])
out={'compra_neta':1.0,'venta_neta':1.0,'RSP':1.0}
for i,p in enumerate(pubs):
    e=entrada(p); s=entrada(pubs[i+1]) if i+1<len(pubs) else None
    px=O.loc[s] if s is not None else C.loc[fin]; r=(px/O.loc[e]-1)
    g=op[op.publicacion==p]; neto=g.assign(s=np.where(g.tipo=='purchase',g.mid,-g.mid)).groupby('ticker').s.sum()
    a=r.reindex(neto[neto>0].index).dropna(); b=r.reindex(neto[neto<0].index).dropna()
    out['compra_neta']*=1+a.mean(); out['venta_neta']*=1+b.mean(); out['RSP']*=1+r['RSP']
    print(p.date(), f'compra neta {100*a.mean():5.1f} (n={len(a)})  venta neta {100*b.mean():5.1f} (n={len(b)})  RSP {100*r["RSP"]:5.1f}  SPY {100*r["SPY"]:5.1f}')
print({k:round(100*(v-1),1) for k,v in out.items()})
