import pandas as pd, numpy as np, sys
df = pd.read_csv(sys.argv[1]); base = df[df.nombre == 'momentum'].set_index('k')
orden = list(dict.fromkeys(df.nombre))
print(f"{'variante':28s} {'cal %/año':>16s} {'dif':>12s} {'comp %/año':>16s} {'dif':>12s} {'caída cal':>10s} {'caída comp':>10s} {'2018':>6s} {'2022':>6s} {'ops':>5s}")
for n in orden:
    x = df[df.nombre == n].set_index('k'); s = f"{n:28s}"
    for p in ('cal', 'comp'):
        d = x[p + '_cagr'] - base[p + '_cagr']
        s += f" {x[p+'_cagr'].mean():6.1f} ±{x[p+'_cagr'].std():4.1f}    {d.mean():+5.1f} ±{d.std():4.1f}"
    s += f" {x.cal_dd.mean():10.1f} {x.comp_dd.mean():10.1f} {x.e2018.mean():6.1f} {x.e2022.mean():6.1f} {x.ops.mean():5.0f}"
    print(s)
