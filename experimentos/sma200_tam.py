"""Media de 200 sesiones como indicador de tamaño en momentum (17-11-2026).
Se eligen las mismas acciones que momentum; cambia el importe de cada compra según la distancia del cierre
a su media de 200 (dist = cierre / media - 1) el día de la decisión. Las posiciones abiertas no se reajustan.
Parámetros fijados antes de ver resultados, a partir de la distribución de dist en las compras de momentum
de 2016-2021 (mediana 19 %, percentil 20: 4 %, percentil 80: 41 %).

- tam_bandas: dist < 0 -> x0,5; 0-40 % -> x1; > 40 % -> x1,5.
- tam_lineal: factor = 1 + (dist - 0,20) / 0,40, entre 0,5 y 1,5 (x0,5 en la media, x1 a +20 %, x1,5 a +40 %).
- tam_inverso (control): bandas al revés: dist < 0 -> x1,5; 0-40 % -> x1; > 40 % -> x0,5.
Tamaño normal = valor de la cartera / 10, con el tope del 20 % por posición del motor.
"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab import *
from multiprocessing import Pool


def bandas(x):
    return 0.5 if x < 0 else (1.0 if x <= 0.40 else 1.5)


def lineal(x):
    return float(min(1.5, max(0.5, 1 + (x - 0.20) / 0.40)))


def inverso(x):
    return 1.5 if x < 0 else (1.0 if x <= 0.40 else 0.5)


class Tam200(Momentum):
    def __init__(self, univ, nombre, fn):
        super().__init__(univ)
        self.nombre, self.fn = nombre, fn

    def candidatos(self, est, d, m, tickers):
        c, s = m.close_ffill.loc[d], m._ind['x_sma200'].loc[d]
        res = []
        for t, mot in super().candidatos(est, d, m, tickers):
            x = c[t] / s[t] - 1 if s.get(t, np.nan) > 0 else 0.2
            f = self.fn(x)
            res.append((t, f'{mot}, media 200 {100 * x:+.0f} %, x{f:.2f}', f))
        return res


univ = universos.cargar(refrescar_pequenas=False)
m = preparar(cargar(), univ)
m._ind['x_sma200'] = m.ind('sma200').ffill()
OFF = list(range(0, 60, 3))
f0 = m.fechas[m.fechas >= '2016-01-01']


def variantes():
    return [Momentum(univ), Tam200(univ, 'tam_bandas', bandas), Tam200(univ, 'tam_lineal', lineal),
            Tam200(univ, 'tam_inverso', inverso)]


def tarea(a):
    k, j = a
    e = variantes()[j]
    h, est = correr(e, m, f0[k])
    r = medidas(h); r.update(nombre=e.nombre, k=k)
    r['e2018'] = 100 * (h[:'2018-12-31'].iloc[-1] / h[:'2018-05-01'].iloc[-1] - 1)
    r['e2022'] = 100 * (h[:'2022-09-30'].iloc[-1] / h[:'2022-04-01'].iloc[-1] - 1)
    for per, (a_, b_) in PER.items():
        r[per + '_vol'] = 100 * h[a_:b_].pct_change().dropna().std() * np.sqrt(252)
    r['ops'] = sum(1 for o in est['operaciones'] if o['tipo'] == 'compra')
    return r, (e.nombre, k, h, est['operaciones'])


if __name__ == '__main__':
    n = len(variantes())
    with Pool(2) as p:
        out = p.map(tarea, [(k, j) for k in OFF for j in range(n)], chunksize=1)
    df = pd.DataFrame([o[0] for o in out])
    df.to_csv(os.path.join(RAIZ, 'experimentos', 'resultados_sma200_tam.csv'), index=False)
    pickle.dump({(a, b): (c, ops) for _, (a, b, c, ops) in out}, open('/tmp/sma200_tam.pkl', 'wb'))
    print(df.groupby('nombre', sort=False)[['cal_cagr', 'comp_cagr', 'cal_dd', 'comp_dd', 'e2018', 'e2022', 'ops']].mean().round(1))
