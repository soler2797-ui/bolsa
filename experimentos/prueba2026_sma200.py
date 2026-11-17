"""Prueba en 2026 de las variantes de la media de 200 (17-11-2026). Datos hasta el último disponible (13-11-2026).
Nada se ajusta con estos datos: parámetros iguales que en sma200.py, sma200_tam.py y sma200_mitad.py.

Criterio fijado antes de ver los datos (el mismo que en la prueba del semáforo): una variante pasa si reduce la caída
máxima de 2026 frente a momentum sin perder más de unos 2 puntos de rentabilidad en 2026.
Candidatas principales (las únicas que recortaban las caídas largas en 2016-2025): sma200_salida y
sma200_entrada_salida. El resto se mide igual, solo como información.

Dos mediciones:
- tramo 2026 de las 20 ejecuciones que empiezan en 2016 (rentabilidad de 2026 y caída máxima dentro de 2026);
- ejecución nueva que empieza el 2026-01-01.
"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab
from lab import *
from multiprocessing import Pool
import sma200 as S, sma200_tam as T, sma200_mitad as M


def cargar_todo():
    d = os.path.join(RAIZ, 'experimentos', 'precios')
    crudo = {c: pd.read_parquet(os.path.join(d, f'{c}.parquet')) for c in ('open', 'high', 'low', 'close', 'volume', 'div', 'split')}
    return datos.Mercado(crudo, pd.read_parquet(os.path.join(d, 'fx.parquet')))


univ = universos.cargar(refrescar_pequenas=False)
m = preparar(cargar_todo(), univ)
m._ind['x_sma200'] = m.ind('sma200').ffill()
S.m = T.m = M.m = m           # StopVol lee la volatilidad del mercado global del módulo
FIN = str(m.fechas[-1].date())
OFF = list(range(0, 60, 3))
f0 = m.fechas[m.fechas >= '2016-01-01']
INI26 = m.fechas[m.fechas >= '2026-01-01'][0]


def variantes():
    return (S.variantes() + T.variantes()[1:] + M.variantes()[2:])


def medir26(h):
    x = h['2025-12-31':] if h.index[0] < pd.Timestamp('2026-01-01') else h
    if h.index[0] < pd.Timestamp('2026-01-01'):
        x = h[:'2025-12-31'].iloc[-1:].pipe(lambda a: pd.concat([a, h['2026-01-01':]]))
    return 100 * (x.iloc[-1] / x.iloc[0] - 1), 100 * (x / x.cummax() - 1).min()


def tarea(a):
    k, j = a
    e = variantes()[j]
    ini = INI26 if k < 0 else f0[k]
    h, est = correr(e, m, ini, FIN)
    r26, dd26 = medir26(h)
    return dict(nombre=e.nombre, k=k, r2026=r26, dd2026=dd26), (e.nombre, k, h)


if __name__ == '__main__':
    n = len(variantes())
    with Pool(2) as p:
        out = p.map(tarea, [(k, j) for k in OFF + [-1] for j in range(n)], chunksize=1)
    df = pd.DataFrame([o[0] for o in out])
    df.to_csv(os.path.join(RAIZ, 'experimentos', 'resultados_prueba2026_sma200.csv'), index=False)
    pickle.dump({(a, b): c for _, (a, b, c) in out}, open('/tmp/p26.pkl', 'wb'))
    print('fin de datos', FIN)
