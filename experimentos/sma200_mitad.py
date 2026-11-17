"""Media de 200 sesiones con ventas parciales en momentum (17-11-2026).
Usa las órdenes nuevas del motor: `reducir` (venta de una fracción) y `ampliar` (compra de más acciones de una
posición abierta). Las compras nuevas son las de momentum, sin cambios. Parámetros fijados antes de ver resultados.

- mitad: si el cierre baja de la media de 200, se vende la mitad a la apertura siguiente (una vez por posición).
  El resto sigue con su stop. El efectivo liberado queda en efectivo (no abre posiciones nuevas: el máximo es de
  10 posiciones).
- mitad_recompra: igual, y si después el cierre vuelve por encima de la media se recompran las acciones vendidas.
  Puede repetirse.
- mitad_recompra_banda: como la anterior con margen para evitar idas y venidas: vende la mitad bajo la media - 3 %
  y recompra sobre la media + 3 %.
Para comparar se repiten momentum y sma200_salida (venta total bajo la media).
"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab import *
from multiprocessing import Pool
from sma200 import Sma200


class Mitad(Momentum):
    def __init__(self, univ, nombre, recompra=False, banda=0.0, fraccion=0.5):
        super().__init__(univ)
        self.nombre, self.recompra, self.banda, self.fraccion = nombre, recompra, banda, fraccion

    def al_cierre(self, est, d, m):
        ordenes = []
        pend = {o['ticker'] for o in est['pendientes']}
        c, s = m.close_ffill.loc[d], m._ind['x_sma200'].loc[d]
        for t, p in est['posiciones'].items():
            if t in pend or np.isnan(s.get(t, np.nan)) or np.isnan(c.get(t, np.nan)):
                continue
            red = p.get('reducida', 0)
            if not red and not p.get('ya_reducida') and c[t] < s[t] * (1 - self.banda):
                ordenes.append(dict(accion='reducir', ticker=t, fraccion=self.fraccion, importe=0,
                                    motivo='bajo media 200: mitad'))
                if not self.recompra:
                    p['ya_reducida'] = True
            elif red and self.recompra and c[t] > s[t] * (1 + self.banda):
                ordenes.append(dict(accion='ampliar', ticker=t, acciones=red, importe=0,
                                    motivo='sobre media 200: recompra'))
        # las órdenes de hoy no cambian el número de posiciones; el efectivo de las recompras se descuenta
        # de lo disponible para compras nuevas al apuntarlas como pendientes antes de decidir
        est['pendientes'].extend(ordenes)
        nuevas = super().al_cierre(est, d, m) or []
        for o in ordenes:
            est['pendientes'].remove(o)
        return ordenes + nuevas


univ = universos.cargar(refrescar_pequenas=False)
m = preparar(cargar(), univ)
m._ind['x_sma200'] = m.ind('sma200').ffill()
OFF = list(range(0, 60, 3))
f0 = m.fechas[m.fechas >= '2016-01-01']


def variantes():
    return [Momentum(univ), Sma200(univ, 'sma200_salida', salida=True),
            Mitad(univ, 'mitad'), Mitad(univ, 'mitad_recompra', recompra=True),
            Mitad(univ, 'mitad_recompra_banda', recompra=True, banda=0.03)]


def tarea(a):
    k, j = a
    e = variantes()[j]
    h, est = correr(e, m, f0[k])
    r = medidas(h); r.update(nombre=e.nombre, k=k)
    r['e2018'] = 100 * (h[:'2018-12-31'].iloc[-1] / h[:'2018-05-01'].iloc[-1] - 1)
    r['e2022'] = 100 * (h[:'2022-09-30'].iloc[-1] / h[:'2022-04-01'].iloc[-1] - 1)
    for per, (a_, b_) in PER.items():
        r[per + '_vol'] = 100 * h[a_:b_].pct_change().dropna().std() * np.sqrt(252)
    ops = est['operaciones']
    r['compras'] = sum(1 for o in ops if o['tipo'] == 'compra')
    r['parciales'] = sum(1 for o in ops if o['tipo'] == 'venta_parcial')
    r['ampliaciones'] = sum(1 for o in ops if o['tipo'] == 'ampliacion')
    r['comisiones'] = sum(o['comision'] for o in ops)
    return r, (e.nombre, k, h, ops)


if __name__ == '__main__':
    n = len(variantes())
    trabajos = [(k, j) for k in OFF for j in range(n)]
    prueba = len(sys.argv) > 1
    if prueba:
        trabajos = [(0, j) for j in range(n)]
    with Pool(2) as p:
        out = p.map(tarea, trabajos, chunksize=1)
    df = pd.DataFrame([o[0] for o in out])
    if not prueba:
        df.to_csv(os.path.join(RAIZ, 'experimentos', 'resultados_sma200_mitad.csv'), index=False)
    pickle.dump({(a, b): (c, ops) for _, (a, b, c, ops) in out}, open('/tmp/sma200_mitad.pkl', 'wb'))
    pd.set_option('display.width', 200)
    print(df.groupby('nombre', sort=False)[['cal_cagr', 'comp_cagr', 'cal_dd', 'comp_dd', 'e2018', 'e2022',
                                             'compras', 'parciales', 'ampliaciones', 'comisiones']].mean().round(1))
