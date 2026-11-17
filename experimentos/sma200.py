"""Media de 200 sesiones (regla de Paul Tudor Jones) en momentum, comparada con las otras formas probadas
de reducir el riesgo de momentum. Mismo método que run1: 20 fechas de inicio, 2016-21 calibración,
2022-25 comprobación, datos cortados en 2025-12-31. Una sola parametrización por idea, fijada antes de ver resultados.

Variantes de la media de 200:
- sma200_entrada: solo se compra si el cierre está por encima de la media de 200 de la acción.
- sma200_salida: además del stop, se vende (a la apertura siguiente) si el cierre baja de la media de 200.
- sma200_entrada_salida: las dos anteriores juntas.
- sma200_mercado: si el MSCI World (IWDA.AS) cierra por debajo de su media de 200 no se hacen compras nuevas;
  las posiciones abiertas siguen con su stop.

Comparación (código de sus ramas, parámetros de los documentos del proyecto):
- momentum_semaforo y momentum_reserva (rama principal).
- sector_bloqueo: 2 stops de fase 1 de un sector en 40 sesiones -> 40 sesiones con máximo 1 acción (rama sectores).
- stopvol_k25: stop de fase 1 = 2,5 x volatilidad mensual de 252 sesiones (10-40 %), fase 2 fija (rama stopvol).
"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab import *
from multiprocessing import Pool
from bolsa.estrategias import MomentumSemaforo
from bolsa.motor import tickers_ocupados

MERCADO = 'IWDA.AS'


class Sma200(Momentum):
    def __init__(self, univ, nombre, entrada=False, salida=False, mercado=False):
        super().__init__(univ)
        self.nombre, self.entrada, self.salida, self.mercado = nombre, entrada, salida, mercado

    def _bajo(self, d, m, t):
        c, s = m.close_ffill.at[d, t], m._ind['x_sma200'].at[d, t]
        return not np.isnan(c) and not np.isnan(s) and c < s

    def candidatos(self, est, d, m, tickers):
        if self.mercado and self._bajo(d, m, MERCADO):
            return []
        res = super().candidatos(est, d, m, tickers)
        if self.entrada:
            res = [r for r in res if not self._bajo(d, m, r[0])]
        return res

    def al_cierre(self, est, d, m):
        ventas = []
        if self.salida:
            pend = {o['ticker'] for o in est['pendientes']}
            for t in est['posiciones']:
                if t not in pend and self._bajo(d, m, t):
                    ventas.append(dict(accion='vender', ticker=t, importe=0, motivo='bajo media 200'))
        est['pendientes'].extend(ventas)        # cuentan como ocupados: no se recompra el mismo día
        nuevas = super().al_cierre(est, d, m) or []
        for v in ventas:
            est['pendientes'].remove(v)
        return ventas + nuevas


class SectorBloqueo(Momentum):
    """Copia de MomentumSector (rama sectores) con la mejor variante de calibración."""
    def __init__(self, univ, nombre='sector_bloqueo', stops=2, ventana=40, duracion=40, tope_bloqueo=1):
        super().__init__(univ)
        self.nombre, self.stops, self.ventana, self.duracion, self.tope_bloqueo = nombre, stops, ventana, duracion, tope_bloqueo

    def _sector(self, t):
        return self.universos['sectores'].get(t, '?')

    def _bloqueos(self, est, d, m):
        ex = est['extra']
        hist = ex.setdefault('stops_sector', [])
        hasta = ex.setdefault('bloqueo_hasta', {})
        i = int(m.fechas.get_loc(d))
        ops = est['operaciones']
        for o in ops[ex.get('ops_vistas', 0):]:
            if o['tipo'] == 'venta' and o['motivo'] == 'stop fase 1':
                sec = self._sector(o['ticker'])
                j = int(m.fechas.get_loc(pd.Timestamp(o['fecha'])))
                hist.append([j, sec])
                if sum(1 for k, s in hist if s == sec and j - k < self.ventana) >= self.stops:
                    hasta[sec] = max(hasta.get(sec, -1), j + self.duracion)
        ex['ops_vistas'] = len(ops)
        ex['stops_sector'] = [h for h in hist if i - h[0] < self.ventana]
        return {s for s, k in hasta.items() if k >= i}

    def candidatos(self, est, d, m, tickers):
        bloq = self._bloqueos(est, d, m)
        cuenta = {}
        for t in tickers_ocupados(est):
            cuenta[self._sector(t)] = cuenta.get(self._sector(t), 0) + 1
        res = []
        for t, mot in super().candidatos(est, d, m, tickers):
            sec = self._sector(t)
            if sec in bloq and cuenta.get(sec, 0) >= self.tope_bloqueo:
                continue
            cuenta[sec] = cuenta.get(sec, 0) + 1
            res.append((t, mot))
        return res

    def al_cierre(self, est, d, m):
        self._bloqueos(est, d, m)
        return super().al_cierre(est, d, m)


class StopVol(Momentum):
    """Copia de MomentumStopVol (rama stopvol): k1 2,5, volatilidad de 252 sesiones, fase 2 fija al 10 %."""
    def __init__(self, univ, nombre='stopvol_k25', k1=2.5, mn=0.10, mx=0.40):
        super().__init__(univ)
        self.nombre, self.k1, self.mn, self.mx = nombre, k1, mn, mx

    def al_cierre(self, est, d, m):
        self._d = d
        return super().al_cierre(est, d, m)

    def al_comprar(self, est, orden):
        v = m.ind('volat252').at[self._d, orden['ticker']]
        if not np.isnan(v):
            orden['stop_inicial'] = float(min(self.mx, max(self.mn, self.k1 * v * np.sqrt(21))))
        return orden


univ = universos.cargar(refrescar_pequenas=False)
m = preparar(cargar(), univ)
m._ind['x_sma200'] = m.ind('sma200').ffill()
OFF = list(range(0, 60, 3))
f0 = m.fechas[m.fechas >= '2016-01-01']


def variantes():
    return [Momentum(univ),
            Sma200(univ, 'sma200_entrada', entrada=True),
            Sma200(univ, 'sma200_salida', salida=True),
            Sma200(univ, 'sma200_entrada_salida', entrada=True, salida=True),
            Sma200(univ, 'sma200_mercado', mercado=True),
            MomentumSemaforo(univ),
            MomentumSemaforo(univ, nombre='momentum_reserva', reservar=True),
            SectorBloqueo(univ),
            StopVol(univ)]


def tarea(a):
    k, j = a
    e = variantes()[j]
    h, est = correr(e, m, f0[k])
    r = medidas(h); r.update(nombre=e.nombre, k=k)
    r['e2018'] = 100 * (h[:'2018-12-31'].iloc[-1] / h[:'2018-05-01'].iloc[-1] - 1)
    r['e2022'] = 100 * (h[:'2022-09-30'].iloc[-1] / h[:'2022-04-01'].iloc[-1] - 1)
    for per, (a_, b_) in PER.items():
        x = h[a_:b_].pct_change().dropna()
        r[per + '_vol'] = 100 * x.std() * np.sqrt(252)
    r['ops'] = sum(1 for o in est['operaciones'] if o['tipo'] == 'compra')
    r['salidas_media'] = sum(1 for o in est['operaciones'] if o.get('motivo') == 'bajo media 200')
    return r, (e.nombre, k, h, est['operaciones'])


if __name__ == '__main__':
    n = len(variantes())
    trabajos = [(k, j) for k in OFF for j in range(n)]
    if len(sys.argv) > 1:                       # prueba rápida: una fecha de inicio
        trabajos = [(0, j) for j in range(n)]
    with Pool(2) as p:
        out = p.map(tarea, trabajos, chunksize=1)
    df = pd.DataFrame([o[0] for o in out])
    df.to_csv(os.path.join(RAIZ, 'experimentos', 'resultados_sma200.csv' if len(sys.argv) == 1 else 'sma200_prueba.csv'), index=False)
    pickle.dump({(a, b): (c, ops) for _, (a, b, c, ops) in out}, open('/tmp/sma200_curvas.pkl', 'wb'))
    print(df.groupby('nombre', sort=False)[['cal_cagr', 'comp_cagr', 'cal_dd', 'comp_dd', 'e2018', 'e2022', 'ops', 'salidas_media']].mean().round(1))
