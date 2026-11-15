"""Pruebas sobre precios con el motor del repositorio. Los datos se cortan en 2025-12-31: 2026 no se carga."""
import os, sys, random, time, pickle
import numpy as np, pandas as pd
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import config
from bolsa import datos, universos
from bolsa.estrategias import Momentum, Puntuacion, AzarGrandes
from bolsa.motor import nuevo_estado, procesar_dia, tickers_ocupados

CORTE = '2025-12-31'
def cargar():
    d = os.path.join(RAIZ, 'experimentos', 'precios')
    crudo = {c: pd.read_parquet(os.path.join(d, f'{c}.parquet'))[:CORTE] for c in ('open','high','low','close','volume','div','split')}
    fx = pd.read_parquet(os.path.join(d, 'fx.parquet'))[:CORTE]
    m = datos.Mercado(crudo, fx)
    assert m.fechas[-1] <= pd.Timestamp(CORTE)
    return m

def preparar(m, univ):
    """Indicadores nuevos, todos con datos hasta el cierre del día d."""
    T = [t for t in univ['grandes'] if t in m.close.columns]
    c = m.close_ffill[T]
    lr = np.log(c).diff()
    lr = lr.where(lr.abs() < 0.5)                    # fuera saltos de datos
    r5 = lr.rolling(5).sum()
    sect = pd.Series({t: univ['sectores'].get(t) for t in T})
    mkt = lr.mean(axis=1); mkt5 = mkt.rolling(5).sum()
    def acum(s):                                     # rentabilidad logarítmica 12-1 en el calendario común
        return s.rolling(231).sum().shift(21)
    def residual(f, f5):
        """f: factor diario (serie o DataFrame por ticker)."""
        if isinstance(f5, pd.Series):
            f5 = pd.DataFrame({t: f5 for t in T}); f = pd.DataFrame({t: f for t in T})
        cov = (r5 * f5).rolling(504, min_periods=252).mean() - r5.rolling(504, min_periods=252).mean() * f5.rolling(504, min_periods=252).mean()
        var = (f5 ** 2).rolling(504, min_periods=252).mean() - f5.rolling(504, min_periods=252).mean() ** 2
        beta = (cov / var).clip(0, 3)
        res = acum(lr) - beta * acum(f)
        sd = (r5 - beta * f5).rolling(252, min_periods=126).std()
        return res, res / sd
    m._ind['x_resid_m_bruto'], m._ind['x_resid_m'] = residual(mkt, mkt5)
    fs = pd.DataFrame({t: lr[sect.index[(sect == sect[t]) & (sect.index != t)]].mean(axis=1) for t in T})  # sector sin la propia acción
    m._ind['x_resid_s_bruto'], m._ind['x_resid_s'] = residual(fs, fs.rolling(5).sum())
    m._ind['x_mom_vol'] = acum(lr) / r5.rolling(252, min_periods=126).std()
    # amplitud: proporción de acciones sobre su media de 200 sesiones
    s200 = m.ind('sma200')[T].ffill()
    m._ind['x_amplitud'] = ((c > s200).sum(axis=1) / s200.notna().sum(axis=1))
    # rentabilidad de cada sector (media simple) a 63 y 126 sesiones
    sl = pd.DataFrame({s: lr[sect.index[sect == s]].mean(axis=1) for s in sorted(sect.unique())})
    m._ind['x_sector_63'] = sl.rolling(63).sum(); m._ind['x_sector_126'] = sl.rolling(126).sum()
    m._ind['x_max252c'] = c.rolling(252, min_periods=200).max()
    m.sect = sect
    return m

class PorIndicador(Momentum):
    """Momentum ordenado por otro indicador (mayor es mejor, se exige > 0)."""
    def __init__(self, univ, nombre, ind):
        super().__init__(univ); self.nombre, self.indn = nombre, ind
    def candidatos(self, est, d, m, tickers):
        v = m._ind[self.indn].loc[d]
        res = sorted(((v[t], t) for t in tickers if v.get(t, np.nan) > 0), reverse=True)
        return [(t, self.indn) for _, t in res]

class Amplitud(Momentum):
    """No compra cuando menos de `umbral` de las acciones está sobre su media de 200. desplaz: control (filtro desplazado en el tiempo)."""
    def __init__(self, univ, nombre, umbral=0.5, desplaz=0):
        super().__init__(univ); self.nombre, self.umbral, self.desplaz = nombre, umbral, desplaz
    def candidatos(self, est, d, m, tickers):
        a = m._ind['x_amplitud']
        if self.desplaz:
            i = (a.index.get_loc(d) + self.desplaz) % len(a)
            i = max(i, 260); val = a.iloc[i]
        else: val = a.loc[d]
        if val < self.umbral: return []
        return super().candidatos(est, d, m, tickers)

class Sector(Momentum):
    """Solo acciones de los `n` sectores con mejor rentabilidad en `ses` sesiones; dentro, por momentum 12-1. azar: control."""
    def __init__(self, univ, nombre, ses=126, n=1, azar=None):
        super().__init__(univ); self.nombre, self.ses, self.n, self.azar = nombre, ses, n, azar
    def candidatos(self, est, d, m, tickers):
        s = m._ind[f'x_sector_{self.ses}'].loc[d].dropna()
        if self.azar is not None:
            rng = random.Random(f'{self.azar}-{d.year}-{d.month}')
            mejores = rng.sample(sorted(s.index), self.n)
        else:
            mejores = list(s.sort_values(ascending=False).index[:self.n])
        return super().candidatos(est, d, m, [t for t in tickers if m.sect.get(t) in mejores])

class Interseccion(Momentum):
    """Top 30 de momentum y a menos del 5 % de su máximo de 52 semanas."""
    def __init__(self, univ, nombre, top=30, dist=0.05):
        super().__init__(univ); self.nombre, self.top, self.dist = nombre, top, dist
    def candidatos(self, est, d, m, tickers):
        mom = m.ind('mom12_1').loc[d].dropna().sort_values(ascending=False)
        top = set(mom.index[:self.top])
        c, mx = m.close_ffill.loc[d], m._ind['x_max252c'].loc[d]
        return [(t, mo) for t, mo in super().candidatos(est, d, m, tickers) if t in top and c[t] >= (1 - self.dist) * mx.get(t, np.inf)]

def correr(e, m, inicio, fin=CORTE):
    est = nuevo_estado(e.nombre)
    for d in m.fechas[(m.fechas >= inicio) & (m.fechas <= pd.Timestamp(fin))]:
        procesar_dia(est, e, d, m)
    h = pd.Series({pd.Timestamp(f): v for f, v in est['historial']})
    return h, est

PER = {'cal': ('2016-04-01', '2021-12-31'), 'comp': ('2022-01-01', '2025-12-31')}
def medidas(h):
    out = {}
    for k, (a, b) in PER.items():
        x = h[a:b]; r = x.pct_change().dropna()
        out[k + '_cagr'] = 100 * ((x.iloc[-1] / x.iloc[0]) ** (252 / len(r)) - 1)
        out[k + '_dd'] = 100 * (x / x.cummax() - 1).min()
    return out
