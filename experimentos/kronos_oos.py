"""Kronos fuera de muestra: julio 2024 - diciembre 2025.

Los datos con los que se entrenó Kronos llegan hasta junio de 2024 y los pesos publicados en Hugging Face
(NeoQuasar/Kronos-small, 30-06-2025) no han cambiado, así que este periodo no lo ha visto. No se usa 2026.

  python experimentos/kronos_oos.py prever --parte 0 --partes 4   -> previsiones semanales de una parte de las semanas
  python experimentos/kronos_oos.py evaluar                       -> acierto, IC y estrategias con filtro de Kronos
"""
import argparse
import datetime as dt
import glob
import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

import config  # noqa: E402
from bolsa import datos, kronos_ia, universos  # noqa: E402
from bolsa.estrategias import Momentum, Puntuacion, primer_dia  # noqa: E402
from bolsa.motor import nuevo_estado, procesar_dia  # noqa: E402

DESDE, HASTA = "2024-07-01", "2025-12-31"
CARPETA = os.path.join(RAIZ, "docs", "kronos_oos")
H = config.KRONOS["horizonte"]


def mercado():
    univ = universos.cargar(refrescar_pequenas=False)
    univ["pequenas"], univ["pequenas_primera"] = [], {}
    m = datos.cargar_mercado(univ["grandes"], (dt.date.today() - dt.date(2022, 6, 1)).days / 365.25)
    return univ, m


def dias_semana(m):
    f = m.fechas[(m.fechas >= DESDE) & (m.fechas <= HASTA)]
    return [d for d in f if primer_dia(m, d, "W")]


def prever(parte, partes):
    univ, m = mercado()
    dias = dias_semana(m)
    trozo = np.array_split(np.arange(len(dias)), partes)[parte]
    filas = []
    os.makedirs(CARPETA, exist_ok=True)
    for k in trozo:
        d = dias[k]
        ts = [t for t in univ["grandes"] if t in m.close.columns and not np.isnan(m.close.at[d, t])]
        res = kronos_ia.prever(m, ts, d)
        filas += [dict(fecha=str(d.date()), ticker=t, prevision_pct=round(v, 3)) for t, v in res.items() if v is not None]
        print(d.date(), sum(v is not None for v in res.values()), "previsiones", flush=True)
        # se guarda cada semana, para no perder lo hecho si el trabajo se corta
        pd.DataFrame(filas).to_csv(os.path.join(CARPETA, f"previsiones_{parte}.csv"), index=False)


# ---------------------------------------------------------------------------
def real(m, d, t):
    s = m.close[t].loc[d:].dropna()
    if len(s) <= H or s.index[H] > pd.Timestamp(HASTA):
        return np.nan
    return 100 * (s.iloc[H] / s.iloc[0] - 1)


def con_filtro(e, nombre, umbral=None, reordenar=None):
    """Envuelve una estrategia: descarta candidatas con previsión < umbral, o reordena las `reordenar` primeras."""
    orig = e.candidatos

    def cand(est, d, m, tickers):
        c = orig(est, d, m, tickers)
        if not c:
            return c
        top = c[:40]
        prev = kronos_ia.prever(m, [x[0] for x in top], d)
        if umbral is not None:
            return [x for x in top if prev.get(x[0]) is None or prev[x[0]] >= umbral] + c[40:]
        k = reordenar
        cabeza = sorted(c[:k], key=lambda x: -(prev.get(x[0]) if prev.get(x[0]) is not None else -99))
        return cabeza + c[k:]
    e.candidatos = cand
    e.nombre = nombre
    return e


def metricas(est):
    h = pd.Series({pd.Timestamp(f): v for f, v in est["historial"]})
    ops = pd.DataFrame(est["operaciones"])
    anios = (h.index[-1] - h.index[0]).days / 365.25
    return dict(valor_final=round(h.iloc[-1]), total_pct=round(100 * (h.iloc[-1] / h.iloc[0] - 1), 1),
                anual_pct=round(100 * ((h.iloc[-1] / h.iloc[0]) ** (1 / anios) - 1), 1),
                caida_max_pct=round(100 * (h / h.cummax() - 1).min(), 1),
                compras=int((ops.tipo == "compra").sum()) if len(ops) else 0)


def evaluar():
    univ, m = mercado()
    p = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(CARPETA, "previsiones_*.csv")))])
    p["fecha"] = pd.to_datetime(p.fecha)
    p["real_pct"] = [real(m, d, t) for d, t in zip(p.fecha, p.ticker)]
    # candidatas de cada estrategia ese día
    mom = m.ind("mom12_1")
    punt = sum(m.ind(s).rank(axis=1, pct=True) for s in config.PUNTUACION["senales"]) / len(config.PUNTUACION["senales"])
    p["rank_mom"] = [mom.loc[d].rank(ascending=False).get(t, np.nan) for d, t in zip(p.fecha, p.ticker)]
    p["punt"] = [punt.at[d, t] if t in punt.columns else np.nan for d, t in zip(p.fecha, p.ticker)]
    p.to_csv(os.path.join(CARPETA, "previsiones_y_real.csv"), index=False)
    q = p.dropna(subset=["real_pct"])

    lineas = []

    def bloque(nombre, x):
        if len(x) < 30:
            return
        acierto = 100 * (np.sign(x.prevision_pct) == np.sign(x.real_pct)).mean()
        sube = 100 * (x.real_pct > 0).mean()
        ics = x.groupby("fecha").apply(lambda g: g.prevision_pct.corr(g.real_pct, method="spearman") if len(g) > 5 else np.nan).dropna()
        x = x.assign(quintil=x.groupby("fecha").prevision_pct.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1 if len(s) >= 5 else np.nan))
        porq = x.groupby("quintil").real_pct.mean().round(2).to_dict()
        lineas.append(dict(grupo=nombre, previsiones=len(x), semanas=x.fecha.nunique(),
                           prevision_mediana=round(x.prevision_pct.median(), 2), real_medio=round(x.real_pct.mean(), 2),
                           acierto_direccion=round(acierto, 1), suben_de_verdad=round(sube, 1),
                           ic_medio=round(ics.mean(), 3), ic_t=round(ics.mean() / ics.std() * np.sqrt(len(ics)), 2),
                           semanas_ic_positivo=round(100 * (ics > 0).mean(), 1),
                           **{f"q{int(k)}_real": v for k, v in porq.items()}))
    bloque("todas", q)
    bloque("top20_momentum", q[q.rank_mom <= 20])
    bloque("top20_puntuacion", q[q.groupby("fecha").punt.rank(ascending=False) <= 20])
    pd.DataFrame(lineas).to_csv(os.path.join(CARPETA, "acierto.csv"), index=False)
    print(pd.DataFrame(lineas).to_string())

    # estrategias con filtro (las previsiones se leen de los CSV; no hace falta el modelo)
    kronos_ia._error = "solo previsiones guardadas"
    fechas = m.fechas[(m.fechas >= DESDE) & (m.fechas <= HASTA)]

    def cargar_cache(col):
        kronos_ia._cache.clear()
        for f, t, v in zip(p.fecha, p.ticker, p[col]):
            kronos_ia._cache[(t, f)] = v

    def correr(variantes, etiqueta, guardar):
        res = []
        for e in variantes:
            est = nuevo_estado(e.nombre)
            for d in fechas:
                procesar_dia(est, e, d, m)
            res.append(dict(previsiones=etiqueta, estrategia=e.nombre, **metricas(est)))
            print(res[-1], flush=True)
            if guardar:
                pd.DataFrame(est["operaciones"]).to_csv(os.path.join(CARPETA, f"ops_{e.nombre}.csv"), index=False)
        return res

    def variantes(base):
        nuevo = (lambda: Momentum(univ)) if base == "momentum" else (lambda: Puntuacion(univ, tamano_variable=False))
        return [con_filtro(nuevo(), f"{base}_k0", umbral=0.0), con_filtro(nuevo(), f"{base}_k1", umbral=1.0),
                con_filtro(nuevo(), f"{base}_k2", umbral=2.0), con_filtro(nuevo(), f"{base}_kneg", umbral=-2.0),
                con_filtro(nuevo(), f"{base}_orden10", reordenar=10), con_filtro(nuevo(), f"{base}_orden20", reordenar=20)]

    filas = correr([Momentum(univ), Puntuacion(univ, tamano_variable=False)], "sin filtro", True)
    cargar_cache("prevision_pct")
    for base in ("momentum", "puntuacion_fija"):
        filas += correr(variantes(base), "kronos", True)
    # control: las mismas reglas con las previsiones de Kronos barajadas entre acciones dentro de cada semana
    for semilla in range(1, 6):
        rng = np.random.default_rng(semilla)
        p["azar"] = p.groupby("fecha").prevision_pct.transform(lambda s: rng.permutation(s.values))
        cargar_cache("azar")
        for base in ("momentum", "puntuacion_fija"):
            filas += correr(variantes(base), f"azar_{semilla}", False)
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(CARPETA, "estrategias.csv"), index=False)
    az = df[df.previsiones.str.startswith("azar")].groupby("estrategia").total_pct.agg(["min", "median", "max"])
    kr = df[df.previsiones == "kronos"].set_index("estrategia").total_pct.rename("kronos")
    control = pd.concat([kr, az.add_prefix("azar_")], axis=1).round(1)
    control.to_csv(os.path.join(CARPETA, "kronos_frente_a_azar.csv"))
    print(control.to_string())
    with open(os.path.join(CARPETA, "index.html"), "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><meta charset='utf-8'><title>Kronos fuera de muestra</title>"
                "<style>body{font-family:sans-serif;margin:2em}td,th{padding:4px 8px;border-bottom:1px solid #ddd;text-align:right}</style>"
                f"<h1>Kronos fuera de muestra ({DESDE} a {HASTA})</h1><h2>Acierto de las previsiones a {H} sesiones</h2>"
                + pd.DataFrame(lineas).to_html(index=False) + "<h2>Kronos frente a previsiones barajadas al azar (rentabilidad total, %)</h2>" + control.to_html()
                + "<h2>Estrategias con filtro de Kronos</h2>" + df[~df.previsiones.str.startswith("azar")].to_html(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["prever", "evaluar"])
    ap.add_argument("--parte", type=int, default=0)
    ap.add_argument("--partes", type=int, default=1)
    a = ap.parse_args()
    prever(a.parte, a.partes) if a.modo == "prever" else evaluar()
