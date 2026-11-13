"""Qué pasa con las acciones después de que salte el stop, y búsqueda de los mejores parámetros."""
import copy
import html
import itertools
import os
import time

import numpy as np
import pandas as pd

import config
from .motor import nuevo_estado, procesar_dia

HORIZONTES = (5, 20, 60)


def analizar_stops(estados, m):
    """Una fila por cada venta por stop: cómo evoluciona el precio en las sesiones siguientes."""
    filas = []
    for n, est in estados.items():
        for o in est["operaciones"]:
            if o["tipo"] != "venta" or not o["motivo"].startswith("stop"):
                continue
            t = o["ticker"]
            if t not in m.close.columns:
                continue
            s = m.close[t].dropna()
            f, f0 = pd.Timestamp(o["fecha"]), pd.Timestamp(o["fecha_entrada"])
            i = s.index.searchsorted(f)
            if i >= len(s):
                continue
            max_tenencia = s.loc[f0:f].max()
            despues = s.iloc[i + 1:i + 1 + max(HORIZONTES)]
            fila = dict(estrategia=n, ticker=t, fecha_entrada=o["fecha_entrada"], fecha_venta=o["fecha"],
                        fase=o["motivo"].split()[-1], precio_venta=o["precio"],
                        resultado_pct=o.get("rentabilidad_pct"))
            for h in HORIZONTES:
                fila[f"tras_{h}"] = 100 * (despues.iloc[h - 1] / o["precio"] - 1) if len(despues) >= h else np.nan
            if len(despues) >= max(HORIZONTES):
                fila["max_60"] = 100 * (despues.max() / o["precio"] - 1)
                fila["supera_maximo"] = bool(despues.max() > max_tenencia)
            else:
                fila["max_60"], fila["supera_maximo"] = np.nan, None
            filas.append(fila)
    return pd.DataFrame(filas)


def resumen_stops(df):
    if df.empty:
        return df
    d = df.dropna(subset=["tras_60"])
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(["estrategia", "fase"])
    return pd.DataFrame({
        "ventas": g.size(),
        "sube_a_20": g["tras_20"].apply(lambda x: 100 * (x > 0).mean()),
        "media_20": g["tras_20"].mean(),
        "media_60": g["tras_60"].mean(),
        "supera_max_60": g["supera_maximo"].apply(lambda x: 100 * x.astype(bool).mean()),
    }).round(1).reset_index()


def html_stops(df):
    """Sección HTML para el informe."""
    if df is None or df.empty:
        return "<h2>Ventas por stop</h2><p>Todavía no hay ventas por stop.</p>"
    r = resumen_stops(df)
    f = lambda x: "" if pd.isna(x) else f"{x:.1f}".replace(".", ",")
    filas = "".join(f"<tr><td>{x.estrategia}</td><td>{x.fase}</td><td>{x.ventas}</td><td>{f(x.sube_a_20)} %</td>"
                    f"<td>{f(x.media_20)} %</td><td>{f(x.media_60)} %</td><td>{f(x.supera_max_60)} %</td></tr>"
                    for x in r.itertuples()) if not r.empty else ""
    peores = df.dropna(subset=["max_60"]).sort_values("max_60", ascending=False).head(40)
    lista = "".join(f"<tr><td>{x.estrategia}</td><td>{x.ticker}</td><td>{x.fecha_entrada}</td><td>{x.fecha_venta}</td>"
                    f"<td>{x.fase}</td><td>{f(x.resultado_pct)} %</td><td>{f(x.tras_5)} %</td><td>{f(x.tras_20)} %</td>"
                    f"<td>{f(x.tras_60)} %</td><td>{f(x.max_60)} %</td><td>{'sí' if x.supera_maximo else 'no'}</td></tr>"
                    for x in peores.itertuples())
    recientes = df[df.tras_60.isna()].sort_values("fecha_venta", ascending=False).head(40)
    lista_rec = "".join(f"<tr><td>{x.estrategia}</td><td>{x.ticker}</td><td>{x.fecha_venta}</td><td>{x.fase}</td>"
                        f"<td>{f(x.resultado_pct)} %</td><td>{f(x.tras_5)} %</td><td>{f(x.tras_20)} %</td></tr>"
                        for x in recientes.itertuples())
    return f"""<h2>Ventas por stop: qué pasó después</h2>
<p>Variación del precio respecto al precio de venta, 5, 20 y 60 sesiones después. "Supera máximo" indica si en las
60 sesiones siguientes la acción superó el máximo que había alcanzado mientras se tenía. Si la mayoría de ventas
por stop van seguidas de subidas, el stop está demasiado ajustado.</p>
<div class="tabla"><table><tr><th>Estrategia</th><th>Fase</th><th>Ventas</th><th>Suben a 20 sesiones</th>
<th>Media a 20</th><th>Media a 60</th><th>Superan el máximo en 60</th></tr>{filas}</table></div>
<h3>Las 40 ventas por stop que más subieron después</h3>
<div class="tabla"><table><tr><th>Estrategia</th><th>Ticker</th><th>Compra</th><th>Venta</th><th>Fase</th>
<th>Resultado</th><th>+5</th><th>+20</th><th>+60</th><th>Máx. 60</th><th>Supera máximo</th></tr>{lista}</table></div>
<h3>Ventas por stop recientes (aún sin 60 sesiones de recorrido)</h3>
<div class="tabla"><table><tr><th>Estrategia</th><th>Ticker</th><th>Venta</th><th>Fase</th><th>Resultado</th>
<th>+5</th><th>+20</th></tr>{lista_rec}</table></div>"""


# ---------------------------------------------------------------------------
# Optimización con dos periodos: se elige en el primero y se comprueba en el segundo

def _anual(h, desde, hasta):
    s = h[(h.index >= desde) & (h.index <= hasta)]
    if len(s) < 20:
        return np.nan
    anios = (s.index[-1] - s.index[0]).days / 365.25
    return 100 * ((s.iloc[-1] / s.iloc[0]) ** (1 / anios) - 1)


def _metricas_periodos(est, corte):
    h = pd.Series({pd.Timestamp(f): v for f, v in est["historial"]})
    corte = pd.Timestamp(corte)
    dd = 100 * (h / h.cummax() - 1).min()
    com = sum(o["comision"] for o in est["operaciones"])
    return dict(anual_1=_anual(h, h.index[0], corte), anual_2=_anual(h, corte, h.index[-1]),
                anual_total=_anual(h, h.index[0], h.index[-1]), caida_max=dd,
                operaciones=len(est["operaciones"]), comisiones=com)


def correr(estrategia, fechas, m):
    est = nuevo_estado(estrategia.nombre)
    for d in fechas:
        procesar_dia(est, estrategia, d, m)
    return est


REJILLA_STOPS = dict(STOP_INICIAL=[0.06, 0.08, 0.10, 0.12, 0.15, 0.20], STOP_FASE2=[0.05, 0.07, 0.10, 0.15])
REJILLA_PARES = dict(z_armado=[2.0, 2.5], z_salida=[0.0, 0.5], vida_media_max=[20, 60], multiplo_tiempo=[2.0, 3.0])


def optimizar(estrategias, fechas, m, corte, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    original = (config.STOP_INICIAL, config.STOP_FASE2, copy.deepcopy(config.PARES))
    filas = []
    t0 = time.time()
    seleccion = [e for e in estrategias if e.usa_stop]
    pares = [e for e in estrategias if e.nombre.startswith("pares")]
    for si, sf in itertools.product(REJILLA_STOPS["STOP_INICIAL"], REJILLA_STOPS["STOP_FASE2"]):
        config.STOP_INICIAL, config.STOP_FASE2 = si, sf
        for e in seleccion:
            mt = _metricas_periodos(correr(e, fechas, m), corte)
            filas.append(dict(grupo="stop", estrategia=e.nombre, parametros=f"inicial {int(si*100)} %, fase 2 {int(sf*100)} %",
                              actual=(si, sf) == original[:2], **mt))
        print(f"  stops {si:.2f}/{sf:.2f} hecho ({time.time() - t0:.0f} s)")
    config.STOP_INICIAL, config.STOP_FASE2 = original[:2]
    claves = list(REJILLA_PARES)
    for valores in itertools.product(*REJILLA_PARES.values()):
        config.PARES = dict(original[2], **dict(zip(claves, valores)))
        es_actual = all(original[2][k] == v for k, v in zip(claves, valores))
        for e in pares:
            mt = _metricas_periodos(correr(e, fechas, m), corte)
            filas.append(dict(grupo="pares", estrategia=e.nombre,
                              parametros=", ".join(f"{k} {v}" for k, v in zip(claves, valores)), actual=es_actual, **mt))
        print(f"  pares {valores} hecho ({time.time() - t0:.0f} s)")
    config.PARES = original[2]
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(carpeta, "optimizacion.csv"), index=False)
    _html_optimizacion(df, fechas, corte, os.path.join(carpeta, "index.html"))
    return df


def _html_optimizacion(df, fechas, corte, ruta):
    f = lambda x: "" if pd.isna(x) else f"{x:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    bloques = []
    for nombre, g in df.groupby("estrategia", sort=False):
        g = g.sort_values("anual_1", ascending=False).reset_index(drop=True)
        g["puesto_2"] = g["anual_2"].rank(ascending=False).astype(int)
        filas = "".join(
            f"<tr style='{'background:#fff3c4' if x.actual else ''}'><td>{i + 1}</td><td>{html.escape(x.parametros)}"
            f"{' (actual)' if x.actual else ''}</td><td>{f(x.anual_1)} %</td><td>{f(x.anual_2)} %</td><td>{x.puesto_2}</td>"
            f"<td>{f(x.anual_total)} %</td><td>{f(x.caida_max)} %</td><td>{x.operaciones}</td><td>{f(x.comisiones)} €</td></tr>"
            for i, x in g.iterrows())
        corr = g["anual_1"].corr(g["anual_2"], method="spearman")
        bloques.append(f"<h2>{nombre}</h2><p>Coincidencia entre la clasificación de los dos periodos (Spearman): "
                       f"{f(corr)}. Cerca de 1, los parámetros buenos en el primer periodo también lo son en el segundo; "
                       f"cerca de 0 o negativa, la diferencia entre parámetros es casualidad.</p>"
                       f"<div class='tabla'><table><tr><th>Puesto</th><th>Parámetros</th><th>Al año, periodo 1</th>"
                       f"<th>Al año, periodo 2</th><th>Puesto en periodo 2</th><th>Al año, total</th><th>Caída máxima</th>"
                       f"<th>Operaciones</th><th>Comisiones</th></tr>{filas}</table></div>")
    doc = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Optimización</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222;max-width:1200px}}
table{{border-collapse:collapse;font-size:13px;margin-bottom:24px}}td,th{{border-bottom:1px solid #ddd;padding:4px 8px;text-align:right;white-space:nowrap}}
td:nth-child(2),th:nth-child(2){{text-align:left}}th{{background:#f4f4f4}}.tabla{{overflow-x:auto}}</style></head><body>
<h1>Optimización de parámetros</h1>
<p>Periodo 1 (para elegir): {fechas[0].date()} a {corte}. Periodo 2 (para comprobar): {corte} a {fechas[-1].date()}.
Cada tabla está ordenada por el resultado del periodo 1. Los parámetros que conviene usar son los que quedan arriba en
los dos periodos, no los mejores del total: con tantas combinaciones, alguna sale bien por casualidad.
La fila amarilla es la configuración actual.</p>{''.join(bloques)}</body></html>"""
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write(doc)
