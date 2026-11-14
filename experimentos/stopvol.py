"""Momentum con stop proporcional a la volatilidad de cada acción.

Dos periodos simulados por separado, cada uno con 10.000 €: calibración 2016-2021 y comprobación 2022-2025.
No se usa ningún día de 2026.

Uso: python experimentos/stopvol.py [carpeta_resultados]
"""
import datetime as dt
import os
import sys

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

from bolsa import datos, universos  # noqa: E402
from bolsa.estrategias import Momentum, MomentumStopVol  # noqa: E402
from bolsa.motor import nuevo_estado, procesar_dia  # noqa: E402

PERIODOS = {"calibracion": ("2016-01-01", "2021-12-31"), "comprobacion": ("2022-01-01", "2025-12-31")}

# nombre: (k1, k2, ventana)
VARIANTES = {
    "v252_k1.5": (1.5, None, 252),
    "v252_k2": (2.0, None, 252),
    "v252_k2.5": (2.5, None, 252),
    "v252_k3": (3.0, None, 252),
    "v252_k2_f0.75": (2.0, 0.75, 252),
    "v252_k2_f1": (2.0, 1.0, 252),
    "v252_k2.5_f0.75": (2.5, 0.75, 252),
    "v252_k2.5_f1": (2.5, 1.0, 252),
    "v60_k2": (2.0, None, 60),
    "v60_k2.5": (2.5, None, 60),
    "v60_k2_f1": (2.0, 1.0, 60),
}


def metricas(est, sectores):
    h = pd.Series({pd.Timestamp(f): v for f, v in est["historial"]})
    anios = (h.index[-1] - h.index[0]).days / 365.25
    ops = pd.DataFrame(est["operaciones"])
    v = ops[ops.tipo == "venta"]
    anual = h.resample("YE").last()
    anual = anual / pd.concat([pd.Series([h.iloc[0]]), anual.iloc[:-1]]).values - 1
    por_sector = v.assign(sector=v.ticker.map(sectores)).groupby("sector").resultado.sum()
    fila = dict(valor_final=round(h.iloc[-1]), anual_pct=round(100 * ((h.iloc[-1] / h.iloc[0]) ** (1 / anios) - 1), 1),
                caida_max_pct=round(100 * (h / h.cummax() - 1).min(), 1), compras=int((ops.tipo == "compra").sum()),
                stops_fase1=int((v.motivo == "stop fase 1").sum()),
                stop_mediano=(ops[ops.tipo == "compra"].motivo.str.extract(r"stop (\d+) %")[0].astype(float).median()
                              if ops[ops.tipo == "compra"].motivo.str.contains("stop ").any() else 20.0),
                perdida_media_f1=round(v[v.motivo == "stop fase 1"].rentabilidad_pct.mean(), 1),
                **{f"a{y.year}": round(100 * x, 1) for y, x in anual.items()})
    return fila, por_sector


def main(carpeta):
    os.makedirs(carpeta, exist_ok=True)
    univ = universos.cargar(refrescar_pequenas=False)
    univ["pequenas"], univ["pequenas_primera"] = [], {}
    m = datos.cargar_mercado(univ["grandes"], (dt.date.today() - dt.date(2014, 6, 1)).days / 365.25)
    estrategias = [Momentum(univ)] + [MomentumStopVol(univ, nombre=n, k1=a, k2=b, ventana=c)
                                      for n, (a, b, c) in VARIANTES.items()]
    filas, sectores = [], []
    for periodo, (desde, hasta) in PERIODOS.items():
        fechas = m.fechas[(m.fechas >= desde) & (m.fechas <= hasta)]
        for e in estrategias:
            est = nuevo_estado(e.nombre)
            for d in fechas:
                procesar_dia(est, e, d, m)
            fila, ps = metricas(est, univ["sectores"])
            filas.append(dict(periodo=periodo, estrategia=e.nombre, **fila))
            sectores.append(ps.rename(f"{periodo}|{e.nombre}"))
            print(periodo, e.nombre, fila["anual_pct"], fila["caida_max_pct"], flush=True)
            pd.DataFrame(est["operaciones"]).to_csv(os.path.join(carpeta, f"ops_{periodo}_{e.nombre}.csv"), index=False)
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(carpeta, "resultados.csv"), index=False)
    ps = pd.concat(sectores, axis=1).fillna(0).round(0)
    ps.to_csv(os.path.join(carpeta, "resultado_por_sector.csv"))
    t = df.pivot(index="estrategia", columns="periodo", values=["anual_pct", "caida_max_pct"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    with open(os.path.join(carpeta, "index.html"), "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><meta charset='utf-8'><title>Momentum con stop por volatilidad</title>"
                "<style>body{font-family:sans-serif;margin:2em}td,th{padding:4px 8px;border-bottom:1px solid #ddd;text-align:right}</style>"
                "<h1>Momentum con stop por volatilidad</h1><p>Calibración 2016-2021 y comprobación 2022-2025, "
                "cada periodo con 10.000 €. Sin datos de 2026.</p>"
                + t.sort_values("anual_pct_calibracion", ascending=False).to_html()
                + "<h2>Detalle</h2>" + df.to_html(index=False)
                + "<h2>Resultado por sector (€)</h2>" + ps.to_html())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, "docs", "stopvol"))
