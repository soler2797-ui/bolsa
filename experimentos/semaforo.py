"""Calibración de momentum_semaforo.

Compara momentum con variantes del semáforo (distintos stops para las acciones con punto amarillo)
en dos periodos que se simulan por separado, cada uno empezando con 10.000 €:
  calibración 2016-2021 y comprobación 2022-2025. No se usa ningún día de 2026.

Uso: python experimentos/semaforo.py [carpeta_resultados]
"""
import datetime as dt
import itertools
import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

from bolsa import datos, universos  # noqa: E402
from bolsa.estrategias import Momentum, MomentumSemaforo  # noqa: E402
from bolsa.motor import nuevo_estado, procesar_dia  # noqa: E402

PERIODOS = {"calibracion": ("2016-01-01", "2021-12-31"), "comprobacion": ("2022-01-01", "2025-12-31")}
STOPS = [0.05, 0.10, 0.15, 0.20]


def metricas(est):
    h = pd.Series({pd.Timestamp(f): v for f, v in est["historial"]})
    anios = (h.index[-1] - h.index[0]).days / 365.25
    caida = (h / h.cummax() - 1).min()
    ops = pd.DataFrame(est["operaciones"])
    ventas = ops[ops.tipo == "venta"] if len(ops) else ops
    anual = h.resample("YE").last()
    anual = anual / pd.concat([pd.Series([h.iloc[0]]), anual.iloc[:-1]]).values - 1
    return dict(valor_final=round(h.iloc[-1]), anual_pct=round(100 * ((h.iloc[-1] / h.iloc[0]) ** (1 / anios) - 1), 1),
                caida_max_pct=round(100 * caida, 1), compras=int((ops.tipo == "compra").sum()) if len(ops) else 0,
                stops_fase1=int((ventas.motivo == "stop fase 1").sum()) if len(ventas) else 0,
                compras_amarillas=int(ops.motivo.str.contains("amarilla").sum()) if len(ops) else 0,
                efectivo_medio_pct=round(100 * pd.Series([e / v for (_, v), e in zip(est["historial"], est["extra"].get("efectivo_hist", [1] * len(est["historial"])))]).mean(), 1) if est["extra"].get("efectivo_hist") else None,
                **{f"a{y.year}": round(100 * v, 1) for y, v in anual.items()})


def main(carpeta):
    os.makedirs(carpeta, exist_ok=True)
    univ = universos.cargar(refrescar_pequenas=False)
    univ["pequenas"], univ["pequenas_primera"] = [], {}
    anios = (dt.date.today() - dt.date(2014, 6, 1)).days / 365.25
    m = datos.cargar_mercado(univ["grandes"], anios)
    variantes = [Momentum(univ),
                 MomentumSemaforo(univ, nombre="semaforo_r20_l20", stop_amarillo_rapida=0.2, stop_amarillo_lenta=0.2),
                 MomentumSemaforo(univ, nombre="reserva_r20_l20", stop_amarillo_rapida=0.2, stop_amarillo_lenta=0.2,
                                  reservar=True),
                 MomentumSemaforo(univ, nombre="reserva_solo_rojo", stop_amarillo_rapida=0.2, stop_amarillo_lenta=0.2,
                                  reservar=True, sesiones_rapida=0)]
    filas, curvas = [], {}
    for periodo, (desde, hasta) in PERIODOS.items():
        fechas = m.fechas[(m.fechas >= desde) & (m.fechas <= hasta)]
        for e in variantes:
            est = nuevo_estado(e.nombre)
            for d in fechas:
                procesar_dia(est, e, d, m)
                est["extra"].setdefault("efectivo_hist", []).append(est["efectivo"])
            curvas[f"{periodo}|{e.nombre}"] = pd.Series({pd.Timestamp(f): v for f, v in est["historial"]})
            fila = dict(periodo=periodo, estrategia=e.nombre, **metricas(est))
            filas.append(fila)
            print(periodo, e.nombre, fila["anual_pct"], fila["caida_max_pct"], flush=True)
            pd.DataFrame(est["operaciones"]).to_csv(os.path.join(carpeta, f"ops_{periodo}_{e.nombre}.csv"), index=False)
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(carpeta, "resultados.csv"), index=False)
    pd.DataFrame(curvas).to_csv(os.path.join(carpeta, "curvas.csv"))
    t = df.pivot(index="estrategia", columns="periodo", values=["anual_pct", "caida_max_pct", "compras_amarillas", "efectivo_medio_pct"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    with open(os.path.join(carpeta, "index.html"), "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><meta charset='utf-8'><title>Calibración momentum_semaforo</title>"
                "<style>body{font-family:sans-serif;margin:2em}td,th{padding:4px 8px;border-bottom:1px solid #ddd;text-align:right}</style>"
                "<h1>Calibración de momentum_semaforo</h1><p>Calibración 2016-2021 y comprobación 2022-2025, "
                "cada periodo empieza con 10.000 €. Sin datos de 2026.</p>"
                + t.sort_values("anual_pct_calibracion", ascending=False).to_html()
                + "<h2>Detalle</h2>" + df.to_html(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, "docs", "semaforo"))
