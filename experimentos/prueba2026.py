"""Prueba en 2026 (periodo reservado) de la reserva del hueco, fijada antes de ver los datos.

Candidata: semáforo con reserva (amarillas con stop normal 20/10, punto borrado tras 10 sesiones fuera del top 20).
Referencias: momentum y semáforo sin reserva.
Criterio fijado de antemano: la reserva funciona si reduce la caída máxima de 2026 sin perder más de unos
2 puntos de rentabilidad frente a momentum.
"""
import datetime as dt
import os
import sys

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

from bolsa import datos, universos  # noqa: E402
from bolsa.estrategias import Momentum, MomentumSemaforo  # noqa: E402
from bolsa.motor import nuevo_estado, procesar_dia  # noqa: E402

CARPETA = os.path.join(RAIZ, "docs", "prueba2026")


def resumen(h, desde):
    h = h[h.index >= pd.Timestamp(desde) - pd.Timedelta(days=7)]
    h = h / h.iloc[0]
    return dict(rentabilidad_pct=round(100 * (h.iloc[-1] - 1), 1), caida_max_pct=round(100 * (h / h.cummax() - 1).min(), 1))


def main():
    os.makedirs(CARPETA, exist_ok=True)
    univ = universos.cargar(refrescar_pequenas=False)
    univ["pequenas"], univ["pequenas_primera"] = [], {}
    m = datos.cargar_mercado(univ["grandes"], (dt.date.today() - dt.date(2020, 6, 1)).days / 365.25)
    con = [t for t in univ["grandes"] if t in m.close.columns and m.close[t].notna().sum() > 0]
    filas, curvas = [], {}
    for inicio in ("2026-01-01", "2022-01-01"):
        for e in [Momentum(univ),
                  MomentumSemaforo(univ, nombre="semaforo_r20_l20", stop_amarillo_rapida=0.2, stop_amarillo_lenta=0.2),
                  MomentumSemaforo(univ, nombre="reserva_r20_l20", stop_amarillo_rapida=0.2, stop_amarillo_lenta=0.2,
                                   reservar=True)]:
            est = nuevo_estado(e.nombre)
            ef = []
            for d in m.fechas[m.fechas >= inicio]:
                procesar_dia(est, e, d, m)
                ef.append(est["efectivo"] / est["historial"][-1][1])
            h = pd.Series({pd.Timestamp(f): v for f, v in est["historial"]})
            ops = pd.DataFrame(est["operaciones"])
            r = resumen(h, "2026-01-01")
            efe = pd.Series(ef, index=h.index)
            filas.append(dict(inicio=inicio, estrategia=e.nombre, ultimo_dia=str(h.index[-1].date()), **r,
                              efectivo_medio_2026_pct=round(100 * efe[efe.index >= "2026-01-01"].mean(), 1),
                              compras_2026=int(((ops.tipo == "compra") & (ops.fecha >= "2026-01-01")).sum()),
                              stops_fase1_2026=int(((ops.motivo == "stop fase 1") & (ops.fecha >= "2026-01-01")).sum())))
            curvas[f"{inicio}|{e.nombre}"] = h
            print(filas[-1], flush=True)
            ops.to_csv(os.path.join(CARPETA, f"ops_{inicio[:4]}_{e.nombre}.csv"), index=False)
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(CARPETA, "resultados.csv"), index=False)
    pd.DataFrame(curvas).to_csv(os.path.join(CARPETA, "curvas.csv"))
    meses = pd.DataFrame({k: v[v.index >= "2025-12-24"].resample("ME").last() for k, v in curvas.items()})
    (100 * (meses / meses.iloc[0] - 1)).round(1).to_csv(os.path.join(CARPETA, "mensual_2026.csv"))
    with open(os.path.join(CARPETA, "descarga.txt"), "w") as f:
        f.write(f"{len(con)} acciones con datos\n")


if __name__ == "__main__":
    main()
