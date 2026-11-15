"""Diagnóstico: ¿da lo mismo la misma regla con dos descargas de datos distintas?"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kronos_oos as K  # noqa: E402


def correr(univ, m, p):
    K.kronos_ia._error = "solo previsiones guardadas"
    K.kronos_ia._cache.clear()
    for f, t, v in zip(p.fecha, p.ticker, p.prevision_pct):
        K.kronos_ia._cache[(t, f)] = v
    fechas = m.fechas[(m.fechas >= K.DESDE) & (m.fechas <= K.HASTA)]
    out = {}
    for nombre, hacer in [("puntuacion_fija", lambda: K.Puntuacion(univ, tamano_variable=False)),
                          ("puntuacion_fija_k0", lambda: K.con_filtro(K.Puntuacion(univ, tamano_variable=False), "x", umbral=0.0)),
                          ("momentum", lambda: K.Momentum(univ))]:
        for rep in (1, 2):
            e = hacer()
            est = K.nuevo_estado(nombre)
            for d in fechas:
                K.procesar_dia(est, e, d, m)
            out[f"{nombre}_rep{rep}"] = K.metricas(est)["total_pct"]
    return out


p = pd.read_csv(os.path.join(K.CARPETA, "previsiones_y_real.csv"), parse_dates=["fecha"])
filas, conjuntos = [], []
for descarga in (1, 2):
    univ, m = K.mercado()
    con = sorted(t for t in univ["grandes"] if t in m.close.columns and m.close[t].notna().sum() > 0)
    conjuntos.append(set(con))
    sub = m.close.loc[K.DESDE:K.HASTA]
    filas.append(dict(descarga=descarga, tickers_con_datos=len(con), suma_cierres=float(np.nansum(sub.values)),
                      **correr(univ, m, p)))
    print(filas[-1], flush=True)
dif = sorted(conjuntos[0] ^ conjuntos[1])
pd.DataFrame(filas).to_csv(os.path.join(K.CARPETA, "diagnostico.csv"), index=False)
with open(os.path.join(K.CARPETA, "diagnostico.txt"), "w") as f:
    f.write(f"tickers distintos entre descargas: {dif}\n")
print("tickers distintos entre descargas:", dif)
