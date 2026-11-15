"""Control ampliado para Kronos fuera de muestra: las variantes que salieron mejor, frente a N sorteos.

Usa docs/kronos_oos/previsiones_y_real.csv (no necesita el modelo). Cada sorteo baraja las previsiones de
Kronos entre las acciones dentro de cada semana y repite la misma regla.
  python experimentos/kronos_control.py [semillas]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kronos_oos as K  # noqa: E402

VARIANTES = [("puntuacion_fija", "k0", dict(umbral=0.0)), ("puntuacion_fija", "k1", dict(umbral=1.0)),
             ("puntuacion_fija", "k2", dict(umbral=2.0)), ("momentum", "orden20", dict(reordenar=20))]


def main(semillas):
    univ, m = K.mercado()
    p = pd.read_csv(os.path.join(K.CARPETA, "previsiones_y_real.csv"), parse_dates=["fecha"])
    K.kronos_ia._error = "solo previsiones guardadas"
    fechas = m.fechas[(m.fechas >= K.DESDE) & (m.fechas <= K.HASTA)]

    def cache(col):
        K.kronos_ia._cache.clear()
        for f, t, v in zip(p.fecha, p.ticker, p[col]):
            K.kronos_ia._cache[(t, f)] = v

    def total(base, nombre, kw):
        e = K.Momentum(univ) if base == "momentum" else K.Puntuacion(univ, tamano_variable=False)
        e = K.con_filtro(e, f"{base}_{nombre}", **kw)
        est = K.nuevo_estado(e.nombre)
        for d in fechas:
            K.procesar_dia(est, e, d, m)
        return K.metricas(est)["total_pct"]

    filas = []
    cache("prevision_pct")
    for base, nombre, kw in VARIANTES:
        filas.append(dict(variante=f"{base}_{nombre}", sorteo="kronos", total_pct=total(base, nombre, kw)))
    for s in range(1, semillas + 1):
        rng = np.random.default_rng(1000 + s)
        p["azar"] = p.groupby("fecha").prevision_pct.transform(lambda x: rng.permutation(x.values))
        cache("azar")
        for base, nombre, kw in VARIANTES:
            filas.append(dict(variante=f"{base}_{nombre}", sorteo=s, total_pct=total(base, nombre, kw)))
        print("sorteo", s, flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(K.CARPETA, "control_ampliado.csv"), index=False)
    res = []
    for v, g in df.groupby("variante"):
        kr = g[g.sorteo == "kronos"].total_pct.iloc[0]
        az = g[g.sorteo != "kronos"].total_pct
        res.append(dict(variante=v, kronos=kr, azar_mediana=az.median(), azar_p90=az.quantile(0.9), azar_max=az.max(),
                        sorteos=len(az), sorteos_mejores_que_kronos=int((az >= kr).sum())))
    r = pd.DataFrame(res).round(1)
    r.to_csv(os.path.join(K.CARPETA, "control_ampliado_resumen.csv"), index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
