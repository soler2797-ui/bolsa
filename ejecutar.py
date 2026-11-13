"""Simulador de estrategias de bolsa con dinero ficticio.

Uso:
  python ejecutar.py diario                         -> procesa los días nuevos y actualiza el informe
  python ejecutar.py backtest --desde 2016-01-01    -> prueba las estrategias con datos históricos
  python ejecutar.py universo                       -> vuelve a buscar la lista de acciones pequeñas
"""
import argparse
import datetime as dt
import json
import os
import sys

import pandas as pd

import config
from bolsa import datos, informe, universos
from bolsa.estrategias import crear_estrategias
from bolsa.motor import nuevo_estado, procesar_dia

RAIZ = os.path.dirname(os.path.abspath(__file__))


def _guardar(estados, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    tmp = ruta + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(estados, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, ruta)


def preparar(con_pequenas=True, mercado=None, anios=None):
    univ = universos.cargar(refrescar_pequenas=con_pequenas)
    if not con_pequenas:
        univ["pequenas"], univ["pequenas_primera"] = [], {}
    estrategias = crear_estrategias(univ)
    if not con_pequenas:
        estrategias = [e for e in estrategias if e.nombre != "pequenas_azar"]
    tickers = sorted({t for e in estrategias for t in e.tickers_necesarios()})
    m = mercado or datos.cargar_mercado(tickers, anios or config.ANIOS_HISTORIA)
    return univ, estrategias, m


def diario(mercado=None, carpeta_estado=None, ahora=None):
    carpeta_estado = carpeta_estado or os.path.join(RAIZ, "estado")
    ruta = os.path.join(carpeta_estado, "estado.json")
    estados = json.load(open(ruta, encoding="utf-8")) if os.path.exists(ruta) else {}
    univ, estrategias, m = preparar(True, mercado)
    fechas = m.fechas
    ahora = ahora or dt.datetime.now(dt.timezone.utc)
    # si los mercados de EE. UU. aún no han cerrado, el último día está incompleto
    if len(fechas) and fechas[-1].date() >= ahora.date() and ahora.hour < 22:
        fechas = fechas[:-1]
    for e in estrategias:
        est = estados.get(e.nombre) or nuevo_estado(e.nombre)
        if est["ultima_fecha"] is None:
            nuevas = fechas[-1:]  # el primer día solo se decide; se compra en la apertura siguiente
        else:
            nuevas = fechas[fechas > pd.Timestamp(est["ultima_fecha"])]
        primera = str(nuevas[0].date()) if len(nuevas) else None
        for d in nuevas:
            procesar_dia(est, e, d, m, splits_antes_de=primera)
        estados[e.nombre] = est
        if len(nuevas):
            print(f"{e.nombre}: {len(nuevas)} día(s) procesados hasta {est['ultima_fecha']}")
    _guardar(estados, ruta)
    desc = {e.nombre: e.descripcion for e in estrategias}
    informe.generar(estados, desc, univ["nombres"], os.path.join(RAIZ, "docs", "index.html"),
                    "Simulación en tiempo real")
    informe.exportar_csv(estados, carpeta_estado)
    return estados


def backtest(desde, hasta=None, con_pequenas=False, mercado=None, carpeta=None):
    carpeta = carpeta or os.path.join(RAIZ, "resultados_backtest")
    anios = (dt.date.today() - pd.Timestamp(desde).date()).days / 365.25 + config.REBOTE["anios_atras"] + 1
    univ, estrategias, m = preparar(con_pequenas, mercado, anios)
    if con_pequenas:
        print("Aviso: la lista de acciones pequeñas es la de hoy; el backtest con ella está sesgado.")
    fechas = m.fechas[(m.fechas >= pd.Timestamp(desde)) & (m.fechas <= pd.Timestamp(hasta or "2100-01-01"))]
    estados = {}
    for e in estrategias:
        est = nuevo_estado(e.nombre)
        for d in fechas:
            procesar_dia(est, e, d, m)
        estados[e.nombre] = est
        print(f"{e.nombre}: {est['historial'][-1][1]:,.0f} €" if est["historial"] else e.nombre)
    _guardar(estados, os.path.join(carpeta, "estado.json"))
    desc = {e.nombre: e.descripcion for e in estrategias}
    informe.generar(estados, desc, univ["nombres"], os.path.join(carpeta, "informe.html"),
                    f"Backtest desde {desde}")
    informe.exportar_csv(estados, carpeta)
    return estados


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["diario", "backtest", "universo"])
    ap.add_argument("--desde", default="2016-01-01")
    ap.add_argument("--hasta", default=None)
    ap.add_argument("--con-pequenas", action="store_true")
    ap.add_argument("--carpeta", default=None, help="carpeta de resultados del backtest")
    a = ap.parse_args()
    os.chdir(RAIZ)
    sys.path.insert(0, RAIZ)
    if a.modo == "diario":
        diario()
    elif a.modo == "backtest":
        backtest(a.desde, a.hasta, a.con_pequenas, carpeta=a.carpeta)
    else:
        universos.buscar_pequenas()
