"""Listas de acciones: grandes (archivo fijo, editable) y pequeñas (buscador de Yahoo Finance)."""
import datetime as dt
import os

import pandas as pd

import config

RAIZ = os.path.dirname(os.path.dirname(__file__))
F_GRANDES = os.path.join(RAIZ, "universos", "grandes.csv")
F_PEQUENAS = os.path.join(RAIZ, "universos", "pequenas.csv")


def buscar_pequenas():
    """Usa el buscador de acciones de Yahoo para hacer la lista de empresas pequeñas."""
    import yfinance as yf
    from yfinance import EquityQuery as Q
    cfg = config.PEQUENAS
    hoy = dt.datetime.now()
    filas = []
    for region in cfg["regiones"]:
        q = Q("and", [Q("eq", ["region", region]),
                      Q("btwn", ["intradaymarketcap", cfg["capitalizacion_min"], cfg["capitalizacion_max"]])])
        obtenidas = 0
        for offset in range(0, cfg["max_por_region"] * 3, 250):
            try:
                r = yf.screen(q, offset=offset, size=250, sortField="intradaymarketcap", sortAsc=False)
            except Exception as e:
                print(f"  aviso: buscador falla en {region}: {e}")
                break
            quotes = r.get("quotes", []) if isinstance(r, dict) else []
            if not quotes:
                break
            for c in quotes:
                ms = c.get("firstTradeDateMilliseconds")
                if not ms:
                    continue
                f0 = dt.datetime.utcfromtimestamp(ms / 1000)
                anios = (hoy - f0).days / 365.25
                if cfg["anios_cotizando_min"] <= anios <= cfg["anios_cotizando_max"] + 1:
                    filas.append(dict(ticker=c["symbol"], nombre=c.get("shortName") or c.get("longName", ""),
                                      region=region, moneda=c.get("currency", ""),
                                      primera_cotizacion=f0.date().isoformat(),
                                      capitalizacion=c.get("marketCap")))
                    obtenidas += 1
            if obtenidas >= cfg["max_por_region"]:
                break
    if filas:
        df = pd.DataFrame(filas).drop_duplicates("ticker")
        df.to_csv(F_PEQUENAS, index=False)
        print(f"Lista de acciones pequeñas actualizada: {len(df)} valores")
    else:
        print("No se ha podido actualizar la lista de acciones pequeñas; se mantiene la anterior")


def cargar(refrescar_pequenas=True):
    g = pd.read_csv(F_GRANDES)
    universos = dict(grandes=list(g.ticker), sectores=dict(zip(g.ticker, g.sector)),
                     nombres=dict(zip(g.ticker, g.nombre)))
    if refrescar_pequenas:
        viejo = (not os.path.exists(F_PEQUENAS) or
                 (dt.datetime.now().timestamp() - os.path.getmtime(F_PEQUENAS)) / 86400 > config.PEQUENAS["dias_refresco"])
        if viejo:
            try:
                buscar_pequenas()
            except Exception as e:
                print(f"  aviso: no se pudo usar el buscador de Yahoo: {e}")
    if os.path.exists(F_PEQUENAS):
        p = pd.read_csv(F_PEQUENAS)
        universos["pequenas"] = list(p.ticker)
        universos["pequenas_primera"] = dict(zip(p.ticker, p.primera_cotizacion))
        from .datos import MONEDAS_CONOCIDAS
        validas = {"EUR", "USD", "GBp", "GBP", "CAD", "CHF", "SEK", "DKK", "NOK", "JPY", "HKD", "AUD"}
        MONEDAS_CONOCIDAS.update({t: mo for t, mo in zip(p.ticker, p.moneda.fillna("")) if mo in validas})
        universos["nombres"].update(dict(zip(p.ticker, p.nombre.fillna(""))))
    else:
        universos["pequenas"] = []
        universos["pequenas_primera"] = {}
    return universos
