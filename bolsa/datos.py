"""Descarga de precios de Yahoo Finance y conversión de todo a euros."""
import datetime as dt
import hashlib
import os
import pickle
import time

import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datos_cache")

# Moneda según el sufijo del ticker en Yahoo
SUFIJOS = {
    "MC": "EUR", "DE": "EUR", "F": "EUR", "PA": "EUR", "AS": "EUR", "MI": "EUR",
    "BR": "EUR", "LS": "EUR", "HE": "EUR", "IR": "EUR", "VI": "EUR",
    "L": "GBp", "SW": "CHF", "ST": "SEK", "CO": "DKK", "OL": "NOK",
    "TO": "CAD", "V": "CAD", "NE": "CAD", "T": "JPY", "HK": "HKD", "AX": "AUD",
    "KS": "KRW", "TW": "TWD", "NS": "INR", "SS": "CNY", "SZ": "CNY",
}


def moneda(ticker):
    if "." in ticker:
        return SUFIJOS.get(ticker.rsplit(".", 1)[1], "USD")
    return "USD"


def _descargar_yahoo(tickers, inicio):
    import yfinance as yf
    trozos = []
    tickers = list(tickers)
    for i in range(0, len(tickers), 80):
        lote = tickers[i:i + 80]
        for intento in range(3):
            try:
                df = yf.download(lote, start=inicio, auto_adjust=False, actions=True,
                                 group_by="column", threads=True, progress=False,
                                 multi_level_index=True)
                break
            except Exception as e:  # red caída, límite de peticiones...
                print(f"  aviso: fallo descargando lote {i}: {e}; reintento")
                time.sleep(5 * (intento + 1))
                df = None
        if df is not None and len(df):
            trozos.append(df)
    if not trozos:
        return {}
    df = pd.concat(trozos, axis=1)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    campos = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
              "Volume": "volume", "Dividends": "div", "Stock Splits": "split"}
    out = {}
    for yc, nombre in campos.items():
        if yc in df.columns.get_level_values(0):
            out[nombre] = df[yc].copy()
    return out


def descargar(tickers, inicio, usar_cache=True):
    """Devuelve dict campo -> DataFrame (fechas x tickers) en moneda local."""
    os.makedirs(CACHE, exist_ok=True)
    h = hashlib.md5(("|".join(sorted(tickers)) + inicio).encode()).hexdigest()[:12]
    clave = os.path.join(CACHE, f"{dt.date.today().isoformat()}_{h}.pkl")
    if usar_cache and os.path.exists(clave):
        with open(clave, "rb") as f:
            return pickle.load(f)
    datos = _descargar_yahoo(tickers, inicio)
    if usar_cache and datos:
        with open(clave, "wb") as f:
            pickle.dump(datos, f)
    return datos


class Mercado:
    """Precios diarios en euros, con dividendos (en euros por acción) y splits."""

    def __init__(self, crudo, fx):
        """crudo: dict campo->DataFrame en moneda local. fx: DataFrame moneda->unidades por 1 EUR."""
        close = crudo["close"]
        fechas = close.index
        crudo = {k: v.reindex(index=fechas, columns=close.columns) for k, v in crudo.items()}
        self.fechas = fechas
        conv = pd.DataFrame(1.0, index=fechas, columns=close.columns)
        fx = fx.reindex(fechas).ffill().bfill() if fx is not None and len(fx.columns) else None
        for t in close.columns:
            m = moneda(t)
            factor = 1.0
            if m == "GBp":
                factor, m = 0.01, "GBP"
            if m != "EUR":
                if fx is None or m not in fx.columns:
                    conv[t] = np.nan  # sin tipo de cambio: se ignora el ticker
                    continue
                conv[t] = factor / fx[m]
            else:
                conv[t] = factor
        self.open = crudo["open"] * conv
        self.high = crudo["high"] * conv
        self.low = crudo["low"] * conv
        self.close = close * conv
        self.volume = crudo.get("volume", pd.DataFrame(0, index=fechas, columns=close.columns))
        div = crudo.get("div")
        self.div = (div.fillna(0) * conv) if div is not None else pd.DataFrame(0.0, index=fechas, columns=close.columns)
        spl = crudo.get("split")
        self.split = spl.fillna(0) if spl is not None else pd.DataFrame(0.0, index=fechas, columns=close.columns)
        # último cierre conocido (para valorar en días sin cotización)
        self.close_ffill = self.close.ffill()
        self.tickers = [t for t in close.columns if self.close[t].notna().sum() > 0]
        self._ind = {}

    # ---- indicadores, calculados por ticker sobre sus propios días de cotización ----
    def _por_ticker(self, df, fn):
        res = {}
        for t in df.columns:
            s = df[t].dropna()
            res[t] = fn(s).reindex(df.index) if len(s) else pd.Series(np.nan, index=df.index)
        return pd.DataFrame(res)

    def ind(self, nombre):
        if nombre in self._ind:
            return self._ind[nombre]
        c, h, v = self.close, self.high, self.volume
        if nombre.endswith("_prev") and nombre != "max252_prev":
            base = self.ind(nombre[:-5]) if nombre[:-5] != "close" else c
            r = pd.DataFrame({t: base[t].where(c[t].notna()).dropna().shift(1).reindex(c.index)
                              for t in c.columns})
        elif nombre == "sma50":
            r = self._por_ticker(c, lambda s: s.rolling(50).mean())
        elif nombre == "sma200":
            r = self._por_ticker(c, lambda s: s.rolling(200).mean())
        elif nombre == "max252_prev":
            r = self._por_ticker(h, lambda s: s.rolling(252).max().shift(1))
        elif nombre == "vol50":
            r = self._por_ticker(v.where(c.notna()), lambda s: s.rolling(50).mean().shift(1))
        elif nombre == "importe50":
            r = self._por_ticker((c * v), lambda s: s.rolling(50).mean())
        elif nombre == "mom12_1":
            r = self._por_ticker(c, lambda s: s.shift(21) / s.shift(252) - 1)
        elif nombre == "volat252":
            r = self._por_ticker(c, lambda s: np.log(s).diff().rolling(252).std())
        elif nombre == "max5a":
            r = self._por_ticker(c, lambda s: s.rolling(1260, min_periods=1000).max())
        elif nombre.startswith("hace_anios_"):
            n = int(nombre.rsplit("_", 1)[1])
            r = self._por_ticker(c, lambda s: s.shift(252 * n))
        elif nombre == "n_sesiones":
            r = self._por_ticker(c, lambda s: pd.Series(np.arange(1, len(s) + 1), index=s.index))
        else:
            raise KeyError(nombre)
        self._ind[nombre] = r
        return r


def tipos_de_cambio(monedas, inicio):
    monedas = sorted({"GBP" if m == "GBp" else m for m in monedas} - {"EUR"})
    if not monedas:
        return pd.DataFrame()
    tick = {f"EUR{m}=X": m for m in monedas}
    crudo = descargar(list(tick), inicio)
    if not crudo:
        return pd.DataFrame()
    fx = crudo["close"].rename(columns=tick)
    return fx


def cargar_mercado(tickers, anios=12):
    inicio = (dt.date.today() - dt.timedelta(days=int(365.25 * anios))).isoformat()
    tickers = sorted(set(tickers))
    print(f"Descargando {len(tickers)} valores desde {inicio}...")
    crudo = descargar(tickers, inicio)
    if not crudo:
        raise RuntimeError("No se han podido descargar precios de Yahoo Finance")
    fx = tipos_de_cambio([moneda(t) for t in tickers], inicio)
    sin_datos = [t for t in tickers if t not in crudo["close"].columns or crudo["close"][t].notna().sum() == 0]
    if sin_datos:
        print(f"  Sin datos ({len(sin_datos)}): {', '.join(sin_datos[:30])}{' ...' if len(sin_datos) > 30 else ''}")
    return Mercado(crudo, fx)
