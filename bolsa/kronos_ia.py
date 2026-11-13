"""Previsiones con Kronos (modelo de IA para velas de bolsa, github.com/shiyu-coder/Kronos).

Se usa para ajustar la velocidad del stop en fase 2. Si el modelo no se puede cargar, las estrategias
con Kronos funcionan como las normales y el informe lo indica.
"""
import os
import sys

import numpy as np
import pandas as pd

import config

RAIZ = os.path.dirname(os.path.dirname(__file__))
_predictor = None
_error = None
_cache = {}  # (ticker, fecha) -> rentabilidad prevista en %


def _cargar():
    global _predictor, _error
    if _predictor is not None or _error is not None:
        return _predictor
    try:
        cfg = config.KRONOS
        ruta = os.path.join(RAIZ, cfg["carpeta_codigo"])
        if ruta not in sys.path:
            sys.path.insert(0, ruta)
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        from model import Kronos, KronosPredictor, KronosTokenizer
        tok = KronosTokenizer.from_pretrained(cfg["tokenizador"])
        mod = Kronos.from_pretrained(cfg["modelo"])
        _predictor = KronosPredictor(mod, tok, device="cpu", max_context=cfg["contexto_max"])
        print(f"Kronos cargado: {cfg['modelo']}")
    except Exception as e:  # sin torch, sin red, cambio en el repositorio de Kronos...
        _error = f"{type(e).__name__}: {e}"
        print(f"  aviso: Kronos no disponible ({_error}); los stops con Kronos usarán la distancia normal")
    return _predictor


_prueba = None


def comprobar(m):
    """Carga el modelo y hace una previsión de prueba, para saber cada día si Kronos funciona."""
    global _prueba
    if _cargar() is None:
        return
    for t in ("SPY", "IWDA.AS", *m.tickers):
        if t in m.close.columns and m.close[t].notna().sum() > config.KRONOS["velas"]:
            d = m.close[t].dropna().index[-1]
            r = prever(m, [t], d)[t]
            if r is not None:
                _prueba = (t, r)
            return


def estado():
    if _predictor is None and _error is None:
        return "no cargado"
    if _error:
        return f"error: {_error}"
    if _prueba:
        return f"funcionando (prueba: prevé {_prueba[1]:+.1f} % para {_prueba[0]} en {config.KRONOS['horizonte']} sesiones)"
    return "cargado, pero la previsión de prueba ha fallado"


def prever(m, tickers, d):
    """Rentabilidad prevista (%) del cierre dentro de KRONOS['horizonte'] sesiones, para cada ticker."""
    cfg = config.KRONOS
    faltan = [t for t in tickers if (t, d) not in _cache]
    if faltan:
        pred = _cargar()
        if pred is None:
            for t in faltan:
                _cache[(t, d)] = None
        else:
            dfs, xs, ys, validos = [], [], [], []
            futuro = pd.bdate_range(d + pd.Timedelta(days=1), periods=cfg["horizonte"])
            for t in faltan:
                df = pd.DataFrame({"open": m.open[t], "high": m.high[t], "low": m.low[t],
                                   "close": m.close[t], "volume": m.volume[t]}).loc[:d].dropna()
                df = df[df["close"] > 0].iloc[-cfg["velas"]:]
                if len(df) < cfg["velas"]:
                    _cache[(t, d)] = None
                    continue
                dfs.append(df.reset_index(drop=True))
                xs.append(pd.Series(df.index))
                ys.append(pd.Series(futuro))
                validos.append((t, df["close"].iloc[-1]))
            for i in range(0, len(dfs), cfg["lote"]):
                try:
                    res = pred.predict_batch(dfs[i:i + cfg["lote"]], xs[i:i + cfg["lote"]], ys[i:i + cfg["lote"]],
                                             pred_len=cfg["horizonte"], T=1.0, top_p=0.9,
                                             sample_count=cfg["muestras"], verbose=False)
                    for (t, ultimo), p in zip(validos[i:i + cfg["lote"]], res):
                        _cache[(t, d)] = float(100 * (p["close"].iloc[-1] / ultimo - 1))
                except Exception as e:
                    print(f"  aviso: fallo en la previsión de Kronos: {e}")
                    for t, _ in validos[i:i + cfg["lote"]]:
                        _cache[(t, d)] = None
    return {t: _cache.get((t, d)) for t in tickers}


def distancia_fase2(prevision):
    """Distancia del stop en fase 2 según lo que prevé Kronos."""
    cfg = config.KRONOS
    base = config.STOP_FASE2
    if prevision is None:
        return base
    if prevision >= cfg["umbral_sube"]:
        return base + cfg["margen_extra"]
    if prevision <= -cfg["umbral_baja"]:
        return max(0.02, base - cfg["margen_menos"])
    return base


# ---------------------------------------------------------------------------
# Seguimiento: qué predijo Kronos y qué pasó después
COLUMNAS = ["fecha", "ticker", "precio", "prevision_pct", "horizonte"]


def registrar(m, estados, ruta):
    """Guarda la previsión de hoy para las acciones que tiene alguna estrategia (más SPY como referencia)."""
    if _cargar() is None:
        return
    cfg = config.KRONOS
    tickers = sorted({t for e in estados.values() for t in e["posiciones"]} - {"SPY"})
    tickers = ["SPY"] + tickers
    tickers = [t for t in tickers if t in m.close.columns and m.close[t].notna().sum() > cfg["velas"]]
    tickers = tickers[:cfg["seguimiento_max"]]
    previas = pd.read_csv(ruta) if os.path.exists(ruta) else pd.DataFrame(columns=COLUMNAS)
    hechas = set(zip(previas["fecha"].astype(str), previas["ticker"]))
    por_fecha = {}
    for t in tickers:
        d = m.close[t].dropna().index[-1]
        if (str(d.date()), t) not in hechas:
            por_fecha.setdefault(d, []).append(t)
    filas = []
    for d, ts in por_fecha.items():
        res = prever(m, ts, d)
        for t in ts:
            if res[t] is not None:
                filas.append(dict(fecha=str(d.date()), ticker=t, precio=round(float(m.close.at[d, t]), 4),
                                  prevision_pct=round(res[t], 3), horizonte=cfg["horizonte"]))
    if filas:
        pd.concat([previas, pd.DataFrame(filas)], ignore_index=True).to_csv(ruta, index=False)
        print(f"Kronos: {len(filas)} previsiones guardadas")


def evaluar(m, ruta):
    """Compara cada previsión con lo que hizo el precio en las `horizonte` sesiones siguientes."""
    if not os.path.exists(ruta):
        return pd.DataFrame()
    p = pd.read_csv(ruta)
    reales = []
    for x in p.itertuples():
        real = np.nan
        if x.ticker in m.close.columns:
            s = m.close[x.ticker].dropna()
            i = s.index.searchsorted(pd.Timestamp(x.fecha))
            if i + x.horizonte < len(s):
                real = 100 * (s.iloc[i + x.horizonte] / s.iloc[i] - 1)
        reales.append(real)
    p["real_pct"] = reales
    return p


def html_seguimiento(p):
    cfg = config.KRONOS
    titulo = "<h2>Kronos: qué predijo y qué pasó</h2>"
    if p is None or p.empty:
        return titulo + "<p>Todavía no hay previsiones guardadas.</p>"
    ev = p.dropna(subset=["real_pct"])
    f = lambda x, d=1: "" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")
    texto = (f"<p>Cada día se guarda lo que Kronos prevé para las {cfg['horizonte']} sesiones siguientes en las acciones "
             f"que tiene alguna estrategia, y el S&P 500 (SPY) como referencia. Previsiones guardadas: {len(p)}; "
             f"ya comprobables: {len(ev)}.</p>")
    if len(ev) < 10:
        return titulo + texto + "<p>Hacen falta al menos 10 previsiones comprobadas para medir el acierto.</p>"

    def fila(nombre, g):
        if len(g) == 0:
            return f"<tr><td>{nombre}</td><td>0</td><td></td><td></td><td></td><td></td></tr>"
        acierto = 100 * (np.sign(g.prevision_pct) == np.sign(g.real_pct)).mean()
        sube = 100 * (g.real_pct > 0).mean()
        corr = g.prevision_pct.corr(g.real_pct, method="spearman") if len(g) > 5 else np.nan
        err = (g.prevision_pct - g.real_pct).abs().mean()
        return (f"<tr><td>{nombre}</td><td>{len(g)}</td><td>{f(acierto, 0)} %</td><td>{f(sube, 0)} %</td>"
                f"<td>{f(corr, 2)}</td><td>{f(err)} %</td></tr>")

    filas = (fila("Todas", ev)
             + fila(f"Prevé subida de {f(cfg['umbral_sube'])} % o más", ev[ev.prevision_pct >= cfg["umbral_sube"]])
             + fila(f"Prevé bajada de {f(cfg['umbral_baja'])} % o más", ev[ev.prevision_pct <= -cfg["umbral_baja"]])
             + fila("Entre medias", ev[ev.prevision_pct.abs() < cfg["umbral_sube"]]))
    ult = ev.sort_values("fecha", ascending=False).head(40)
    lista = "".join(f"<tr><td>{x.fecha}</td><td>{x.ticker}</td><td>{f(x.prevision_pct)} %</td><td>{f(x.real_pct)} %</td>"
                    f"<td>{'sí' if np.sign(x.prevision_pct) == np.sign(x.real_pct) else 'no'}</td></tr>"
                    for x in ult.itertuples())
    return (titulo + texto +
            "<p>Acierto de dirección: en qué porcentaje de casos acertó si subiría o bajaría. Para juzgarlo hay que "
            "compararlo con la columna siguiente, el porcentaje de veces que la acción subió: si sube casi siempre, "
            "decir siempre 'sube' ya acierta mucho. Correlación (Spearman): si las acciones con mejor previsión "
            "acabaron mejor que las demás (0 = nada, 1 = perfecto).</p>"
            "<div class='tabla'><table><tr><th>Previsiones</th><th>N</th><th>Acierta la dirección</th>"
            "<th>La acción subió</th><th>Correlación</th><th>Error medio</th></tr>" + filas + "</table></div>"
            "<h3>Últimas previsiones comprobadas</h3><div class='tabla'><table><tr><th>Fecha</th><th>Ticker</th>"
            "<th>Previsión</th><th>Real</th><th>Acierta la dirección</th></tr>" + lista + "</table></div>")
