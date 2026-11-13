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


def estado():
    return "no cargado" if _predictor is None and _error is None else ("funcionando" if _predictor else f"error: {_error}")


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
