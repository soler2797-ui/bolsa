"""Descarga los precios del universo de acciones grandes y de los índices de referencia y los guarda
en experimentos/precios/ (un parquet por campo, en moneda local, más los tipos de cambio).
Solo sirve para hacer pruebas fuera de GitHub; la simulación diaria no lo usa."""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import config
from bolsa import datos, universos

INICIO = "2004-01-01"
univ = universos.cargar(refrescar_pequenas=False)
tickers = sorted(set(univ["grandes"]) | {t for t, _ in config.REFERENCIAS.values()})
crudo = datos.descargar(tickers, INICIO, usar_cache=False)
if not crudo:
    raise SystemExit("sin datos de Yahoo")
fx = datos.tipos_de_cambio([datos.moneda(t) for t in tickers], INICIO)
destino = os.path.join(RAIZ, "experimentos", "precios")
os.makedirs(destino, exist_ok=True)
for campo, df in crudo.items():
    df.to_parquet(os.path.join(destino, f"{campo}.parquet"), compression="zstd")
fx.to_parquet(os.path.join(destino, "fx.parquet"), compression="zstd")
c = crudo["close"]
print(f"{c.shape[1]} valores, {c.shape[0]} sesiones, de {c.index[0].date()} a {c.index[-1].date()}")
print("sin datos:", [t for t in tickers if t not in c.columns or c[t].notna().sum() == 0])
