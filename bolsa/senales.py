"""Búsqueda de señales independientes.

Para cada señal y cada fecha de muestra se ordenan las acciones por la señal y se compara ese orden con la
rentabilidad de las sesiones siguientes, descontando la media de todas las acciones ese día (así no cuenta
si el mercado subió o bajó). Se mide:
 - IC: correlación de Spearman entre señal y rentabilidad futura, de media en todas las fechas.
 - t: si el IC es distinto de cero de forma consistente (más de 2 en valor absoluto suele indicar que no es casualidad).
 - Diferencia entre el 20 % de acciones con la señal más alta y el 20 % con la más baja.
 - Correlación entre señales: si dos se parecen mucho, miden lo mismo.
 - Regresión con todas las señales a la vez (Fama-MacBeth): lo que aporta cada señal descontando las demás.
"""
import base64
import html
import io
import os

import numpy as np
import pandas as pd

NOMBRES = {
    "mom_12_1": "Momentum 12 meses sin el último",
    "mom_6_1": "Momentum 6 meses sin el último",
    "ret_1m": "Rentabilidad del último mes",
    "ret_1s": "Rentabilidad de la última semana",
    "dist_max52": "Distancia al máximo de 52 semanas",
    "ruptura_vol": "Ruptura del máximo de 52 semanas con volumen (últimos 5 días)",
    "volumen_rel": "Volumen de 20 sesiones frente al de un año",
    "dist_sma200": "Distancia a la media de 200 sesiones",
    "pend_sma200": "Pendiente de la media de 200 sesiones (1 mes)",
    "dist_sma50": "Distancia a la media de 50 sesiones",
    "caida_5a": "Caída desde el máximo de 5 años",
    "volatilidad": "Volatilidad de 1 año",
    "rsi14": "RSI de 14 sesiones",
}


def _serie(df, fn):
    out = {}
    for t in df.columns:
        s = df[t].dropna()
        out[t] = fn(s).reindex(df.index) if len(s) > 300 else pd.Series(np.nan, index=df.index)
    return pd.DataFrame(out)


def _rsi(s, n=14):
    d = s.diff()
    sube = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    baja = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + sube / baja)


def calcular_senales(m, tickers):
    c = m.close[tickers]
    h = m.high[tickers]
    v = m.volume[tickers].where(c.notna())
    s = {}
    s["mom_12_1"] = _serie(c, lambda x: x.shift(21) / x.shift(252) - 1)
    s["mom_6_1"] = _serie(c, lambda x: x.shift(21) / x.shift(126) - 1)
    s["ret_1m"] = _serie(c, lambda x: x / x.shift(21) - 1)
    s["ret_1s"] = _serie(c, lambda x: x / x.shift(5) - 1)
    s["dist_max52"] = _serie(c, lambda x: x / x.rolling(252).max() - 1)
    rupt = {}
    for t in tickers:
        x, hh, vv = c[t].dropna(), h[t].dropna(), v[t].dropna()
        if len(x) < 300:
            rupt[t] = pd.Series(np.nan, index=c.index)
            continue
        hh, vv = hh.reindex(x.index), vv.reindex(x.index)
        r = ((x > hh.rolling(252).max().shift(1)) & (vv > 1.5 * vv.rolling(50).mean().shift(1))).astype(float)
        rupt[t] = r.rolling(5).max().reindex(c.index)
    s["ruptura_vol"] = pd.DataFrame(rupt)
    s["volumen_rel"] = _serie(v, lambda x: np.log((x.rolling(20).mean() + 1) / (x.rolling(252).mean() + 1)))
    s["dist_sma200"] = _serie(c, lambda x: x / x.rolling(200).mean() - 1)
    s["pend_sma200"] = _serie(c, lambda x: x.rolling(200).mean() / x.rolling(200).mean().shift(21) - 1)
    s["dist_sma50"] = _serie(c, lambda x: x / x.rolling(50).mean() - 1)
    s["caida_5a"] = _serie(c, lambda x: x / x.rolling(1260, min_periods=1000).max() - 1)
    s["volatilidad"] = _serie(c, lambda x: np.log(x).diff().rolling(252).std() * np.sqrt(252))
    s["rsi14"] = _serie(c, _rsi)
    return s


def futuras(m, tickers, n):
    return _serie(m.close[tickers], lambda x: x.shift(-n) / x - 1)


def analizar(m, tickers, desde, hasta, carpeta, paso=10):
    os.makedirs(carpeta, exist_ok=True)
    desde, hasta = pd.Timestamp(desde), pd.Timestamp(hasta)
    sen = calcular_senales(m, tickers)
    fut = {n: futuras(m, tickers, n) for n in (20, 60)}
    fechas = m.fechas[(m.fechas >= desde) & (m.fechas <= hasta)]
    # las rentabilidades futuras no pueden usar precios posteriores a 'hasta' (periodo de comprobación)
    limite = {n: hasta - pd.tseries.offsets.BDay(int(n * 1.05) + 2) for n in fut}
    muestras = fechas[::paso]
    sen = {k: v.ffill(limit=3) for k, v in sen.items()}
    fut = {k: v.ffill(limit=3) for k, v in fut.items()}

    filas, ics = [], {n: {k: [] for k in sen} for n in fut}
    cors = []
    fm = {n: [] for n in fut}
    claves = list(sen)
    for d in muestras:
        X = pd.DataFrame({k: sen[k].loc[d] for k in claves})
        cors.append(X.rank().corr(method="pearson"))
        for n in fut:
            if d > limite[n]:
                continue
            y = fut[n].loc[d]
            ok = y.notna()
            if ok.sum() < 30:
                continue
            y = y[ok] - y[ok].mean()
            Xn = X[ok]
            for k in claves:
                x = Xn[k]
                v = x.notna()
                if v.sum() < 30 or x[v].nunique() < 2:
                    continue
                ic = x[v].rank().corr(y[v].rank())
                q = pd.qcut(x[v].rank(method="first"), 5, labels=False)
                dif = y[v][q == 4].mean() - y[v][q == 0].mean()
                ics[n][k].append((ic, dif))
            # regresión transversal con todas las señales estandarizadas (rangos)
            Z = Xn.rank(pct=True) - 0.5
            Z = Z.dropna(axis=1, how="all").fillna(0)
            if len(Z) > len(Z.columns) + 10:
                A = np.column_stack([np.ones(len(Z)), Z.values])
                coef = np.linalg.lstsq(A, y.values, rcond=None)[0][1:]
                fm[n].append(pd.Series(coef, index=Z.columns))

    for k in claves:
        fila = dict(senal=k, nombre=NOMBRES[k])
        for n in fut:
            arr = np.array(ics[n][k])
            if len(arr) < 10:
                continue
            ic, dif = arr[:, 0], arr[:, 1]
            fila[f"ic_{n}"] = ic.mean()
            fila[f"t_{n}"] = ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic) / (n / paso)) if ic.std() > 0 else np.nan
            fila[f"pos_{n}"] = 100 * (ic > 0).mean()
            fila[f"dif_{n}"] = 100 * dif.mean()
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    corr = sum(cors) / len(cors)
    regr = {}
    for n in fut:
        if fm[n]:
            F = pd.DataFrame(fm[n])
            regr[n] = pd.DataFrame({"coef": 100 * F.mean(),
                                    "t": F.mean() / F.std(ddof=1) * np.sqrt(len(F) / (n / paso))})
    tabla.to_csv(os.path.join(carpeta, "senales.csv"), index=False)
    corr.to_csv(os.path.join(carpeta, "correlaciones.csv"))
    _html(tabla, corr, regr, muestras, desde, hasta, os.path.join(carpeta, "index.html"))
    return tabla, corr, regr


def _mapa(corr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=110)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(corr.values[i, j]) > 0.6 else "black")
    fig.colorbar(im, fraction=0.04)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png")
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def _html(tabla, corr, regr, muestras, desde, hasta, ruta):
    f = lambda x, d=3: "" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")
    t = tabla.sort_values("t_60", key=lambda s: -s.abs())
    filas = "".join(
        f"<tr><td>{x.senal}</td><td>{html.escape(x.nombre)}</td>"
        + "".join(f"<td>{f(getattr(x, f'ic_{n}', np.nan))}</td><td>{f(getattr(x, f't_{n}', np.nan), 1)}</td>"
                  f"<td>{f(getattr(x, f'pos_{n}', np.nan), 0)} %</td><td>{f(getattr(x, f'dif_{n}', np.nan), 2)} %</td>"
                  for n in (20, 60)) + "</tr>" for x in t.itertuples())
    bloques_r = ""
    for n, r in regr.items():
        r = r.sort_values("t", key=lambda s: -s.abs())
        bloques_r += (f"<h3>Rentabilidad a {n} sesiones</h3><table><tr><th>Señal</th><th>Coeficiente (%)</th><th>t</th></tr>"
                      + "".join(f"<tr><td>{k}</td><td>{f(v.coef, 2)}</td><td>{f(v.t, 1)}</td></tr>" for k, v in r.iterrows())
                      + "</table>")
    img = _mapa(corr)
    doc = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Señales</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222;max-width:1200px}}
table{{border-collapse:collapse;font-size:13px;margin-bottom:24px}}td,th{{border-bottom:1px solid #ddd;padding:4px 8px;text-align:right;white-space:nowrap}}
td:nth-child(-n+2),th:nth-child(-n+2){{text-align:left}}th{{background:#f4f4f4}}</style></head><body>
<h1>Búsqueda de señales</h1>
<p>Periodo: {desde.date()} a {hasta.date()} ({len(muestras)} fechas de muestra, una cada 10 sesiones). Las rentabilidades
futuras no usan precios posteriores al {hasta.date()}, así que el periodo siguiente queda intacto para comprobar.
Rentabilidades descontando la media de todas las acciones ese día.</p>
<h2>Cada señal por separado</h2>
<p>IC: correlación media entre la señal y la rentabilidad futura (0,02-0,05 ya es útil en bolsa). t: más de 2 en valor
absoluto suele indicar que no es casualidad (corregido por el solapamiento de las rentabilidades). "IC positivo": en qué
porcentaje de fechas acertó el signo. Q5-Q1: rentabilidad del 20 % de acciones con la señal más alta menos la del 20 % más
baja. Un IC negativo también sirve: significa que la señal funciona al revés.</p>
<table><tr><th rowspan=2>Señal</th><th rowspan=2>Descripción</th><th colspan=4>20 sesiones</th><th colspan=4>60 sesiones</th></tr>
<tr><th>IC</th><th>t</th><th>IC positivo</th><th>Q5-Q1</th><th>IC</th><th>t</th><th>IC positivo</th><th>Q5-Q1</th></tr>{filas}</table>
<h2>Parecido entre señales</h2>
<p>Correlación media entre los rangos de las señales. Por encima de 0,7 en valor absoluto, las dos señales miden casi lo mismo.</p>
<img src="data:image/png;base64,{img}" style="max-width:800px;width:100%">
<h2>Todas las señales a la vez</h2>
<p>Regresión de la rentabilidad futura sobre todas las señales juntas. El coeficiente es la diferencia de rentabilidad entre la
acción con la señal más alta y la más baja si las demás señales fueran iguales. Una señal con t alto aquí aporta información
que las demás no tienen.</p>{bloques_r}</body></html>"""
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write(doc)
