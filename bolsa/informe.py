"""Informe HTML con la evolución de cada estrategia, sus métricas y las operaciones."""
import base64
import html
import io
import os

import numpy as np
import pandas as pd


def metricas(est):
    h = pd.DataFrame(est["historial"], columns=["fecha", "valor"])
    if h.empty:
        return {}
    h["fecha"] = pd.to_datetime(h["fecha"])
    v = h["valor"].values
    ini = est["capital_inicial"]
    dd = (v / np.maximum.accumulate(np.r_[ini, v])[1:] - 1).min()
    dias = (h["fecha"].iloc[-1] - h["fecha"].iloc[0]).days
    cerradas = [o for o in est["operaciones"] if "resultado" in o]
    ganadoras = [o for o in cerradas if o["resultado"] > 0]
    anual = ((v[-1] / ini) ** (365.25 / dias) - 1) if dias >= 365 else None
    return dict(valor=v[-1], rent=100 * (v[-1] / ini - 1), anual=None if anual is None else 100 * anual,
                dd=100 * dd, cerradas=len(cerradas),
                pct_gan=100 * len(ganadoras) / len(cerradas) if cerradas else None,
                abiertas=len(est["posiciones"]), desde=h["fecha"].iloc[0].date(), hasta=h["fecha"].iloc[-1].date())


def _grafico(estados, descripciones):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=110)
    for nombre, est in estados.items():
        h = pd.DataFrame(est["historial"], columns=["fecha", "valor"])
        if h.empty:
            continue
        h["fecha"] = pd.to_datetime(h["fecha"])
        y = 100 * h["valor"] / est["capital_inicial"]
        ref = nombre.startswith("ref_")
        ax.plot(h["fecha"], y, label=nombre, lw=1.2 if ref else 1.8, ls="--" if ref else "-")
    ax.axhline(100, color="#888", lw=0.8)
    ax.set_ylabel("Valor (inicio = 100)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _fmt(x, dec=1, suf=""):
    return "" if x is None else f"{x:,.{dec}f}{suf}".replace(",", "X").replace(".", ",").replace("X", ".")


def generar(estados, descripciones, nombres, ruta, titulo="Simulador de estrategias"):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    filas = []
    for n, est in estados.items():
        mt = metricas(est)
        if not mt:
            continue
        filas.append((mt["rent"], f"<tr><td title='{html.escape(descripciones.get(n, ''))}'>{n}</td>"
                      f"<td>{_fmt(mt['valor'], 0, ' €')}</td><td>{_fmt(mt['rent'], 1, ' %')}</td>"
                      f"<td>{_fmt(mt['anual'], 1, ' %')}</td><td>{_fmt(mt['dd'], 1, ' %')}</td>"
                      f"<td>{mt['cerradas']}</td><td>{_fmt(mt['pct_gan'], 0, ' %')}</td><td>{mt['abiertas']}</td></tr>"))
    filas.sort(key=lambda x: -x[0])
    desde = min((metricas(e)["desde"] for e in estados.values() if e["historial"]), default="")
    hasta = max((metricas(e)["hasta"] for e in estados.values() if e["historial"]), default="")

    pos = []
    for n, est in estados.items():
        for t, p in est["posiciones"].items():
            pos.append(f"<tr><td>{n}</td><td>{t}</td><td>{html.escape(str(nombres.get(t, '')))}</td>"
                       f"<td>{_fmt(p['acciones'], 0)}</td><td>{_fmt(p['precio_entrada'], 2)}</td>"
                       f"<td>{p['fecha_entrada']}</td><td>{_fmt(p.get('stop'), 2)}</td>"
                       f"<td>{p.get('fase', '')}</td><td>{p.get('grupo', '')}</td></tr>")
    ops = []
    for n, est in estados.items():
        for o in est["operaciones"]:
            ops.append((o["fecha"], f"<tr><td>{o['fecha']}</td><td>{n}</td><td>{o['tipo']}</td><td>{o['ticker']}</td>"
                        f"<td>{html.escape(str(nombres.get(o['ticker'], '')))}</td><td>{_fmt(o['acciones'], 0)}</td>"
                        f"<td>{_fmt(o['precio'], 2)}</td><td>{_fmt(o.get('resultado'), 2, ' €')}</td>"
                        f"<td>{_fmt(o.get('rentabilidad_pct'), 1, ' %')}</td><td>{html.escape(o['motivo'])}</td></tr>"))
    ops.sort(key=lambda x: x[0], reverse=True)
    desc = "".join(f"<li><b>{n}</b>: {html.escape(d)}</li>" for n, d in descripciones.items())
    img = _grafico(estados, descripciones)
    doc = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>{titulo}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222;max-width:1200px}}
table{{border-collapse:collapse;font-size:13px;margin-bottom:24px}}td,th{{border-bottom:1px solid #ddd;padding:4px 8px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}th{{background:#f4f4f4}}h2{{margin-top:32px}}</style></head><body>
<h1>{titulo}</h1><p>Del {desde} al {hasta}. Valores en euros, comisiones incluidas. Capital inicial por estrategia: el mismo para todas.</p>
<img src="data:image/png;base64,{img}" style="width:100%">
<h2>Resumen</h2><table><tr><th>Estrategia</th><th>Valor</th><th>Rentabilidad</th><th>Anualizada</th>
<th>Caída máxima</th><th>Operaciones cerradas</th><th>Ganadoras</th><th>Abiertas</th></tr>{''.join(f for _, f in filas)}</table>
<h2>Posiciones abiertas</h2><table><tr><th>Estrategia</th><th>Ticker</th><th>Nombre</th><th>Acciones</th><th>Precio entrada (€)</th>
<th>Fecha</th><th>Stop (€)</th><th>Fase</th><th>Par</th></tr>{''.join(pos)}</table>
<h2>Operaciones (las 300 últimas)</h2><table><tr><th>Fecha</th><th>Estrategia</th><th>Tipo</th><th>Ticker</th><th>Nombre</th>
<th>Acciones</th><th>Precio (€)</th><th>Resultado</th><th>%</th><th>Motivo</th></tr>{''.join(f for _, f in ops[:300])}</table>
<h2>Estrategias</h2><ul>{desc}</ul></body></html>"""
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(doc)


def exportar_csv(estados, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    hist = {}
    for n, est in estados.items():
        if est["historial"]:
            h = pd.DataFrame(est["historial"], columns=["fecha", n]).set_index("fecha")
            hist[n] = h[n]
    if hist:
        pd.DataFrame(hist).sort_index().to_csv(os.path.join(carpeta, "historico.csv"))
    ops = [dict(estrategia=n, **o) for n, est in estados.items() for o in est["operaciones"]]
    if ops:
        pd.DataFrame(ops).sort_values("fecha").to_csv(os.path.join(carpeta, "operaciones.csv"), index=False)
