"""Guarda una copia de cada backtest en docs/backtest/anteriores y mantiene la lista."""
import datetime as dt
import html
import json
import os
import shutil

LISTA = "lista.json"


def archivar(carpeta, desde, nota="", fecha=None, origen=None):
    """Copia el informe de `carpeta` (o el html `origen`) a carpeta/anteriores/<fecha>_desde_<desde>/."""
    fecha = fecha or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M")
    destino_rel = f"{fecha}_desde_{desde}"
    base = os.path.join(carpeta, "anteriores")
    destino = os.path.join(base, destino_rel)
    os.makedirs(destino, exist_ok=True)
    shutil.copy(origen or os.path.join(carpeta, "index.html"), os.path.join(destino, "index.html"))
    for f in ("historico.csv", "operaciones.csv", "stops.csv"):
        if origen is None and os.path.exists(os.path.join(carpeta, f)):
            shutil.copy(os.path.join(carpeta, f), os.path.join(destino, f))
    ruta_lista = os.path.join(base, LISTA)
    lista = json.load(open(ruta_lista, encoding="utf-8")) if os.path.exists(ruta_lista) else []
    lista = [x for x in lista if x["carpeta"] != destino_rel]
    lista.append(dict(carpeta=destino_rel, fecha=fecha, desde=desde, nota=nota))
    lista.sort(key=lambda x: x["fecha"], reverse=True)
    json.dump(lista, open(ruta_lista, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    filas = "".join(f"<tr><td>{html.escape(x['fecha'].replace('_', ' '))}</td><td>{html.escape(x['desde'])}</td>"
                    f"<td><a href='{x['carpeta']}/'>ver</a></td><td>{html.escape(x['nota'])}</td></tr>" for x in lista)
    doc = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Backtests anteriores</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222;max-width:1000px}}
table{{border-collapse:collapse;font-size:14px}}td,th{{border-bottom:1px solid #ddd;padding:6px 10px;text-align:left}}
th{{background:#f4f4f4}}</style></head><body><h1>Backtests anteriores</h1>
<p><a href="../">Último backtest</a>. Hora en UTC (en España, una hora más en invierno).</p>
<table><tr><th>Ejecutado</th><th>Desde</th><th>Informe</th><th>Notas</th></tr>{filas}</table></body></html>"""
    with open(os.path.join(base, "index.html"), "w", encoding="utf-8") as f:
        f.write(doc)
    return destino
