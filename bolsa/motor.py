"""Cartera ficticia, ejecución de órdenes, stop dinámico y bucle diario.

Orden de cada día d:
 1. splits y dividendos del día
 2. coste diario de las posiciones cortas
 3. se ejecutan las órdenes pendientes al precio de apertura de d
 4. stops: si el mínimo del día toca el stop, se vende al stop (o a la apertura si abre por debajo)
 5. al cierre: se sube el stop con el cierre y la estrategia decide órdenes para mañana
 6. se apunta el valor de la cartera
"""
import math

import numpy as np

import config


def nuevo_estado(nombre, capital=None):
    capital = capital or config.CAPITAL_ESTRATEGIA.get(nombre, config.CAPITAL_INICIAL)
    return dict(nombre=nombre, efectivo=float(capital), capital_inicial=float(capital),
                posiciones={}, pendientes=[], extra={}, vendidas={},
                historial=[], operaciones=[], ultima_fecha=None)


def comision(importe):
    return max(config.COMISION_MINIMA, abs(importe) * config.COMISION_PCT)


def _f(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def valor_cartera(est, d, m):
    v = est["efectivo"]
    for t, p in est["posiciones"].items():
        px = m.close_ffill.at[d, t] if t in m.close_ffill.columns else np.nan
        if np.isnan(px):
            px = p["precio_entrada"]
        v += p["acciones"] * px
    return v


def _apuntar(est, d, tipo, t, acciones, precio, com, motivo, grupo=None):
    op = dict(fecha=str(d.date()), tipo=tipo, ticker=t, acciones=acciones,
              precio=round(precio, 4), comision=round(com, 2), motivo=motivo)
    if grupo:
        op["grupo"] = grupo
    est["operaciones"].append(op)
    return op


def _cerrar(est, d, t, precio, motivo):
    p = est["posiciones"].pop(t)
    n = p["acciones"]
    importe = abs(n) * precio
    com = comision(importe)
    if n > 0:
        est["efectivo"] += importe - com
        tipo = "venta"
        resultado = (precio - p["precio_entrada"]) * n - com - p.get("comision_entrada", 0)
    else:
        est["efectivo"] -= importe + com
        tipo = "cierre_corto"
        resultado = (p["precio_entrada"] - precio) * abs(n) - com - p.get("comision_entrada", 0)
    op = _apuntar(est, d, tipo, t, abs(n), precio, com, motivo, p.get("grupo"))
    op["resultado"] = round(resultado + p.get("dividendos", 0), 2)
    op["rentabilidad_pct"] = round(100 * (precio / p["precio_entrada"] - 1) * (1 if n > 0 else -1), 2)
    op["fecha_entrada"] = p["fecha_entrada"]
    est["vendidas"][t] = str(d.date())


def _reducir(est, d, t, fraccion, precio, motivo):
    """Venta parcial de una posición larga: vende `fraccion` de las acciones (enteras si se usan enteras).
    El resto sigue con el mismo stop, máximo y fase. Si no queda nada, se cierra la posición."""
    p = est["posiciones"][t]
    if p["acciones"] <= 0:
        return False
    n = p["acciones"] * fraccion
    if config.ACCIONES_ENTERAS:
        n = math.floor(n)
    if n <= 0:
        return False
    if n >= p["acciones"] - 1e-9:
        _cerrar(est, d, t, precio, motivo)
        return True
    parte = n / p["acciones"]
    importe = n * precio
    com = comision(importe)
    est["efectivo"] += importe - com
    com_ent = p.get("comision_entrada", 0) * parte
    div = p.get("dividendos", 0) * parte
    op = _apuntar(est, d, "venta_parcial", t, n, precio, com, motivo, p.get("grupo"))
    op["resultado"] = round((precio - p["precio_entrada"]) * n - com - com_ent + div, 2)
    op["rentabilidad_pct"] = round(100 * (precio / p["precio_entrada"] - 1), 2)
    op["fecha_entrada"] = p["fecha_entrada"]
    p["acciones"] -= n
    p["comision_entrada"] = p.get("comision_entrada", 0) - com_ent
    p["dividendos"] = p.get("dividendos", 0) - div
    p["reducida"] = p.get("reducida", 0) + n
    return True


def _ampliar(est, d, t, acciones, precio, motivo):
    """Compra `acciones` más de una posición larga abierta. El precio de entrada pasa a ser el medio ponderado;
    stop, máximo y fase no cambian."""
    p = est["posiciones"][t]
    n = acciones
    if config.ACCIONES_ENTERAS:
        n = math.floor(n)
    imp = n * precio
    com = comision(imp)
    if n <= 0 or imp + com > est["efectivo"] + 1e-9:
        n = (est["efectivo"] - com) / precio
        if config.ACCIONES_ENTERAS:
            n = math.floor(n)
        imp = n * precio
        com = comision(imp)
        if n <= 0 or imp + com > est["efectivo"] + 1e-9:
            return False
    est["efectivo"] -= imp + com
    total = p["acciones"] + n
    p["precio_entrada"] = (p["precio_entrada"] * p["acciones"] + precio * n) / total
    p["acciones"] = total
    p["comision_entrada"] = p.get("comision_entrada", 0) + com
    p["reducida"] = max(0, p.get("reducida", 0) - n)
    _apuntar(est, d, "ampliacion", t, n, precio, com, motivo, p.get("grupo"))
    return True


def _abrir(est, d, o, precio, con_stop):
    t, importe = o["ticker"], o["importe"]
    importe = min(importe, max(0.0, est["efectivo"] - comision(importe))) if o["accion"] == "comprar" else importe
    n = importe / precio
    if config.ACCIONES_ENTERAS:
        n = math.floor(n)
    if n <= 0:
        return False
    imp = n * precio
    com = comision(imp)
    if o["accion"] == "comprar":
        if imp + com > est["efectivo"] + 1e-9:
            return False
        est["efectivo"] -= imp + com
        acciones = n
        tipo = "compra"
    else:  # abrir_corto
        est["efectivo"] += imp - com
        acciones = -n
        tipo = "corto"
    pos = dict(acciones=acciones, precio_entrada=precio, fecha_entrada=str(d.date()),
               comision_entrada=com, dividendos=0.0)
    if o.get("grupo"):
        pos["grupo"] = o["grupo"]
    if con_stop and acciones > 0:
        d1 = o.get("stop_inicial", config.STOP_INICIAL)
        pos.update(maximo=precio, stop=precio * (1 - d1), fase=1)
        if d1 != config.STOP_INICIAL:
            pos["stop_inicial"] = d1  # stop más ajustado (momentum_semaforo, acciones con punto amarillo)
    est["posiciones"][t] = pos
    _apuntar(est, d, tipo, t, n, precio, com, o.get("motivo", ""), o.get("grupo"))
    return True


def actualizar_stop(pos, precio, distancia_fase2=None):
    """Sube el stop según el precio de cierre. Nunca baja."""
    d2 = distancia_fase2 if distancia_fase2 is not None else config.STOP_FASE2
    pos["maximo"] = max(pos["maximo"], precio)
    if pos["fase"] == 1:
        pos["stop"] = max(pos["stop"], pos["maximo"] * (1 - pos.get("stop_inicial", config.STOP_INICIAL)))
        if pos["stop"] >= pos["precio_entrada"] * (1 - 1e-5):
            pos["fase"] = 2
    if pos["fase"] == 2:
        pos["stop"] = max(pos["stop"], pos["maximo"] * (1 - d2))


def procesar_dia(est, estrategia, d, m, splits_antes_de=None):
    """splits_antes_de: Yahoo da el histórico ya ajustado por splits, así que solo hay que ajustar
    las posiciones guardadas en ejecuciones anteriores (abiertas antes de esa fecha). En backtest, None."""
    pos = est["posiciones"]
    # 1. splits y dividendos
    for t, p in list(pos.items()):
        if t not in m.close.columns:
            continue
        r = m.split.at[d, t]
        if (splits_antes_de is not None and p["fecha_entrada"] < splits_antes_de
                and r and r > 0 and r != 1):
            p["acciones"] *= r
            for k in ("precio_entrada", "stop", "maximo"):
                if k in p:
                    p[k] /= r
        dv = m.div.at[d, t]
        if dv and dv > 0:
            cobro = p["acciones"] * dv  # los cortos pagan el dividendo
            est["efectivo"] += cobro
            p["dividendos"] = p.get("dividendos", 0) + cobro
        # 2. coste de los cortos
        if p["acciones"] < 0:
            px = m.close_ffill.at[d, t]
            if not np.isnan(px):
                est["efectivo"] -= abs(p["acciones"]) * px * config.COSTE_CORTO_ANUAL / 252

    # 3. órdenes pendientes a la apertura
    quedan = []
    for o in est["pendientes"]:
        t = o["ticker"]
        px = m.open.at[d, t] if t in m.open.columns else np.nan
        if np.isnan(px) or px <= 0:
            o["intentos"] = o.get("intentos", 0) + 1
            if o["intentos"] < 5:
                quedan.append(o)
            continue
        if o["accion"] in ("comprar", "abrir_corto"):
            if t not in pos:
                _abrir(est, d, o, px, estrategia.usa_stop)
        elif o["accion"] == "reducir":
            if t in pos:
                _reducir(est, d, t, o["fraccion"], px, o.get("motivo", ""))
        elif o["accion"] == "ampliar":
            if t in pos:
                _ampliar(est, d, t, o["acciones"], px, o.get("motivo", ""))
        elif t in pos:
            _cerrar(est, d, t, px, o.get("motivo", ""))
    est["pendientes"] = quedan

    # 4. stops
    if estrategia.usa_stop:
        for t, p in list(pos.items()):
            if "stop" not in p:
                continue
            lo, op = m.low.at[d, t], m.open.at[d, t]
            if np.isnan(lo):
                continue
            if lo <= p["stop"]:
                precio = min(op, p["stop"]) if not np.isnan(op) else p["stop"]
                _cerrar(est, d, t, precio, f"stop fase {p['fase']}")
        # 5a. el stop sube con el cierre (las variantes con Kronos eligen la distancia en fase 2)
        distancias = estrategia.distancias_fase2(est, d, m) if hasattr(estrategia, "distancias_fase2") else {}
        for t, p in pos.items():
            if "stop" in p:
                c = m.close.at[d, t]
                if not np.isnan(c):
                    actualizar_stop(p, c, distancias.get(t))

    # 5b. decisiones para mañana
    nuevas = estrategia.al_cierre(est, d, m) or []
    est["pendientes"].extend(nuevas)

    # 6. valor
    est["historial"].append([str(d.date()), round(valor_cartera(est, d, m), 2)])
    est["ultima_fecha"] = str(d.date())


def tickers_ocupados(est):
    return set(est["posiciones"]) | {o["ticker"] for o in est["pendientes"]}


def en_espera(est, t, d, m):
    """True si la acción se vendió hace menos de DIAS_ESPERA_RECOMPRA sesiones."""
    f = est["vendidas"].get(t)
    if not f:
        return False
    import pandas as pd
    n = ((m.fechas > pd.Timestamp(f)) & (m.fechas <= d)).sum()
    return n < config.DIAS_ESPERA_RECOMPRA
