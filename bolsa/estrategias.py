"""Estrategias. Cada una decide al cierre qué órdenes se ejecutan en la apertura siguiente."""
import itertools
import random

import numpy as np
import pandas as pd

import config
from .datos import moneda
from .motor import en_espera, tickers_ocupados, valor_cartera


def primer_dia(m, d, periodo):
    """True si d es la primera sesión de la semana ('W') o del mes ('M') en el calendario."""
    i = m.fechas.get_loc(d)
    if i == 0:
        return True
    a = m.fechas[i - 1]
    if periodo == "W":
        return a.isocalendar()[:2] != d.isocalendar()[:2]
    return (a.year, a.month) != (d.year, d.month)


class Base:
    nombre = ""
    descripcion = ""
    usa_stop = False

    def __init__(self, universos):
        self.universos = universos

    def tickers_necesarios(self):
        return []

    def al_cierre(self, est, d, m):
        return []


# ---------------------------------------------------------------------------
# Comprar y mantener (índices de referencia)
class ComprarMantener(Base):
    def __init__(self, universos, clave, ticker, descripcion):
        super().__init__(universos)
        self.nombre, self.ticker = clave, ticker
        self.descripcion = f"Comprar y mantener {ticker}: {descripcion}"

    def tickers_necesarios(self):
        return [self.ticker]

    def al_cierre(self, est, d, m):
        if est["posiciones"] or est["pendientes"] or self.ticker not in m.close.columns:
            return []
        if np.isnan(m.close.at[d, self.ticker]):
            return []
        imp = est["efectivo"] * 0.99
        return [dict(accion="comprar", ticker=self.ticker, importe=imp, motivo="referencia")]


# ---------------------------------------------------------------------------
# Selección de acciones + stop dinámico
class Seleccion(Base):
    usa_stop = True
    frecuencia = "D"        # D: cada día, W: primera sesión de la semana
    universo = ("grandes",)
    sesiones_min = 260

    def tickers_necesarios(self):
        return sorted(set().union(*[set(self.universos[u]) for u in self.universo]))

    def elegibles(self, est, d, m):
        ocupados = tickers_ocupados(est)
        c = m.close.loc[d]
        n = m.ind("n_sesiones").loc[d]
        res = []
        for t in self.tickers_necesarios():
            if t not in m.close.columns or t in ocupados:
                continue
            if np.isnan(c.get(t, np.nan)) or n.get(t, 0) < self.sesiones_min:
                continue
            if en_espera(est, t, d, m):
                continue
            res.append(t)
        return res

    def candidatos(self, est, d, m, tickers):
        raise NotImplementedError

    def al_cierre(self, est, d, m):
        # las semanales rellenan huecos el primer día de la semana, o cualquier día si la cartera está vacía
        if self.frecuencia == "W" and not primer_dia(m, d, "W") and tickers_ocupados(est):
            return []
        libres = config.MAX_POSICIONES - len(tickers_ocupados(est))
        if libres <= 0:
            return []
        cands = self.candidatos(est, d, m, self.elegibles(est, d, m))
        if not cands:
            return []
        importe = valor_cartera(est, d, m) / config.MAX_POSICIONES
        disponible = est["efectivo"] - sum(o["importe"] for o in est["pendientes"] if o["accion"] == "comprar")
        valor = valor_cartera(est, d, m)
        ordenes = []
        for c in cands[:libres]:
            t, motivo = c[0], c[1]
            factor = c[2] if len(c) > 2 else 1.0  # tamaño relativo (estrategias de puntuación)
            if disponible < importe * 0.5:
                break
            imp = min(importe * factor, valor * config.PUNTUACION["max_posicion"], disponible)
            ordenes.append(dict(accion="comprar", ticker=t, importe=imp, motivo=motivo))
            disponible -= imp
        return ordenes


class AzarGrandes(Seleccion):
    nombre = "azar_grandes"
    descripcion = "Grupo de control: acciones grandes al azar, cada semana se rellenan los huecos"
    frecuencia = "W"

    def candidatos(self, est, d, m, tickers):
        rng = random.Random(f"{self.nombre}-{d.date()}")
        rng.shuffle(tickers)
        return [(t, "azar") for t in tickers]


class PequenasAzar(Seleccion):
    nombre = "pequenas_azar"
    descripcion = ("Acciones pequeñas al azar que llevan entre "
                   f"{config.PEQUENAS['anios_cotizando_min']} y {config.PEQUENAS['anios_cotizando_max']} años en bolsa")
    frecuencia = "W"
    universo = ("pequenas",)
    sesiones_min = 50

    def candidatos(self, est, d, m, tickers):
        cfg = config.PEQUENAS
        primera = self.universos.get("pequenas_primera", {})
        imp = m.ind("importe50").loc[d]
        ok = []
        for t in tickers:
            f0 = primera.get(t)
            if f0 is None:
                s = m.close[t].dropna()
                f0 = s.index[0]
            anios = (d - pd.Timestamp(f0)).days / 365.25
            if not (cfg["anios_cotizando_min"] <= anios <= cfg["anios_cotizando_max"]):
                continue
            if m.close.at[d, t] < cfg["precio_min_eur"] or imp.get(t, 0) < cfg["volumen_min_eur"]:
                continue
            ok.append(t)
        rng = random.Random(f"{self.nombre}-{d.date()}")
        rng.shuffle(ok)
        return [(t, "pequeña al azar") for t in ok]


class MomentumPequenas(PequenasAzar):
    nombre = "momentum_pequenas"
    descripcion = ("Acciones pequeñas (mismos filtros que pequenas_azar) que más han subido en 12 meses sin contar "
                   "el último (cada semana se rellenan huecos)")
    sesiones_min = 260

    def candidatos(self, est, d, m, tickers):
        validas = {t for t, _ in super().candidatos(est, d, m, tickers)}
        mom = m.ind("mom12_1").loc[d]
        res = sorted(((mom[t], t) for t in validas if mom.get(t, np.nan) > 0), reverse=True)
        return [(t, f"momentum 12-1: {100 * r:+.0f} %") for r, t in res]


class Puntuacion(Seleccion):
    """Puntuación = media de la posición en el ranking de cada señal (0 = la peor, 1 = la mejor)."""
    frecuencia = "W"

    def __init__(self, universos, tamano_variable):
        super().__init__(universos)
        self.tamano_variable = tamano_variable
        self.nombre = "puntuacion_tamano" if tamano_variable else "puntuacion_fija"
        cfg = config.PUNTUACION
        self.descripcion = ("Puntuación con momentum 12 meses y volatilidad (media de rankings); compra las mejores "
                            f"con puntuación > {cfg['minimo']}" +
                            (f"; tamaño de x{cfg['tam_min']} a x{cfg['tam_max']} según la puntuación, máximo "
                             f"{int(100 * cfg['max_posicion'])} % por posición" if tamano_variable else "; tamaño fijo"))

    def candidatos(self, est, d, m, tickers):
        cfg = config.PUNTUACION
        todos = [t for t in self.tickers_necesarios() if t in m.close.columns]
        rangos = []
        for senal in cfg["senales"]:
            v = m.ind(senal).loc[d, todos]
            rangos.append(v.rank(pct=True))
        punt = sum(rangos) / len(rangos)
        punt = punt.reindex(tickers).dropna()
        punt = punt[punt > cfg["minimo"]].sort_values(ascending=False)
        res = []
        for t, p in punt.items():
            if self.tamano_variable:
                f = cfg["tam_min"] + (cfg["tam_max"] - cfg["tam_min"]) * (p - cfg["minimo"]) / (1 - cfg["minimo"])
            else:
                f = 1.0
            res.append((t, f"puntuación {p:.2f}, tamaño x{f:.2f}", f))
        return res


class Maximo52s(Seleccion):
    nombre = "maximo_52s"
    descripcion = "Cierre por encima del máximo de 52 semanas con volumen > 1,5 x la media de 50 sesiones"
    universo = ("grandes", "pequenas")

    def candidatos(self, est, d, m, tickers):
        c = m.close.loc[d]
        mx = m.ind("max252_prev").loc[d]
        v = m.volume.loc[d]
        v50 = m.ind("vol50").loc[d]
        imp = m.ind("importe50").loc[d]
        res = []
        for t in tickers:
            if np.isnan(mx.get(t, np.nan)) or not v50.get(t, 0) > 0:
                continue
            if t in self.universos["pequenas"] and imp.get(t, 0) < config.PEQUENAS["volumen_min_eur"]:
                continue
            ratio = v.get(t, 0) / v50[t]
            if c[t] > mx[t] and ratio > config.MAXIMO_52S["factor_volumen"]:
                res.append((ratio, t))
        res.sort(reverse=True)
        return [(t, f"máximo 52 semanas, volumen x{r:.1f}") for r, t in res]


class Momentum(Seleccion):
    nombre = "momentum"
    descripcion = "Las que más han subido en 12 meses sin contar el último (cada semana se rellenan huecos)"
    frecuencia = "W"

    def candidatos(self, est, d, m, tickers):
        mom = m.ind("mom12_1").loc[d]
        res = sorted(((mom[t], t) for t in tickers if mom.get(t, np.nan) > 0), reverse=True)
        return [(t, f"momentum 12-1: {100 * r:+.0f} %") for r, t in res]


class Tendencia200(Seleccion):
    nombre = "tendencia_200"
    descripcion = "El precio cruza por encima de su media de 200 sesiones"

    def candidatos(self, est, d, m, tickers):
        c, s = m.close.loc[d], m.ind("sma200").loc[d]
        cp, sp = m.ind("close_prev").loc[d], m.ind("sma200_prev").loc[d]
        mom = m.ind("mom12_1").loc[d]
        res = []
        for t in tickers:
            if np.isnan(s.get(t, np.nan)) or np.isnan(sp.get(t, np.nan)):
                continue
            if cp[t] <= sp[t] and c[t] > s[t]:
                res.append((mom.get(t, 0) if not np.isnan(mom.get(t, np.nan)) else 0, t))
        res.sort(reverse=True)
        return [(t, "cruza media 200") for _, t in res]


class Rebote(Seleccion):
    nombre = "rebote"
    descripcion = (f"Caída > {int(100 * config.REBOTE['caida_min'])} % desde el máximo de 5 años, precio <= al de hace "
                   f"{config.REBOTE['anios_atras']} años y cruce al alza de la media de 50 sesiones")

    def candidatos(self, est, d, m, tickers):
        c, mx = m.close.loc[d], m.ind("max5a").loc[d]
        antes = m.ind(f"hace_anios_{config.REBOTE['anios_atras']}").loc[d]
        s50, cp, sp = m.ind("sma50").loc[d], m.ind("close_prev").loc[d], m.ind("sma50_prev").loc[d]
        res = []
        for t in tickers:
            vals = [mx.get(t), antes.get(t), s50.get(t), sp.get(t), cp.get(t)]
            if any(v is None or np.isnan(v) for v in vals):
                continue
            caida = 1 - c[t] / mx[t]
            if caida >= config.REBOTE["caida_min"] and c[t] <= antes[t] and cp[t] <= sp[t] and c[t] > s50[t]:
                res.append((caida, t))
        res.sort(reverse=True)
        return [(t, f"rebote, caída {100 * cd:.0f} % desde máximo 5 años") for cd, t in res]


class BajaVolatilidad(Seleccion):
    nombre = "baja_volatilidad"
    descripcion = "Las que menos se han movido en el último año (cada semana se rellenan huecos)"
    frecuencia = "W"

    def candidatos(self, est, d, m, tickers):
        vol = m.ind("volat252").loc[d]
        res = sorted((vol[t], t) for t in tickers if not np.isnan(vol.get(t, np.nan)))
        return [(t, f"volatilidad {100 * v * np.sqrt(252):.0f} % anual") for v, t in res]


# ---------------------------------------------------------------------------
# Pares
_CACHE_PARES = {}

def _analizar_par(la, lb):
    from statsmodels.tsa.stattools import coint
    x = np.column_stack([np.ones(len(lb)), lb])
    beta = np.linalg.lstsq(x, la, rcond=None)[0][1]
    s = la - beta * lb
    ds, s1 = np.diff(s), s[:-1] - s[:-1].mean()
    lam = np.dot(s1, ds) / np.dot(s1, s1)
    vida = -np.log(2) / lam if lam < 0 else np.inf
    p = coint(la, lb)[1]
    return beta, vida, p


class Pares(Base):
    def __init__(self, universos, solo_largo):
        super().__init__(universos)
        self.solo_largo = solo_largo
        self.nombre = "pares_solo_largo" if solo_largo else "pares_largo_corto"
        self.descripcion = ("Pares cointegrados del mismo sector y moneda. "
                            + ("Solo se compra la que se ha quedado atrás." if solo_largo
                               else "Largo en la que se ha quedado atrás y corto en la otra."))

    def tickers_necesarios(self):
        return list(self.universos["grandes"])

    def _logs(self, m, d, ta, tb, n):
        df = m.close.loc[:d, [ta, tb]].dropna().iloc[-n:]
        return np.log(df[ta].values), np.log(df[tb].values)

    def _candidatos(self, m, d):
        """Todos los pares cointegrados en la fecha d (se guardan para no repetir el cálculo)."""
        cfg = config.PARES
        clave = (id(m), d, cfg["dias_formacion"], cfg["correlacion_minima"], cfg["pvalor_maximo"])
        if clave in _CACHE_PARES:
            return _CACHE_PARES[clave]
        n = cfg["dias_formacion"]
        hist = m.close.loc[:d].iloc[-n:]
        sectores = self.universos["sectores"]
        validos = [t for t in self.tickers_necesarios() if t in hist.columns and hist[t].notna().mean() > 0.9]
        grupos = {}
        for t in validos:
            grupos.setdefault((sectores.get(t), moneda(t)), []).append(t)
        cands = []
        for _, ts in grupos.items():
            for a, b in itertools.combinations(sorted(ts), 2):
                df = np.log(hist[[a, b]].dropna())
                if len(df) < n * 0.8 or df[a].corr(df[b]) < cfg["correlacion_minima"]:
                    continue
                beta, vida, p = _analizar_par(df[a].values, df[b].values)
                if p < cfg["pvalor_maximo"]:
                    cands.append((p, a, b, beta, vida))
        cands.sort()
        _CACHE_PARES[clave] = cands
        return cands

    def buscar(self, m, d, excluir, cuantos):
        cfg = config.PARES
        cands = [c for c in self._candidatos(m, d) if cfg["vida_media_min"] <= c[4] <= cfg["vida_media_max"]]
        elegidos, usados = [], set(excluir)
        for p, a, b, beta, vida in cands:
            if a in usados or b in usados:
                continue
            elegidos.append(dict(a=a, b=b, beta=float(beta), vida=float(vida), p=float(p), armado=0, abierto=None))
            usados |= {a, b}
            if len(elegidos) >= cuantos:
                break
        return elegidos

    def _ordenes_cierre(self, est, par, motivo):
        g = f"{par['a']}|{par['b']}"
        est["pendientes"] = [o for o in est["pendientes"] if o.get("grupo") != g]
        res = []
        for t, p in est["posiciones"].items():
            if p.get("grupo") == g:
                res.append(dict(accion="vender" if p["acciones"] > 0 else "cerrar_corto", ticker=t, motivo=motivo))
        return res

    def al_cierre(self, est, d, m):
        cfg = config.PARES
        ex = est["extra"]
        pares = ex.setdefault("pares", [])
        ordenes = []
        # revisión mensual
        if primer_dia(m, d, "M") or not pares:
            seguir = []
            for par in pares:
                la, lb = self._logs(m, d, par["a"], par["b"], cfg["dias_formacion"])
                _, _, p = _analizar_par(la, lb) if len(la) > 100 else (0, 0, 1)
                if p > cfg["pvalor_mantener"]:
                    if par["abierto"]:
                        ordenes += self._ordenes_cierre(est, par, f"par deja de estar cointegrado (p={p:.2f})")
                else:
                    par["p"] = float(p)
                    seguir.append(par)
            pares[:] = seguir
            if len(pares) < cfg["max_pares"]:
                usados = {t for par in pares for t in (par["a"], par["b"])}
                pares += self.buscar(m, d, usados, cfg["max_pares"] - len(pares))
            ex["ultima_revision"] = str(d.date())

        hueco = valor_cartera(est, d, m) / cfg["max_pares"]
        for par in pares:
            a, b = par["a"], par["b"]
            if np.isnan(m.close.at[d, a]) or np.isnan(m.close.at[d, b]):
                continue
            la, lb = self._logs(m, d, a, b, cfg["ventana_z"])
            s = la - par["beta"] * lb
            if len(s) < 20 or s.std() == 0:
                continue
            z = float((s[-1] - s.mean()) / s.std())
            par["z"] = round(z, 2)
            g = f"{a}|{b}"
            ab = par["abierto"]
            if ab:
                ab["dias"] += 1
                limite = max(10, cfg["multiplo_tiempo"] * par["vida"])
                motivo = None
                if (ab["dir"] > 0 and z < cfg["z_salida"]) or (ab["dir"] < 0 and z > -cfg["z_salida"]):
                    motivo = f"se juntan (z={z:.2f})"
                elif abs(z) > cfg["z_stop"]:
                    motivo = f"stop por separación (z={z:.2f})"
                elif ab["dias"] > limite:
                    motivo = f"stop por tiempo ({ab['dias']} días)"
                if motivo:
                    ordenes += self._ordenes_cierre(est, par, motivo)
                    par["abierto"] = None
                    par["armado"] = 0
                continue
            # armado: la separación ha pasado del umbral
            if abs(z) >= cfg["z_armado"]:
                par["armado"] = 1 if z > 0 else -1
                continue
            if par["armado"] and abs(z) < cfg["z_entrada_min"]:
                par["armado"] = 0  # volvió sin dar tiempo a entrar
                continue
            if par["armado"] and abs(z) < cfg["z_entrada"] and np.sign(z) == par["armado"]:
                direc = par["armado"]
                cara, barata = (a, b) if direc > 0 else (b, a)
                motivo = f"diferencial vuelve (z={z:.2f})"
                ordenes.append(dict(accion="comprar", ticker=barata, importe=hueco / 2, motivo=motivo, grupo=g))
                if not self.solo_largo:
                    ordenes.append(dict(accion="abrir_corto", ticker=cara, importe=hueco / 2, motivo=motivo, grupo=g))
                par["abierto"] = dict(dir=int(direc), fecha=str(d.date()), dias=0)
                par["armado"] = 0
        return ordenes


# ---------------------------------------------------------------------------
# Selección con Kronos: compra las que el modelo prevé que más van a subir
class KronosPrevision(Seleccion):
    nombre = "kronos_prevision"
    frecuencia = "W"
    solo_diario = True

    def __init__(self, universos):
        super().__init__(universos)
        h = config.KRONOS["horizonte"]
        self.descripcion = (f"Cada semana, Kronos predice las {h} sesiones siguientes de todas las acciones grandes y se compran "
                            f"las de mayor subida prevista (al menos {config.KRONOS['seleccion_minimo']} %). Stop normal")

    def candidatos(self, est, d, m, tickers):
        from . import kronos_ia
        prev = kronos_ia.prever(m, tickers, d)
        res = sorted(((p, t) for t, p in prev.items() if p is not None and p >= config.KRONOS["seleccion_minimo"]),
                     reverse=True)
        return [(t, f"Kronos prevé {p:+.1f} % en {config.KRONOS['horizonte']} sesiones") for p, t in res]


# ---------------------------------------------------------------------------
# Variante con Kronos: compra igual que la estrategia base; solo cambia cómo sube el stop en fase 2
class ConKronos(Base):
    usa_stop = True
    solo_diario = True

    def __init__(self, universos, base):
        super().__init__(universos)
        self.base = base
        self.nombre = f"{base.nombre}_kronos"
        self.descripcion = (base.descripcion + ". Stop en fase 2 ajustado con la previsión de Kronos")

    def tickers_necesarios(self):
        return self.base.tickers_necesarios()

    def al_cierre(self, est, d, m):
        return self.base.al_cierre(est, d, m)

    def distancias_fase2(self, est, d, m):
        from . import kronos_ia
        fase2 = [t for t, p in est["posiciones"].items() if p.get("fase") == 2 and not np.isnan(m.close.at[d, t])]
        if not fase2:
            return {}
        prev = kronos_ia.prever(m, fase2, d)
        res = {}
        for t in fase2:
            dist = kronos_ia.distancia_fase2(prev[t])
            p = est["posiciones"][t]
            p["prevision_kronos"] = None if prev[t] is None else round(prev[t], 2)
            p["distancia_stop"] = round(dist, 3)
            res[t] = dist
        return res


# ---------------------------------------------------------------------------
def crear_estrategias(universos):
    disponibles = {
        "pares_largo_corto": lambda: Pares(universos, solo_largo=False),
        "pares_solo_largo": lambda: Pares(universos, solo_largo=True),
        "pequenas_azar": lambda: PequenasAzar(universos),
        "momentum_pequenas": lambda: MomentumPequenas(universos),
        "maximo_52s": lambda: Maximo52s(universos),
        "kronos_prevision": lambda: KronosPrevision(universos),
        "puntuacion_fija": lambda: Puntuacion(universos, tamano_variable=False),
        "puntuacion_tamano": lambda: Puntuacion(universos, tamano_variable=True),
        "momentum": lambda: Momentum(universos),
        "tendencia_200": lambda: Tendencia200(universos),
        "rebote": lambda: Rebote(universos),
        "baja_volatilidad": lambda: BajaVolatilidad(universos),
        "azar_grandes": lambda: AzarGrandes(universos),
    }
    for clave, (ticker, desc) in config.REFERENCIAS.items():
        disponibles[clave] = (lambda c=clave, t=ticker, de=desc: ComprarMantener(universos, c, t, de))
    for n in list(disponibles):
        disponibles[f"{n}_kronos"] = (lambda b=n: ConKronos(universos, disponibles[b]()))
    return [disponibles[n]() for n in config.ESTRATEGIAS_ACTIVAS if n in disponibles]
