"""Configuración del simulador. Todo lo que se puede ajustar sin tocar el código está aquí."""

# Dinero ficticio inicial de cada estrategia, en euros
CAPITAL_INICIAL = 10_000
# Excepciones: los pares necesitan más capital para que la comisión mínima no se coma el resultado
CAPITAL_ESTRATEGIA = {
    "pares_largo_corto": 50_000,
    "pares_solo_largo": 50_000,
}

# Comisión por operación: porcentaje del importe con un mínimo en euros
COMISION_PCT = 0.001       # 0,1 %
COMISION_MINIMA = 2.0      # euros

# Solo se compran acciones enteras, como en un bróker normal
ACCIONES_ENTERAS = True

# Coste anual de tener acciones prestadas para ponerse en corto (se cobra a diario)
COSTE_CORTO_ANUAL = 0.01   # 1 %

# Años de historia que se descargan (hace falta para el método de "rebote" a precios de hace 10 años)
ANIOS_HISTORIA = 12

# ---------------------------------------------------------------------------
# Stop dinámico (estrategias de selección de acciones)
# Fase 1: stop al STOP_INICIAL por debajo del máximo alcanzado desde la compra.
# Cuando ese stop llega al precio de compra, pasa a fase 2:
# stop al STOP_FASE2 por debajo del máximo. El stop nunca baja.
# Ejemplo: compra a 1000 -> stop 900. Precio 1111,11 -> stop 1000 -> pasa a 1033,33.
STOP_INICIAL = 0.10
STOP_FASE2 = 0.07
# El máximo se actualiza con el precio de cierre de cada día
# Días hábiles sin volver a comprar una acción después de venderla
DIAS_ESPERA_RECOMPRA = 10

# Número máximo de posiciones abiertas por estrategia de selección
MAX_POSICIONES = 10

# ---------------------------------------------------------------------------
# Estrategia de pares
PARES = dict(
    max_pares=5,             # pares vigilados a la vez
    dias_formacion=504,      # unos 2 años de precios para buscar pares
    correlacion_minima=0.85, # filtro previo sobre log-precios
    pvalor_maximo=0.05,      # test de cointegración de Engle-Granger
    pvalor_mantener=0.10,    # en la revisión mensual, si se supera se cierra el par
    vida_media_min=5,        # días que tarda el diferencial en recorrer la mitad del camino a la media
    vida_media_max=60,
    ventana_z=60,            # días para calcular media y desviación del diferencial
    z_armado=2.0,            # la separación tiene que pasar de aquí...
    z_entrada=2.0,           # ...y luego volver por debajo de aquí para entrar
    z_entrada_min=0.75,      # si al volver ya está por debajo de esto, no se entra
    z_salida=0.5,            # se cierra cuando la separación baja de aquí
    z_stop=4.0,              # stop si la separación sigue creciendo
    multiplo_tiempo=2.0,     # stop por tiempo: vida media x este múltiplo (mínimo 10 días)
)

# ---------------------------------------------------------------------------
# Acciones pequeñas
PEQUENAS = dict(
    regiones=["es", "fr", "de", "it", "nl", "gb", "us", "ca"],
    capitalizacion_min=50e6,
    capitalizacion_max=2e9,
    anios_cotizando_min=2,    # se descartan las que llevan menos de 2 años en bolsa
    anios_cotizando_max=6,    # "salidas recientes": como mucho 6 años en bolsa
    precio_min_eur=1.0,       # fuera chicharros de céntimos
    volumen_min_eur=50_000,   # importe medio negociado al día (50 sesiones)
    max_por_region=150,
    dias_refresco=30,         # cada cuánto se vuelve a buscar la lista
)

# ---------------------------------------------------------------------------
# Métodos de selección
MAXIMO_52S = dict(factor_volumen=1.5)        # volumen del día > 1,5 x media de 50 sesiones
REBOTE = dict(caida_min=0.40, anios_atras=10) # caída desde máximo de 5 años y precio <= al de hace 10 años

# Índices de referencia (comprar y mantener con todo el capital el primer día)
REFERENCIAS = {
    "ref_mundo": ("IWDA.AS", "MSCI World (iShares, en euros)"),
    "ref_sp500": ("SPY", "S&P 500"),
    "ref_mundo_sin_eeuu": ("ACWX", "MSCI ACWI ex USA"),
    "ref_europa": ("EXSA.DE", "STOXX Europe 600 (iShares, en euros)"),
    "ref_bono_eeuu": ("IEF", "Bonos del Tesoro de EE. UU. 7-10 años"),
    "ref_bono_china": ("CBON", "Bonos de China (VanEck)"),
}

# ---------------------------------------------------------------------------
# Kronos: variantes de las estrategias de selección que ajustan el stop en fase 2 según la previsión del modelo.
# Si prevé una subida >= umbral_sube % en `horizonte` sesiones, el stop sube más despacio (distancia + margen_extra).
# Si prevé una bajada >= umbral_baja %, sube más deprisa (distancia - margen_menos). El stop nunca baja.
# Solo en la simulación diaria: en un backtest el modelo ya ha visto esos precios al entrenarse.
KRONOS = dict(
    activo=True,
    estrategias=["pequenas_azar", "maximo_52s", "momentum", "tendencia_200", "rebote", "baja_volatilidad", "azar_grandes"],
    carpeta_codigo="kronos_src",          # copia del repositorio de Kronos (la descarga GitHub Actions)
    modelo="NeoQuasar/Kronos-small",
    tokenizador="NeoQuasar/Kronos-Tokenizer-base",
    contexto_max=512,
    velas=256,          # velas diarias que recibe el modelo
    horizonte=10,       # sesiones que predice
    muestras=3,         # predicciones por acción (se hace la media)
    lote=32,
    umbral_sube=2.0,
    umbral_baja=2.0,
    margen_extra=0.03,
    margen_menos=0.02,
)

# Qué estrategias se ejecutan (se pueden comentar las que no interesen)
ESTRATEGIAS_ACTIVAS = [
    "pares_largo_corto",
    "pares_solo_largo",
    "pequenas_azar",
    "maximo_52s",
    "momentum",
    "tendencia_200",
    "rebote",
    "baja_volatilidad",
    "azar_grandes",
    *REFERENCIAS.keys(),
]
if KRONOS["activo"]:
    ESTRATEGIAS_ACTIVAS += [f"{n}_kronos" for n in KRONOS["estrategias"]]
