# Simulador de estrategias de bolsa

Simula con dinero ficticio varias estrategias a la vez, cada una con su propia cartera de 10.000 € (50.000 € las de pares, para que la comisión mínima no se coma el resultado), y las compara con índices de referencia. Todo se calcula en euros con el tipo de cambio de cada día, con comisiones (0,1 %, mínimo 2 €) y solo con acciones enteras.

Cada día, después del cierre, el programa descarga los precios de Yahoo Finance, ejecuta las órdenes decididas el día anterior al precio de apertura, comprueba los stops con el mínimo del día y decide las órdenes del día siguiente. Guarda el estado en `estado/estado.json` y genera el informe en `docs/index.html`.

## Estrategias

| Nombre | Qué hace |
|---|---|
| pares_largo_corto | Busca pares de acciones del mismo sector y moneda que se mueven juntas (test de cointegración de Engle-Granger sobre 2 años). Cuando se separan más de 2,5 desviaciones típicas espera a que empiecen a volver por debajo de 2 y entonces compra la que se ha quedado atrás y se pone corto en la otra. Cierra cuando la separación vuelve a la media, si sigue creciendo hasta 4 (stop) o si tarda el triple de lo normal en juntarse (stop por tiempo). Revisa los pares cada mes. |
| pares_solo_largo | Lo mismo, pero solo compra la que se ha quedado atrás, sin corto. |
| pequenas_azar | Acciones pequeñas (50 a 2.000 millones de capitalización) que llevan entre 2 y 6 años en bolsa, al azar. Fuera las de menos de 1 € y las que negocian menos de 50.000 € al día. |
| puntuacion_fija | Cada semana puntúa las acciones grandes con la media de su posición en el ranking de momentum de 12 meses y en el de volatilidad (las dos señales que se confirmaron en 2015-2021 y en 2022-2026). Compra las de mayor puntuación, si pasa de 0,5, todas con el mismo tamaño. |
| puntuacion_tamano | La misma selección, con un tamaño que va de la mitad de lo normal (puntuación 0,5) al doble (puntuación 1), con un máximo del 20 % de la cartera por posición. |
| maximo_52s | Cierre por encima del máximo de 52 semanas con volumen mayor que 1,5 veces la media de 50 sesiones. |
| momentum | Las que más han subido en 12 meses sin contar el último. |
| tendencia_200 | El precio cruza por encima de su media de 200 sesiones. |
| rebote | Caída de más del 40 % desde el máximo de 5 años, precio igual o inferior al de hace 10 años y cruce al alza de la media de 50 sesiones. |
| baja_volatilidad | Las que menos se han movido en el último año. |
| azar_grandes | Acciones grandes al azar. Es el grupo de control. |
| ref_* | Comprar y mantener: MSCI World, S&P 500, MSCI ACWI sin EE. UU., STOXX Europe 600, bonos de EE. UU. 7-10 años y bonos de China. |

Todas las estrategias de selección (de pequenas_azar a azar_grandes) usan el mismo stop dinámico y tienen como máximo 10 posiciones de igual tamaño. Las que se basan en una clasificación (momentum, baja volatilidad y las dos de azar) rellenan los huecos el primer día de cada semana; las que se basan en una señal lo miran cada día. Después de vender una acción no se vuelve a comprar en 10 sesiones.

### El stop dinámico

Al comprar, el stop queda un 20 % por debajo del precio máximo alcanzado desde la compra. Cuando ese stop llega al precio de compra, pasa a estar un 10 % por debajo del máximo. El stop nunca baja. El máximo se actualiza con el precio de cierre. Los porcentajes salen de la optimización (al principio eran 10 % y 7 %).

Ejemplo con compra a 1000: stop en 800. Con la acción a 1250 el stop llega a 1000 y pasa a 1125. Si después baja a 1200, el stop sigue en 1125.

Las variantes `_kronos` compran lo mismo que su estrategia base, pero en fase 2 consultan cada día a Kronos (modelo de IA para velas de bolsa). Si prevé una subida de al menos un 2 % en 10 sesiones, el stop pasa a seguir el máximo a un 13 % en lugar del 10 %; si prevé una bajada de al menos un 2 %, a un 8 %. El stop nunca baja: solo cambia la velocidad a la que sube. Solo funcionan en la simulación diaria, porque en un backtest el modelo ya habría visto esos precios al entrenarse.

Si el mínimo de un día toca el stop, se vende al precio del stop, o al de apertura si la acción abre ya por debajo.

Los porcentajes, el capital, las comisiones y el resto de parámetros están en `config.py`.

## Las listas de acciones

- `universos/grandes.csv`: 239 empresas grandes de Europa, EE. UU., Canadá, Japón, Hong Kong, Australia, Taiwán, Corea e India, con su sector (el sector se usa para buscar pares). Se puede editar: añadir o quitar filas.
- `universos/pequenas.csv`: se genera solo con el buscador de acciones de Yahoo Finance (regiones en `config.py`) y se renueva cada 30 días. Para generarla a mano: `python ejecutar.py universo`.

## Puesta en marcha en GitHub

1. Crea una cuenta en github.com.
2. Crea un repositorio nuevo (botón "New"). Ponle un nombre, por ejemplo `bolsa`.
3. Sube todos los archivos de esta carpeta: en el repositorio, "Add file" > "Upload files" y arrastra el contenido de la carpeta (las subcarpetas `bolsa` y `universos` incluidas). También puedes usar GitHub Desktop.
   Comprueba después que existe `.github/workflows/diario.yml`: es el archivo que programa la ejecución diaria.
4. En el repositorio, ve a Settings > Actions > General y, en "Workflow permissions", marca "Read and write permissions". Así el programa puede guardar el estado cada día.
5. Ve a la pestaña Actions, elige "Simulacion diaria" y pulsa "Run workflow" para lanzarlo la primera vez. El primer día solo decide qué comprar; las compras se hacen en la apertura siguiente.

A partir de ahí se ejecuta solo de lunes a viernes a las 22:30 UTC (23:30 en España en invierno). Cada ejecución procesa todos los días pendientes, así que si un día falla, al siguiente se pone al día.

### Ver el informe

- Repositorio público: en Settings > Pages, elige "Deploy from a branch", rama `main` y carpeta `/docs`. El informe queda en `https://TU_USUARIO.github.io/bolsa/`.
- Repositorio privado: GitHub Pages en repositorios privados es de pago. Descarga `docs/index.html` y ábrelo en el navegador.

`estado/historico.csv` tiene el valor diario de cada estrategia y `estado/operaciones.csv` todas las operaciones, para abrirlos en Excel.

## Backtest en GitHub

En la pestaña Actions, elige "Backtest", pulsa "Run workflow", pon la fecha de inicio y vuelve a pulsar "Run workflow". Tarda unos minutos. El resultado queda en `https://TU_USUARIO.github.io/bolsa/backtest/`. Cada backtest sustituye al anterior.

## Análisis de los stops

El informe diario y el del backtest tienen una sección "Ventas por stop: qué pasó después". Para cada venta por stop muestra cómo se movió el precio 5, 20 y 60 sesiones después, y si la acción llegó a superar el máximo que había alcanzado mientras se tenía. Si la mayoría de las ventas por stop van seguidas de subidas, el stop está demasiado ajustado. El backtest deja además la lista completa en `stops.csv`.

## Optimización de parámetros

En Actions, "Optimizar parametros" prueba 24 combinaciones del stop (inicial del 6 al 20 %, fase 2 del 5 al 15 %) en las estrategias de selección y 16 combinaciones de los parámetros de los pares. Divide el histórico en dos periodos: en el primero se clasifican las combinaciones y en el segundo se comprueba si las buenas siguen siendo buenas. Tarda entre 30 y 60 minutos. El resultado queda en `https://TU_USUARIO.github.io/bolsa/optimizacion/`. Las combinaciones que se prueban están en `bolsa/analisis.py` (`REJILLA_STOPS` y `REJILLA_PARES`).

## Búsqueda de señales

En Actions, "Buscar senales" mide 13 señales (momentum, rupturas, medias, volumen, volatilidad, RSI...) sobre las acciones grandes: cuánto predice cada una la rentabilidad de las 20 y 60 sesiones siguientes, cuánto se parecen entre sí y qué aporta cada una descontando las demás. Por defecto usa 2015-2021 y no toca ningún precio posterior, para dejar 2022-2026 como periodo de comprobación. El resultado queda en `https://TU_USUARIO.github.io/bolsa/senales/`.

## Uso en tu ordenador

```
pip install -r requirements.txt
python ejecutar.py backtest --desde 2016-01-01
```

El backtest prueba las estrategias con datos históricos y deja el informe en `resultados_backtest/informe.html`. Por defecto no incluye las acciones pequeñas, porque la lista es la de hoy: solo contiene empresas que han sobrevivido, y eso infla los resultados. Con `--con-pequenas` se incluyen igualmente. La lista de grandes tiene el mismo problema en menor medida: son las grandes de hoy, no las de 2016.

No ejecutes `python ejecutar.py diario` en tu ordenador si ya lo hace GitHub, porque tendrías dos estados distintos.

## Limitaciones

- Yahoo Finance no tiene una API oficial. La librería yfinance a veces falla durante unos días hasta que la actualizan; el programa reintenta y se pone al día cuando vuelve a funcionar.
- Los cortos se simulan sin requisitos de garantías y con un coste fijo del 1 % anual. Los dividendos se cobran (y en los cortos se pagan) en efectivo.
- Las ejecuciones son al precio de apertura o de stop, sin deslizamiento.
- Los pares se cubren con el mismo importe en euros en cada pata, no con el coeficiente de la regresión.
- Si una empresa deja de cotizar (fusión, exclusión), su posición se queda valorada al último precio conocido.
