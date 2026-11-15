from lab import *
from multiprocessing import Pool
from bolsa.estrategias import Maximo52s, Tendencia200, Rebote, BajaVolatilidad, primer_dia
univ = universos.cargar(refrescar_pequenas=False)
univ['pequenas'], univ['pequenas_primera'] = [], {}   # como en el backtest
m = cargar()
f0 = m.fechas[m.fechas >= '2016-01-01']
def metodos():
    return {'momentum': Momentum(univ), 'puntuacion_fija': Puntuacion(univ, False), 'maximo_52s': Maximo52s(univ),
            'tendencia_200': Tendencia200(univ), 'rebote': Rebote(univ), 'baja_volatilidad': BajaVolatilidad(univ)}

class Rotacion(Momentum):
    """Rellena cada hueco con el método que mejor fue en los L meses anteriores. Las posiciones abiertas no se tocan."""
    frecuencia = 'D'
    def __init__(self, nombre, L, mens, azar=None):
        super().__init__(univ); self.nombre, self.L, self.mens, self.azar = nombre, L, mens, azar
        self.met = metodos(); self.eleccion = {}
    def elegido(self, d):
        mes = pd.Timestamp(d.year, d.month, 1)
        if mes not in self.eleccion:
            if self.azar is not None:
                e = random.Random(f'{self.azar}-{mes.date()}').choice(sorted(self.met))
            else:
                prev = self.mens[self.mens.index < mes]          # meses ya cerrados
                e = 'momentum' if len(prev) < self.L + 1 else (prev.iloc[-1] / prev.iloc[-1 - self.L] - 1).idxmax()
            self.eleccion[mes] = e
        return self.eleccion[mes]
    def candidatos(self, est, d, m, tickers):
        e = self.met[self.elegido(d)]
        if e.frecuencia == 'W' and not primer_dia(m, d, 'W') and tickers_ocupados(est):
            return []
        return e.candidatos(est, d, m, tickers)

def base(n):
    h, _ = correr(metodos()[n], m, f0[0]); return n, h
def tarea(a):
    k, j, mens = a
    e = [Momentum(univ), Rotacion('rotacion_1m', 1, mens), Rotacion('rotacion_3m', 3, mens),
         Rotacion('control_azar', 1, mens, azar=k)][j]
    h, est = correr(e, m, f0[k])
    r = medidas(h); r.update(nombre=e.nombre, k=k, ops=len(est['operaciones']))
    r['e2018'] = 100 * (h[:'2018-12-31'].iloc[-1] / h[:'2018-05-01'].iloc[-1] - 1)
    r['e2022'] = 100 * (h[:'2022-09-30'].iloc[-1] / h[:'2022-04-01'].iloc[-1] - 1)
    if j in (1, 2): r['meses'] = pd.Series(e.eleccion).value_counts().to_dict()
    return r
if __name__ == '__main__':
    with Pool(2) as p:
        curvas = dict(p.map(base, list(metodos())))
    mens = pd.DataFrame(curvas).resample('MS').first()   # valor al inicio de cada mes
    mens = pd.DataFrame(curvas).resample('ME').last(); mens.index = mens.index.to_period('M').to_timestamp()
    for n, h in curvas.items(): print(n, {k: round(v, 1) for k, v in medidas(h).items()})
    with Pool(2) as p:
        out = p.map(tarea, [(k, j, mens) for k in range(0, 60, 3) for j in range(4)], chunksize=2)
    df = pd.DataFrame(out); df.to_csv('run2.csv', index=False)
    print(df[df.k == 0][['nombre', 'meses']].to_string())
