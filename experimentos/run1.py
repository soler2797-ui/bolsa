from lab import *
from multiprocessing import Pool
univ = universos.cargar(refrescar_pequenas=False)
m = preparar(cargar(), univ)
OFF = list(range(0, 60, 3))        # 20 fechas de inicio: 2016-01-01 + k sesiones
f0 = m.fechas[m.fechas >= '2016-01-01']
def variantes(k):
    return [Momentum(univ), Puntuacion(univ, False),
        PorIndicador(univ, 'resid_mercado', 'x_resid_m'), PorIndicador(univ, 'resid_sector', 'x_resid_s'),
        PorIndicador(univ, 'resid_mercado_sin_dividir', 'x_resid_m_bruto'), PorIndicador(univ, 'resid_sector_sin_dividir', 'x_resid_s_bruto'),
        PorIndicador(univ, 'mom_entre_volatilidad', 'x_mom_vol'),
        Amplitud(univ, 'amplitud_50'), Amplitud(univ, 'amplitud_50_control', desplaz=400 + 97 * k),
        Sector(univ, 'sector1_6m', 126, 1), Sector(univ, 'sector1_3m', 63, 1), Sector(univ, 'sector3_6m', 126, 3),
        Sector(univ, 'sector1_azar', 126, 1, azar=k), Sector(univ, 'sector3_azar', 126, 3, azar=k),
        Interseccion(univ, 'interseccion_max52')]
def tarea(a):
    k, j = a
    e = variantes(k)[j]
    h, est = correr(e, m, f0[k])
    r = medidas(h); r.update(nombre=e.nombre, k=k)
    r['e2018'] = 100 * (h[:'2018-12-31'].iloc[-1] / h[:'2018-05-01'].iloc[-1] - 1)
    r['e2022'] = 100 * (h[:'2022-09-30'].iloc[-1] / h[:'2022-04-01'].iloc[-1] - 1)
    r['inv'] = 100 * np.mean([1]) ; r['ops'] = len(est['operaciones'])
    return r, (e.nombre, k, h)
if __name__ == '__main__':
    n = len(variantes(0))
    with Pool(2) as p:
        out = p.map(tarea, [(k, j) for k in OFF for j in range(n)], chunksize=4)
    df = pd.DataFrame([o[0] for o in out]); df.to_csv('run1.csv', index=False)
    pickle.dump({(a, b): c for _, (a, b, c) in out}, open('run1_curvas.pkl', 'wb'))
