#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# Cargar datos de Excel
file_path = 'parallel_experiments.xlsx'
df = pd.read_excel(file_path)

line_styles = {True: '-', False: ':'}
plt.figure(figsize=(6, 5))


for parallel_value in [True, False]:

    df_filtered = df[df['PARALLEL'] == parallel_value]
    grouped = df_filtered.groupby(['P_SAMPLES', 'P_FEATS'])
    for (p_samples, p_feats), group in grouped:
        plt.plot(group['N_ESTIMATORS'], group['T_COORD'],
                 linestyle=line_styles[parallel_value],
                 color='blue' if p_samples == 1 else 'red',
                 marker='o')

# Crear leyendas separadas
# Leyenda para los tipos de línea (PARALLEL True/False)
line_cont = mlines.Line2D([], [], color='black', linestyle='-', label='En PARALELO')
line_dashed = mlines.Line2D([], [], color='black', linestyle=':', label='En SERIE')

# Leyenda para los colores de P_SAMPLES y P_FEATS
color_legend = [mlines.Line2D([], [], color='blue',
                              marker='o', linestyle='',
                              label=r'$\rho_s \times \rho_f = 1$'),
                mlines.Line2D([], [], color='red',
                              marker='o', linestyle='',
                              label=r'$\rho_s \times \rho_f = 1/4$')]

# Añadir las leyendas al gráfico
handles = [line_cont, line_dashed] + color_legend
labels = [h.get_label() for h in handles]
plt.legend(handles, labels, loc='best')

plt.title('Tiempo empleado por coordinador FedHEONN')
plt.xlabel(r'Estimadores base ($T$)')
plt.ylabel(r'$T_{COORD}\quad(s)$')
plt.grid(True)
plt.show()
