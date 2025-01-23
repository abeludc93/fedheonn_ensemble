#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# Cargar datos de Excel
file_path = 'parallel_experiments.xlsx'
df = pd.read_excel(file_path)

line_styles = {True: '-', False: ':'}

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

for parallel_value in [True, False]:

    df_filtered = df[df['PARALLEL'] == parallel_value]
    grouped = df_filtered.groupby(['P_SAMPLES', 'P_FEATS'])
    for (p_samples, p_feats), group in grouped:
        axes[0].plot(group['N_ESTIMATORS'], group['T_FIT'],
                     linestyle=line_styles[parallel_value],
                     color='blue' if p_samples == 1 else 'red',
                     marker='o')
        axes[1].plot(group['N_ESTIMATORS'], group['T_AGG'],
                     linestyle=line_styles[parallel_value],
                     color='blue' if p_samples == 1 else 'red',
                     marker='o')
        axes[2].plot(group['N_ESTIMATORS'], group['T_CALC'],
                     linestyle=line_styles[parallel_value],
                     color='blue' if p_samples == 1 else 'red',
                     marker='o')

# Crear leyendas separadas
# Leyenda para los tipos de línea (PARALLEL True/False)
line_cont = mlines.Line2D([], [], color='black', linestyle='-', label='Parallel')
line_dashed = mlines.Line2D([], [], color='black', linestyle=':', label='Series')

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

# Graficar en cada subplot


axes[0].set_title(r'Client fit times')
axes[0].set_xlabel(r'No. of estimators ($T$)')
axes[0].set_ylabel(r'$T_{FIT}\quad(s)$')
axes[0].grid(True)

axes[1].set_title(r'Coord. data aggregation times')
axes[1].set_xlabel(r'No. of estimators ($T$)')
axes[1].set_ylabel(r'$T_{AGG}\quad(s)$')
axes[1].grid(True)

axes[2].set_title(r'Coord. opt. weights computation times')
axes[2].set_xlabel(r'No. of estimators ($T$)')
axes[2].set_ylabel(r'$T_{CALC}\quad(s)$')
axes[2].grid(True)

axes[0].legend(handles, labels, loc='best')

plt.suptitle("FedHEONN-ensemble")
plt.tight_layout()
plt.show()
