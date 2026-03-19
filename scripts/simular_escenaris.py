# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 20:00:43 2026

@author: tirki
"""

hydro_esc1_level = reconstruir_volum_escenari(hydro_base_level,escenaris_simulacio['Historic'],escenaris_simulacio['Sequera_Extensa'],max_capacity_int).dropna()
aport_hist_mensual = escenaris_simulacio['Historic'].resample('ME').sum()
aport_esc_mensual = escenaris_simulacio['Sequera_Extensa'].resample('ME').sum()
energia_turbinada_esc_int = (datos.energia_turbinada_mensual_internes * (aport_esc_mensual / aport_hist_mensual) ** 0.58).dropna()
f_ebre = 0.23+(1-0.23)*(aport_esc_mensual / aport_hist_mensual) ** 0.58
energia_turbinada_esc_ebre = (datos.energia_turbinada_mensual_ebre * f_ebre).dropna()


# hydro_esc2_level = reconstruir_volum_escenari(hydro_base_level,escenaris_simulacio['Historic'],escenaris_simulacio['Sequera_Anticipada'],max_capacity_int).dropna()
# aport_hist_mensual = escenaris_simulacio['Historic'].resample('ME').sum()
# aport_esc_mensual = escenaris_simulacio['Sequera_Anticipada'].resample('ME').sum()
# energia_turbinada_esc_int = (datos.energia_turbinada_mensual_internes * (aport_esc_mensual / aport_hist_mensual) ** 0.58).dropna()
# f_ebre = 0.23+(1-0.23)*(aport_esc_mensual / aport_hist_mensual) ** 0.58
# energia_turbinada_esc_ebre = (datos.energia_turbinada_mensual_ebre * f_ebre).dropna()



hydro_esc3_level = reconstruir_volum_escenari(hydro_base_level,escenaris_simulacio['Historic'],escenaris_simulacio['Clima_2050'],max_capacity_int).dropna()
aport_hist_mensual = escenaris_simulacio['Historic'].resample('ME').sum()
aport_esc_mensual = escenaris_simulacio['Clima_2050'].resample('ME').sum()
energia_turbinada_esc_int = (datos.energia_turbinada_mensual_internes * (aport_esc_mensual / aport_hist_mensual) ** 0.58).dropna()
f_ebre = 0.23+(1-0.23)*(aport_esc_mensual / aport_hist_mensual) ** 0.58
energia_turbinada_esc_ebre = (datos.energia_turbinada_mensual_ebre * f_ebre).dropna()


hydro_esc4_level = reconstruir_volum_escenari(hydro_base_level,escenaris_simulacio['Historic'],escenaris_simulacio['Torrencialitat'],max_capacity_int).dropna()
aport_hist_mensual = escenaris_simulacio['Historic'].resample('ME').sum()
aport_esc_mensual = escenaris_simulacio['Torrencialitat'].resample('ME').sum()
energia_turbinada_esc_int = (datos.energia_turbinada_mensual_internes * (aport_esc_mensual / aport_hist_mensual) ** 0.58).dropna()
f_ebre = 0.23+(1-0.23)*(aport_esc_mensual / aport_hist_mensual) ** 0.58
energia_turbinada_esc_ebre = (datos.energia_turbinada_mensual_ebre * f_ebre).dropna()


# energia_turbinada_esc_int.plot()
energia_turbinada_esc_ebre.plot()
datos.energia_turbinada_mensual_ebre['2016-01-01':].plot()


#%%
%%time

from EnerSimFunc import (simulate_full_water_management)
# Suposem que tens df_tests amb els resultats dels 100 escenaris
# i scenarios_params amb els paràmetres d'entrada

# # Escollir esceari per index de solucions NSGA-II
# idx = df_pareto[df_pareto_nsga['mean_level'] > 80].total_costs.idxmin()
# # idx = df_pareto[df_pareto_nsga['min_level'] > 50].total_costs.idxmin()
# row = df_pareto_nsga.iloc[idx]

# # Índexs de solucions exemplars
# idx_max_min_level = front_nsga['min_level'].idxmax()      # Màxim nivell mínim
# idx_max_mean_level = front_nsga['mean_level'].idxmax()    # Màxim nivell mitjà
# idx_min_residu = front_nsga['obj_gasimports'].idxmin()       # Mínim residu tèrmic
# idx_min_cost = front_nsga['obj_costs'].idxmin()           # Mínim cost
# # idx_min_residu_r0
idx_min_cost_r0 = front_nsga[front_nsga['restriction_days']==0]['obj_costs'].idxmin()



# # parametres fixes segons escenari
# nucleares_activas = [True, True, True]
# potencia_cogeneracio = 542
# demanda_electrica = 1.25

nucleares_activas = [False, False, False]
potencia_cogeneracio = 122.4
demanda_electrica = 1.5
potencia_autoconsumo = 5000

row = df_rgs_2040.iloc[16971] # la millor solucio
row = df_rgs_2040.iloc[23464] # la millor amb llindar_desal_max > 2

# row = front_nsga.iloc[913]

# row = front_nsga.iloc[idx_compromis]
# Mapeo de nombres de parámetros
params = {
    'potencia_solar': row['potencia_solar'],
    'potencia_eolica': row['potencia_eolica'],
    'potencia_baterias': row['potencia_baterias'],
    'min_run_hours': int(row['min_run_hours']),
    'max_desalation': row['max_desalation'],
    # 'max_desalation': 64,
    'max_regen': row['max_regen'],
    # 'max_regen': 0.350,
    # 'llindar_desal_max': int(row['llindar_activacio_desal_max']),
    'llindar_desal_max': int(row['llindar_desal_max']),
    # 'llindar_desal_max': 2,                   # ALERTA -------------------------
    'midpoint_estimation': row['midpoint_estimation'],
    # 'overflow_threshold': row['overflow_threshold_pct'],
    'overflow_threshold': row['overflow_threshold'],
    'seasonal_phase': 0.0,
    'seasonal_amplitude': 0.0,
    'regen_base_pct': 0.5,
    # 'llindar_regen_max': int(row['llindar_activacio_regen_max']),
    'llindar_regen_max': int(row['llindar_regen_max']),
    'derivada_nivell': 0.0,
    'x1_base_eme': int(row['x1_base_eme']),
    'x2_gap_exc': int(row['x2_gap_exc']),
    # 'x1_base_eme': 16,
    # 'x2_gap_exc': 9,
    'x3_gap_ale': int(row['x3_gap_ale']),   #ALERTA -------------------------
    'x4_gap_pre': int(row['x4_gap_pre']),
    # 'x3_gap_ale': 40,
    # 'x4_gap_pre': 60,

}

# # Escollir escenari per índex (ex: el millor segons una mètrica)
# idx = df_rgs['min_level'].idxmax()  # O qualsevol criteri

# # Obtenir paràmetres d'aquest escenari de solucions RGS
# params = escenaris[idx]

umbrales = increments_a_llindars(
    params['x1_base_eme'],
    params['x2_gap_exc'],
    params['x3_gap_ale'],
    params['x4_gap_pre']
)

# Executar individualment
# result = run_case(params)


# O accedir al df_sintetic complet:
result_complet = procesar_escenario(
    # --- Sèries de dades ---
    df_demanda=datos.demanda,
    df_nuclear=datos.nuclears_base,
    df_cogeneracion=datos.cogeneracion_h,
    df_solar=datos.solar_h,
    df_eolica=datos.eolica_h,
    df_autoconsum=datos.autoconsum_hourly,
    df_potencia=datos.potencia,
    df_niveles_int=datos.df_pct_int_h.squeeze(),
    df_niveles_ebro=datos.df_pct_ebre_h.squeeze(),
    
    #ESCENARI HIDROLOGIC HISTORIC
    df_energia_turbinada_mensual_internes=datos.energia_turbinada_mensual_internes,
    df_energia_turbinada_mensual_ebre=datos.energia_turbinada_mensual_ebre,
    df_nivel_si=hydro_base_level,

    # ESCENARI HIDROLOGIC PERSONALITZAT    
    # df_energia_turbinada_mensual_internes=energia_turbinada_esc_int,
    # df_energia_turbinada_mensual_ebre=energia_turbinada_esc_ebre,
    # df_nivel_si=hydro_esc1_level,

        
    # --- Paràmetres físics ---
    max_capacity_int=max_capacity_int,
    max_capacity_ebro=max_capacity_ebro,
    potencia_max_int=potencia_max_hidraulica_int,
    potencia_max_ebro=potencia_max_hidraulica_ebro,
    sensibility_int=sensibility_int,
    sensibility_ebro=sensibility_ebro,
    # hydro_min_for_level=hydro_min_for_level,  # NOU: Afegir aquesta variable
    
    # --- Paràmetres hídrics ---
    consumo_base_diario_estacional_hm3=consumo_base_diario_estacional,  # NOU
    save_hm3_per_mwh=1/desal_sensibility, 

    precomputed=precomputed,
    potencia_solar=params['potencia_solar'],
    potencia_eolica=params['potencia_eolica'],
    potencia_baterias=params['potencia_baterias'],
    min_run_hours=params['min_run_hours'],
    max_desalation=params['max_desalation'],
    midpoint_estimation=params['midpoint_estimation'],
    overflow_threshold_pct=params['overflow_threshold'],
    seasonal_phase_months=params['seasonal_phase'],
    seasonal_desal_amplitude=params['seasonal_amplitude'],
    max_regen = params['max_regen'],
    regen_base_pct=params['regen_base_pct'],
    llindar_activacio_desal_max=6, #params['llindar_desal_max'],
    llindar_activacio_regen_max=params['llindar_regen_max'],
    k_deriv=params['derivada_nivell'],
    umbrales_sequia=umbrales,
    nucleares_activas = nucleares_activas,
    potencia_cogeneracion = potencia_cogeneracion,
    # potencia_autoconsumo = 1188.6267,
    # potencia_autoconsumo = 1381,
    potencia_autoconsumo = potencia_autoconsumo,
    demanda_electrica = demanda_electrica,
    # nucleares_activas = [False, False, False],
    # potencia_cogeneracion = 122.4,    
)

df_sintetic = result_complet['energy_data']

# result_complet['hydro_metrics']
#%%
result_complet['level_final'].plot()

print('restriccions', result_complet['savings_final']['2020-01-01':].sum())
print('regeneració', result_complet['regen_final']['2020-01-01':].sum())
print('dessalinització', result_complet['desal_final_hm3']['2020-01-01':].sum())
print('vessaments', result_complet['spillage_hm3']['2020-01-01':].sum())
print('total', result_complet['regen_final']['2020-01-01':].sum() + result_complet['desal_final_hm3']['2020-01-01':].sum() + result_complet['savings_final']['2020-01-01':].sum())
print('dias amb restriccions',result_complet['hydro_metrics']['Restricciones escenario (días)'])
print('------------')
print('diferencia aportacions',(escenaris_simulacio['Historic'] - escenaris_simulacio['Sequera_Anticipada']).sum())
print('diferencia aportacions',(escenaris_simulacio['Historic'] - escenaris_simulacio['Sequera_Extensa']).sum())

print('------------')
print('estalvi_historic',precomputed['estalvi'][-1] - precomputed['estalvi']['2020-01-01 00:00:00'])
print('dessal_historic',datos.dessalacio_diaria['2020-01-01':].sum())
print('regen_historic',datos.regeneracio_diaria['2020-01-01':].sum())
print('total_historic',precomputed['estalvi'][-1] - precomputed['estalvi']['2020-01-01 00:00:00']+datos.dessalacio_diaria['2020-01-01':].sum()+datos.regeneracio_diaria['2020-01-01':].sum())


#%%
result_complet['savings_final'].sum()

result_complet['desal_final_hm3'].sum()
result_complet['desal_final_hm3'].plot()
result_complet['regen_final'].sum()
result_complet['regen_final'].plot()


result_complet['desal_final_hm3']['2023-01-01':'2023-12-31'].sum()/(365*24*0.0458) #78%
datos.dessalacio_diaria['2023-01-01':'2023-12-31'].sum()/(365*24*0.00917) #93%    
    # return {
    #     'energy_data': results,
    #     'energy_metrics_pct': energy_metrics_pct,
    #     'energy_metrics_MWh': energy_metrics_MWh,
    #     'energy_metrics_pct2': energy_metrics_pct2,
    #     'hydro_metrics': hydro_metrics,
    #     'level_final': level_final,
    #     'regen_final': regen_final,
    #     'desal_final': desal_final,
    #     'desal_final_hm3': desal_final_hm3,
    #     'savings_final': savings_final,
    #     'savings_agro_final': savings_agro_final,
    #     'savings_ind_final': savings_ind_final,
    #     'savings_urba_final': savings_urba_final,
    #     'spillage_hm3': spillage_final,
    #     'extra_hydro': extra_hydro_final,
    #     'excedents': surpluses_net[surpluses_net>0],
    #     'deficits': surpluses_net[surpluses_net<0],
    #     'capacity_factor': int(100 * desal_final.mean() / max_desalation),
    #     'costes': costes
    # }
    
#%%

# Configurar estilo académico
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# Crear figura
fig, ax = plt.subplots(figsize=(14, 7))

# Datos empíricos (línea destacada)
empirical_data = (datos.df_pct_int_h * 100)['2016-01-01':]
ax.plot(empirical_data.index, empirical_data.values, 
        color='black', linewidth=2, linestyle='-',
        label='Datos empíricos', zorder=5)

# Modelos (con diferentes estilos de línea)
model_series = [s0, s2]
model_labels = ['Escenari S0', 'Escenari S2/S3']
line_styles = ['--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]  # diferentes estilos
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']    # paleta de colores académica

for i, (series, label, style, color) in enumerate(zip(model_series, model_labels, line_styles, colors)):
    ax.plot(series.index, series.values, 
            linewidth=1.8, linestyle=style, color=color,
            alpha=0.85, label=label, zorder=4-i)

# Personalización
ax.set_xlabel('Data', fontweight='bold')
ax.set_ylabel('Nivell embassaments conques internes (%)', fontweight='bold')
ax.set_title('Comparació entre dades empíriques i escenaris del model', 
             fontsize=16, fontweight='bold', pad=15)

# Añadir cuadrícula sutil
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

# Leyenda
legend = ax.legend(loc='best', frameon=True, shadow=True, 
                   fancybox=True, framealpha=0.95, edgecolor='black')
legend.get_frame().set_linewidth(1)

# Ajustar márgenes
plt.tight_layout()

# Mostrar gráfico
plt.show()

# Opcional: Guardar en alta resolución para tesis
# fig.savefig('comparacion_escenarios_modelo.png', dpi=300, bbox_inches='tight')
# fig.savefig('comparacion_escenarios_modelo.pdf', bbox_inches='tight')

#%%

# df_sintetic = resultats_escenaris['S_comprom']['energy_data']

sample = df_sintetic[['Demanda','Nuclear','Cogen_w','Eòlica_w','Solar_w', 'Hidràulica', 'Bateries', 'Gas+Imports', 'Càrrega']]
sample.columns = ['Demanda','Nuclear','Cogeneració','Eòlica','Solar', 'Hidràulica', 'Bateries', 'Gas+Import.', 'Càrrega']


color_dict = {
    'Nuclear': '#9467bd',
    'Eòlica': '#2ca02c',
    'Solar': '#ff7f0e',
    'Cogeneració': '#8c564b',
    'Hidràulica': '#1f78b4',
    'Bateries': '#d62728',  # rojo intenso, consistente con el estilo de la paleta
    'Gas+Import.': '#7f7f7f'
}

dia = '2023-12-31'

# dia = '2023-04-01'
# dia = '2023-07-01'
# dia = '2020-04-01'
# dia = '2020-05-01'
# dia = '2019-12-05'
dia = '2019-06-05'
dia = '2018-01-05'
inicio = pd.to_datetime(dia)
fin = inicio + pd.Timedelta(hours=23)
fin = inicio + pd.Timedelta(hours=24*7)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(12, 6))
# ax.set_ylim([0, 6000])

# Graficar el área acumulada del DataFrame
sample[inicio:fin].iloc[:,1:-1].plot.area(
# sample[inicio:fin].iloc[:,1:-1].plot.area(
    ax=ax,
    stacked=True,
    alpha=0.5,
    color=[color_dict.get(col, '#000000') for col in sample.columns[1:]]
)


# # Graficar la línea de la Serie
# sample[inicio:fin]['Càrrega'].plot(ax=ax, color='black', style='--', linewidth=2)
# (sample[inicio:fin]['Demanda']).plot(ax=ax, color='grey', linewidth=2, label='Demanda')

sample[inicio:fin]['Càrrega'].plot(ax=ax, color='black', linestyle='--', linewidth=2.5, zorder=10)
(sample[inicio:fin]['Demanda']).plot(ax=ax, color='#d62728', linewidth=3, zorder=11, label='Demanda')

ax.set_xlabel('Data', labelpad=-15)
ax.set_ylabel('Potència (MW)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, frameon=False)
# ax.legend(loc='upper left')
plt.show()


#%%

# Expandir el diccionari de mètriques percentuals a columnes individuals
df_metrics_pct = pd.DataFrame(df_rgs['energy_metrics_pct'].tolist())
df_metrics_mwh = pd.DataFrame(df_rgs['energy_metrics_MWh'].tolist())
# df_metrics_pct['Fossil+Imports %'].describe()

df_rgs.min_level.describe()
df_rgs.mean_level.describe()
df_rgs.squared_dev.describe()
df_rgs.restriction_days.describe()
(df_rgs.total_costs/(1000000*9)).describe()
(df_rgs.gas_imports/(1000*9)).describe()
df_metrics_pct = pd.DataFrame(df_rgs['energy_metrics_pct'].tolist())
df_metrics_pct['Cicles + Import. %'].describe()
df_rgs.regen_hm3.describe()
df_rgs.desal_hm3.describe()
df_rgs.restriction_savings.describe()

# # Ara pots fer estadístiques fàcilment
# df_metrics_pct.describe()

# # O accedir a columnes individuals
# df_metrics_pct['Wind %'].mean()
# df_metrics_pct['Renewables %'].max()

# # Si vols tot integrat en un sol DataFrame:
# df_expanded = pd.concat([
#     df_rgs.drop(columns=['energy_metrics_pct', 'energy_metrics_MWh']),  # Columnes originals sense diccionaris
#     pd.DataFrame(df_rgs['energy_metrics_pct'].tolist()).add_suffix('_pct'),
#     pd.DataFrame(df_rgs['energy_metrics_MWh'].tolist()).add_suffix('_MWh')
# ], axis=1)

# # Ara tens tot pla i pots fer:
# df_expanded.describe()


#%%

# Diccionari de criteris: {nom: (columna_filtre, valor_filtre, columna_optimitzar, sentit)}
criteris = {
    'S1': ('min_level', '> 70', 'total_costs', 'min'),
    'S2': ('mean_level', '> 80', 'spillage_hm3', 'min'),
    'S3': ('mean_level', '> 80', 'total_costs', 'min'),
    'S4': ('restriction_days', '< 0.1', 'total_costs', 'min'),
    'S5': ('gas_imports', '< 1.0', 'total_costs', 'min'),
}

solucions = {}
for nom, (col_filtre, condicio, col_opt, sentit) in criteris.items():
    mask = df_rgs.eval(f'{col_filtre} {condicio}')
    if mask.any():
        if sentit == 'min':
            idx = df_rgs.loc[mask, col_opt].idxmin()
        else:
            idx = df_rgs.loc[mask, col_opt].idxmax()
        solucions[nom] = df_rgs.loc[idx]
    else:
        print(f"{nom}: Cap solució compleix {col_filtre} {condicio}")

df_seleccio = pd.DataFrame(solucions).T
df_seleccio.gas_imports = df_seleccio.gas_imports / (1000000*9)
df_seleccio.total_costs = df_seleccio.total_costs / (1000000*9)

#%%
# =============================================================================
# CONFIGURACIÓ (fàcil de modificar)
# =============================================================================

# Criteris de selecció: {nom: (columna_filtre, condicio, columna_optimitzar, sentit)}
criteris = {
    'S0': ('min_level', '> 50', 'total_costs', 'min'),
    'S1': ('min_level', '> 70', 'total_costs', 'min'),
    'S2': ('mean_level', '> 80', 'spillage_hm3', 'min'),
    'S3': ('mean_level', '> 80', 'total_costs', 'min'),
    'S4': ('restriction_days', '< 0.1', 'total_costs', 'min'),
    'S5': ('gas_imports', '< 1.0', 'total_costs', 'min'),
}

# Columnes a mostrar (afegeix o treu segons necessitat)
cols_objectius = {
    'total_costs': 'Cost (M€/y)',
    'min_level': 'Niv. Mín (%)',
    'mean_level': 'Niv. Mitjà (%)',
    'gas_imports': 'Residu (TWh/y)',
    'spillage_hm3': 'Vessaments (hm³/y)',
    'restriction_days': 'Restriccions (d)',
}

cols_variables = {
    'potencia_solar': 'Solar (GW)',
    'potencia_eolica': 'Eòlica (GW)',
    'potencia_baterias': 'Bateries (GW)',
    'max_desalation': 'Desal. (MW)',
    'llindar_desal_max': 'Llindar saturació des.',    
    'max_regen': 'Regen. (hm³/d)',
    'llindar_regen_max': 'Llindar saturació reg.',
    'L_prealerta': 'Llindar Prealerta (%)',
    'L_alerta': 'Llindar Alerta (%)',
}

# =============================================================================
# EXTRACCIÓ DE SOLUCIONS
# =============================================================================
# df_pareto = df_rgs
df_pareto = front_nsga
solucions = {}
for nom, (col_filtre, condicio, col_opt, sentit) in criteris.items():
    mask = df_pareto.eval(f'{col_filtre} {condicio}')
    if mask.any():
        if sentit == 'min':
            idx = df_pareto.loc[mask, col_opt].idxmin()
        else:
            idx = df_pareto.loc[mask, col_opt].idxmax()
        solucions[nom] = df_pareto.loc[idx]
    else:
        print(f"⚠️  {nom}: Cap solució compleix {col_filtre} {condicio}")

df_seleccio = pd.DataFrame(solucions).T
# df_seleccio.gas_imports = df_seleccio.gas_imports / (1000000*9)
# df_seleccio.total_costs = df_seleccio.total_costs / (1000000*9)
df_seleccio.gas_imports = df_seleccio.gas_imports
df_seleccio.total_costs = df_seleccio.total_costs
# =============================================================================
# FUNCIÓ PER IMPRIMIR TAULES
# =============================================================================

def imprimir_taula(df, cols_dict, titol, decimals=2):
    """
    Imprimeix una taula formatejada al shell.
    
    Paràmetres:
    -----------
    df : pd.DataFrame
        DataFrame amb les solucions (files = solucions)
    cols_dict : dict
        {nom_columna_original: nom_mostrar}
    titol : str
        Títol de la taula
    decimals : int
        Nombre de decimals a mostrar
    """
    # Filtrar columnes existents
    cols_existents = {k: v for k, v in cols_dict.items() if k in df.columns}
    cols_faltants = [k for k in cols_dict.keys() if k not in df.columns]
    
    if cols_faltants:
        print(f"⚠️  Columnes no trobades: {cols_faltants}\n")
    
    # Crear subtaula amb noms nous
    df_taula = df[list(cols_existents.keys())].copy()
    df_taula.columns = list(cols_existents.values())
    df_taula = df_taula.round(decimals)
    
    # Imprimir
    print("=" * 80)
    print(f" {titol}")
    print("=" * 80)
    # print(df_taula.to_string())
    # print(df_taula.to_string(float_format=lambda x: f'{x:.{decimals}f}'))
    # print(df_taula.to_string(float_format=lambda x: f'{x:.{decimals}f}', index=False))
    print(df_taula.to_string(float_format=lambda x: f'{x:,.{decimals}f}'.replace(',', 'X').replace('.', ',').replace('X', '.'), index=False))
    print()
    
    return df_taula

# =============================================================================
# IMPRIMIR TAULES
# =============================================================================

# Taula A: Objectius
df_obj = imprimir_taula(
    df_seleccio, 
    cols_objectius, 
    "TAULA A: OBJECTIUS I MÈTRIQUES DE RESULTAT",
    decimals=1
)

# Taula B: Variables
df_var = imprimir_taula(
    df_seleccio, 
    cols_variables, 
    "TAULA B: VARIABLES DE DECISIÓ",
    decimals=3
)

# =============================================================================
# OPCIONAL: Exportar a LaTeX / Markdown / CSV
# =============================================================================

# LaTeX
# print(df_obj.to_latex(caption='Mètriques de les solucions representatives'))
# print(df_var.to_latex(caption='Variables de decisió'))

# Markdown (per copiar a documentació)
# print(df_obj.to_markdown())

# CSV
# df_obj.to_csv('solucions_objectius.csv')
# df_var.to_csv('solucions_variables.csv')

# =============================================================================
# AFEGIR MÉS TAULES (exemple)
# =============================================================================

# cols_adicionals = {
#     'renewables_pct': 'Renovables (%)',
#     'clean_coverage_pct': 'Cob. Neta (%)',
#     'surpluses_twh': 'Excedents (TWh/y)',
# }
# 
# df_extra = imprimir_taula(
#     df_seleccio,
#     cols_adicionals,
#     "TAULA C: MÈTRIQUES ADDICIONALS"
# )


## Exemple de Sortida
# ```
# ================================================================================
#  TAULA A: OBJECTIUS I MÈTRIQUES DE RESULTAT
# ================================================================================
#     Cost (M€/y)  Niv. Mín (%)  Niv. Mitjà (%)  Residu (TWh/y)  Vessaments (hm³/y)  Restriccions (hm³/y)
# S1       523.45         71.23           82.45            3.21               45.67                 12.34
# S2       687.12         65.43           84.56            2.87               12.34                 23.45
# S3       498.76         62.34           81.23            4.12               34.56                 15.67
# S4       892.34         78.90           89.12            1.23               23.45                  0.00
# S5      1234.56         75.67           87.89            0.45               56.78                  8.90

# ================================================================================
#  TAULA B: VARIABLES DE DECISIÓ
# ================================================================================
#     Solar (GW)  Eòlica (GW)  Bateries (GWh)  Desal. (MW)  Regen. (hm³/d)  Llindar Alerta (%)  Llindar Excep. (%)
# S1       12.34        18.56           4.50        45.00            0.15               40.00               25.00
# S2       15.67        22.34           6.20        60.00            0.20               38.00               22.00
# ...

#%%
demanda_total_gwh = df_metrics_mwh['Fossil+Imports (GWh/y)'] / (df_metrics_pct['Fossil+Imports %'] / 100)

(df_rgs.gas_imports/(1000*9) / demanda_total_gwh *100).max()
(df_rgs.gas_imports/(1000*9) / demanda_total_gwh *100).std()

#%%
def calcular_distancia_utopia(df, cols_objectius, pesos=None):
    """
    Troba la solució més propera al punt ideal (utopia).
    
    cols_objectius: dict {columna: 'min' o 'max'}
    """
    if pesos is None:
        pesos = {col: 1.0 for col in cols_objectius}
    
    # Normalitzar cada objectiu a [0, 1]
    df_norm = pd.DataFrame(index=df.index)
    for col, sentit in cols_objectius.items():
        if sentit == 'min':
            # Mínim = 0 (ideal), Màxim = 1 (pitjor)
            df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:  # 'max'
            # Màxim = 0 (ideal), Mínim = 1 (pitjor)
            df_norm[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min())
    
    # Distància euclidiana ponderada al punt ideal (0, 0, ...)
    distancia = sum(pesos[col] * df_norm[col]**2 for col in cols_objectius) ** 0.5
    
    return distancia.idxmin(), distancia

# Ús:
# cols_objectius = {
#     'obj_costs': 'min',
#     'min_level': 'max',
#     'gas_imports': 'min'
# }

cols_objectius = {
    'obj_costs': 'min',
    'min_level': 'max',
    # 'obj_gapsq': 'min'
    'obj_gasimports': 'min'
}

pesos_jerarquics = {
    # 'obj_gapsq': 0.3,
    'obj_gasimports': 0.3,
    'min_level': 0.5,
    'obj_costs': 0.2
}
idx_compromis, distancies = calcular_distancia_utopia(front_nsga, cols_objectius)
idx_compromis_2, distancies = calcular_distancia_utopia(front_nsga, cols_objectius, pesos=pesos_jerarquics)
print(f"Índex solució compromís: {idx_compromis}")
print(f"Índex solució compromís: {idx_compromis_2}")

#%%
# Índexs de solucions exemplars
idx_max_min_level = front_nsga['min_level'].idxmax()      # Màxim nivell mínim
idx_max_mean_level = front_nsga['mean_level'].idxmax()    # Màxim nivell mitjà
# idx_min_gas = front_nsga['gas_imports'].idxmin()       # Mínim residu tèrmic
# idx_min_residu = front_nsga['obj_gapsq'].idxmin()       # Mínim residu tèrmic
# idx_min_exced = front_nsga['surpluses_total'].idxmin()       # Mínim residu tèrmic
idx_min_cost = front_nsga['obj_costs'].idxmin()           # Mínim cost
# idx_min_residu_r0 = front_nsga[front_nsga['restriction_days']==0]['obj_gasimports'].idxmin()
# idx_min_residu_r0 = front_nsga[front_nsga['restriction_days']==0]['gas_imports'].idxmin()
# idx_min_cost_r0 = front_nsga[front_nsga['restriction_days']==0]['obj_costs'].idxmin()
idx_compromis, _ = calcular_distancia_utopia(front_nsga, cols_objectius)
idx_compromis_2, _ = calcular_distancia_utopia(front_nsga, cols_objectius, pesos=pesos_jerarquics)

# =============================================================================
# CONFIGURACIÓ D'ESCENARIS
# =============================================================================

# Paràmetres fixes segons escenari 2030
config_base = {
    'nucleares_activas': [True, True, True],
    'potencia_cogeneracion': 542,
    'potencia_autoconsumo': 2185, #1381,
    'demanda_electrica': 1.25,
    'max_desalation': 64,
    'max_regen': 0.250,
    'x1_base_eme': 16,
    'x2_gap_exc': 9,
}


# # Paràmetres fixes segons escenari 2050
# config_base = {
#     'nucleares_activas': [False, False, False],
#     'potencia_cogeneracion': 122,
#     'demanda_electrica': 2,
#     'max_desalation': 64,
#     'max_regen': 0.350,
#     'x1_base_eme': 16,
#     'x2_gap_exc': 9,
# }

# Definició dels escenaris a avaluar
escenaris_config = {
    # 'S_minL': idx_max_min_level,
    'S_meanL': idx_max_mean_level,
    'S_decarb': idx_min_gas,
    'S_cost': idx_min_cost,
    'S_eficient': idx_min_residu,
    # 'S_decarb_r0': idx_min_residu_r0,
    # 'S_cost_r0': idx_min_cost_r0,
    'S_comprom': idx_compromis,
    'S_comprom_2': idx_compromis_2,
}

# =============================================================================
# FUNCIÓ PER EXECUTAR UN ESCENARI
# =============================================================================

def executar_escenari(row, config_base):
    """Executa un escenari i retorna el resultat complet."""
    
    params = {
        'potencia_solar': row['potencia_solar'],
        'potencia_eolica': row['potencia_eolica'],
        'potencia_baterias': row['potencia_baterias'],
        'min_run_hours': int(row['min_run_hours']),
        'max_desalation': config_base['max_desalation'],
        # 'max_desalation': row['max_desalation'],
        'max_regen': config_base['max_regen'],
        # 'llindar_desal_max': int(row['llindar_activacio_desal_max']),
        'llindar_desal_max': int(config_base['llindar_desal_max']) if 'llindar_desal_max' in config_base else int(row['llindar_activacio_desal_max']),
        # 'llindar_desal_max': 2,
        'midpoint_estimation': row['midpoint_estimation'],
        'overflow_threshold': row['overflow_threshold_pct'],
        'seasonal_phase': 0.0,
        'seasonal_amplitude': 0.0,
        'regen_base_pct': 0.5,
        'llindar_regen_max': int(row['llindar_activacio_regen_max']),
        'derivada_nivell': 0.0,
        'x1_base_eme': config_base['x1_base_eme'],
        'x2_gap_exc': config_base['x2_gap_exc'],
        'x3_gap_ale': int(config_base['x3_gap_ale']) if 'x3_gap_ale' in config_base else int(row['x3_gap_ale']),
        'x4_gap_pre': int(config_base['x4_gap_pre']) if 'x4_gap_pre' in config_base else int(row['x4_gap_pre']),
    }
    
    umbrales = increments_a_llindars(
        params['x1_base_eme'],
        params['x2_gap_exc'],
        params['x3_gap_ale'],
        params['x4_gap_pre']
    )
    
    result = procesar_escenario(
        df_demanda=datos.demanda,
        df_nuclear=datos.nuclears_base,
        df_cogeneracion=datos.cogeneracion_h,
        df_solar=datos.solar_h,
        df_eolica=datos.eolica_h,
        df_autoconsum=datos.autoconsum_hourly,
        df_potencia=datos.potencia,
        df_niveles_int=datos.df_pct_int_h.squeeze(),
        df_niveles_ebro=datos.df_pct_ebre_h.squeeze(),

        # #ESCENARI HIDROLOGIC HISTORIC
        df_energia_turbinada_mensual_internes=datos.energia_turbinada_mensual_internes,
        df_energia_turbinada_mensual_ebre=datos.energia_turbinada_mensual_ebre,
        df_nivel_si=hydro_base_level,

        # ESCENARI HIDROLOGIC PERSONALITZAT    
        # df_energia_turbinada_mensual_internes=energia_turbinada_esc_int,
        # df_energia_turbinada_mensual_ebre=energia_turbinada_esc_ebre,
        # df_nivel_si=hydro_esc1_level,

        max_capacity_int=max_capacity_int,
        max_capacity_ebro=max_capacity_ebro,
        potencia_max_int=potencia_max_hidraulica_int,
        potencia_max_ebro=potencia_max_hidraulica_ebro,
        sensibility_int=sensibility_int,
        sensibility_ebro=sensibility_ebro,
        consumo_base_diario_estacional_hm3=consumo_base_diario_estacional,
        save_hm3_per_mwh=1/desal_sensibility,
        precomputed=precomputed,
        potencia_solar=params['potencia_solar'],
        potencia_eolica=params['potencia_eolica'],
        potencia_baterias=params['potencia_baterias'],
        potencia_autoconsumo=config_base['potencia_autoconsumo'],
        min_run_hours=params['min_run_hours'],
        max_desalation=params['max_desalation'],
        midpoint_estimation=params['midpoint_estimation'],
        overflow_threshold_pct=params['overflow_threshold'],
        seasonal_phase_months=params['seasonal_phase'],
        seasonal_desal_amplitude=params['seasonal_amplitude'],
        max_regen=params['max_regen'],
        regen_base_pct=params['regen_base_pct'],
        llindar_activacio_desal_max=params['llindar_desal_max'],
        llindar_activacio_regen_max=params['llindar_regen_max'],
        k_deriv=params['derivada_nivell'],
        umbrales_sequia=umbrales,
        nucleares_activas=config_base['nucleares_activas'],
        potencia_cogeneracion=config_base['potencia_cogeneracion'],
        # potencia_autoconsumo=1381,
        demanda_electrica=config_base['demanda_electrica'],
        CF_eolica_obj = None,
        usar_CF_automatic = True,
    )
    
    return result

# =============================================================================
# EXECUTAR TOTS ELS ESCENARIS I ACUMULAR
# =============================================================================

resultats_escenaris = {}
nivells_escenaris = {}

for nom, idx in escenaris_config.items():
    print(f"Executant escenari {nom} (idx={idx})...")
    row = front_nsga.loc[idx]
    result = executar_escenari(row, config_base)
    
    resultats_escenaris[nom] = result
    nivells_escenaris[nom] = result['level_final']
    
    print(f"  → min_level: {result['level_final'].min():.1f}%, "
          f"mean_level: {result['level_final'].mean():.1f}%")

print("\n✓ Tots els escenaris executats!")

#%%

# Configurar estilo académico
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# Configuració dels escenaris a graficar
escenaris_plot = {
    # 'S_minL': {'label': 'Màx. seguretat (L_min)', 'style': '--', 'color': '#1f77b4'},
    # 'S_meanL': {'label': 'Màx. seguretat (L_mean)', 'style': '-.', 'color': '#ff7f0e'},
    # 'S_decarb': {'label': 'Màx. descarbonització', 'style': ':', 'color': '#2ca02c'},
    # 'S_cost': {'label': 'Mín. cost', 'style': ':', 'color': '#1f77b4'},    
    # 'S_cost_r0': {'label': 'Mínim cost (sense restriccions)', 'style': (0, (3, 1, 1, 1)), 'color': '#d62728'},
    # 'S_comprom': {'label': 'Compromís', 'style': (0, (5, 2)), 'color': '#9467bd'},
    'S_comprom': {'label': 'Solució model', 'style': (0, (5, 2)), 'color': '#9467bd'},
}

# Crear figura
fig, ax = plt.subplots(figsize=(14, 7))

# Dades empíriques (línia destacada)
empirical_data = (datos.df_pct_int_h * 100)['2016-01-01':]
# empirical_data = hydro_base_level
ax.plot(empirical_data.index, empirical_data.values, 
        color='black', linewidth=1.75, linestyle='-',
        label='Dades històriques', zorder=10)

# Escenaris
for i, (nom, config) in enumerate(escenaris_plot.items()):
    if nom in nivells_escenaris:
        series = nivells_escenaris[nom]
        ax.plot(series.index, series.values, 
                linewidth=1.8, 
                linestyle=config['style'], 
                color=config['color'],
                alpha=0.85, 
                label=config['label'], 
                zorder=5-i)

# Personalització
ax.set_xlabel('Data', fontweight='bold')
ax.set_ylabel('Nivell embassaments conques internes (%)', fontweight='bold')
ax.set_title('Comparació de nivells: dades empíriques vs escenaris optimitzats', 
             fontsize=16, fontweight='bold', pad=15)

# Límits eix Y
ax.set_ylim(0, 105)

# Cuadrícula
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# Llegenda
legend = ax.legend(loc='lower left', frameon=True, shadow=True, 
                   fancybox=True, framealpha=0.95, edgecolor='black',
                   ncol=2)
legend.get_frame().set_linewidth(1)

# Ajustar marges
plt.tight_layout()
plt.show()

#%%
# Mètriques a incloure amb nom i columna
metrics = {
    'Nivell mínim (%)': front_nsga['min_level'],
    'Nivell mitjà (%)': front_nsga['mean_level'],
    'Dies restricció (d)': front_nsga['restriction_days'],
    'Residu tèrmic (TWh/y)': front_nsga['gas_imports'] / 9,  # Si està en MWh
    'Excedents (TWh/y)': front_nsga['surpluses_total'] / (1e6*9),
    'Cost econòmic (M€/y)': front_nsga['obj_costs'] / 9,       # Si està en €
}

# Funció format espanyol (punt milers, coma decimals)
def fmt(x):
    return f"{x:,.1f}".replace(',', 'X').replace('.', ',').replace('X', '.')

# Construir dades
rows = []
for nom, col in metrics.items():
    rows.append({   
        'Mètrica': nom,
        'Mínim': fmt(np.min(col)),
        'P25': fmt(np.percentile(col, 25)),
        'Mediana': fmt(np.percentile(col, 50)),
        'P75': fmt(np.percentile(col, 75)),
        'Màxim': fmt(np.max(col))
    })

df_taula = pd.DataFrame(rows)

# Imprimir per copiar a Word (tabulacions)
print(df_taula.to_string(index=False))

#%%

# =============================================================================
# TAULA COMPARATIVA D'ESCENARIS
# =============================================================================

n_anys = 9  # Nombre d'anys de simulació

def extreure_metriques(result, row=None):
    """Extreu mètriques d'un resultat d'escenari."""
    return {
        'Potència solar (MW)': row['potencia_solar'],
        'Potència eòlica (MW)': row['potencia_eolica'],
        'Potència bateries (MW)': row['potencia_baterias'],
        'Mínim hores des. (h)': row['min_run_hours'],
        'Punt Inflexió (%)': row['midpoint_estimation'],
        'Llindar Sobreeiximent (%)': row['overflow_threshold_pct'],
        'Llindar Prealerta (%)': row['L_prealerta'] if row is not None else None,
        'Llindar Alerta (%)': row['L_alerta'] if row is not None else None,
        'Llindar Saturació des. (%)': row['llindar_activacio_desal_max'],
        'Llindar Saturació reg. (%)': row['llindar_activacio_regen_max'],        
        'Nivell mínim (%)': result['level_final'].min(),
        'Nivell mitjà (%)': result['level_final'].mean(),
        'Dies restricció (d)': result['hydro_metrics']['Restricciones escenario (días)'],
        'Volum restringit (hm³)': result['savings_final'].sum(),
        'Volum dessalat (hm³)': result['desal_final_hm3'].sum(),
        'Volum regenerat (hm³)': result['regen_final'].sum(),
        'Vessaments (hm³)': result['spillage_hm3'].sum(),
        'Residu tèrmic (TWh/y)': result['energy_data']['Gas+Imports'].sum() / 1e6 / n_anys,
        # 'Residu tèrmic (TWh/y)': row['gas_imports'] / n_anys,
        'Excedents (TWh/y)': result['excedents'].sum() / 1e6 / n_anys,
        'Cost econòmic (M€/y)': result['costes']['total'] / 1e6 / n_anys,
        # 'Potència dessal (MW)': row['max_desalation'],
    }

# Construir taula
taula_metriques = {}

for nom, idx in escenaris_config.items():   
    if nom in resultats_escenaris:
        row = front_nsga.loc[idx]
        taula_metriques[nom] = extreure_metriques(resultats_escenaris[nom], row)

# Afegir columna històric (sense intervenció)
taula_metriques['Històric'] = {
    'Nivell mínim (%)': np.min((datos.df_pct_int_h * 100)['2016-01-01':'2024-12-31']),
    'Nivell mitjà (%)': np.mean((datos.df_pct_int_h * 100)['2016-01-01':'2024-12-31']),
    'Dies restricció (d)': 870,  # O calcular si tens dades
    'Volum restringit (hm³)': 781.7,
    'Volum dessalat (hm³)': 345.7,
    'Vessaments (hm³)': 0, #adicionals
    'Residu tèrmic (TWh/y)': 9.7,
    'Cost econòmic (M€/y)': 0, #adicional
    'Llindar Prealerta (%)': 60,  # Valors actuals
    'Llindar Alerta (%)': 40,
}

# Crear DataFrame
df_comparativa = pd.DataFrame(taula_metriques)

# =============================================================================
# FORMATACIÓ PER IMPRIMIR
# =============================================================================

def format_valor(x, decimals=1):
    """Formata valor amb coma decimal i punts de milers."""
    if pd.isna(x) or x is None:
        return '—'
    return f'{x:,.{decimals}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')

# Aplicar format
df_formatat = df_comparativa.applymap(format_valor)

# Imprimir
print("=" * 100)
print(" TAULA COMPARATIVA D'ESCENARIS")
print("=" * 100)
print(df_formatat.to_string())
print()

# # =============================================================================
# # VERSIÓ TRANSPOSADA (escenaris com a columnes)
# # =============================================================================

# print("\n" + "=" * 100)
# print(" TAULA TRANSPOSADA (per copiar a Word)")
# print("=" * 100)
# print(df_formatat.T.to_string())

# **Sortida esperada (exemple):**
# ```
#                           S_minL    S_meanL   S_decarb   S_cost_r0  S_comprom   Històric
# Nivell mínim (%)            72,3       68,5       45,2        65,1       67,8       25,4
# Nivell mitjà (%)            89,2       91,5       76,3        82,4       85,6       62,3
# Dies restricció (d)            0          0         12           0          0          —
# Volum restringit (hm³)       0,0        0,0       45,2         0,0        0,0          —
# Volum dessalat (hm³)       156,3      148,7       89,4       132,5      145,2          —
# Vessaments (hm³)            45,2       52,3       12,4        38,7       41,5          —
# Residu tèrmic (TWh/y)        2,3        3,1       0,5          8,7        4,2          —
# Cost econòmic (M€/y)       892,4      756,3    1.234,5       523,4      687,2          —
# Llindar Prealerta (%)         65         62         58          60         63         60
# Llindar Alerta (%)            42         40         35          40         41         40



#%%

from pathlib import Path


# Carrega tots els E1-E4 i assigna clau 1-4 directament
fronts_saturacio = {
    int(f.stem.split('E')[1].split('_')[0]): pd.read_parquet(f)
    for f in Path('.').glob('df_nsga_*7vE*.parquet')
}

# Si necessites invertir l'ordre (E1=Prealerta->clau 4, E4=Emergència->clau 1):
fronts_saturacio = {5-k: v for k, v in fronts_saturacio.items()}

# =============================================================================
# CONFIGURACIÓ DELS FRONTS AMB DIFERENTS LLINDARS DE SATURACIÓ
# =============================================================================

# Diccionari amb els fronts de Pareto (ajusta els noms segons els teus)
# fronts_saturacio = {
#     1: front_nsga_sat1,  # Saturació a Emergència
#     2: front_nsga_sat2,  # Saturació a Excepcionalitat
#     3: front_nsga_sat3,  # Saturació a Alerta
#     4: front_nsga_sat4,  # Saturació a Prealerta
# }

etiquetes_saturacio = {
    1: 'Emergència',
    2: 'Excepcionalitat', 
    3: 'Alerta',
    4: 'Prealerta',
}

# Paràmetres fixes segons escenari 2030
config_base = {
    'nucleares_activas': [True, True, True],
    'potencia_cogeneracion': 542,
    'demanda_electrica': 1.25,
    'max_desalation': 64,
    'max_regen': 0.250,
    'x1_base_eme': 16,
    'x2_gap_exc': 9,
    'x3_gap_ale': 15,
    'x4_gap_pre': 20,
}

# =============================================================================
# FUNCIÓ PER CALCULAR ESCENARI DE COMPROMÍS
# =============================================================================

def calcular_idx_compromis(df, cols_objectius=None):
    """Troba l'índex de la solució de compromís (distància utopia)."""
    if cols_objectius is None:
        cols_objectius = {
            'obj_costs': 'min',
            'min_level': 'max',
            'gas_imports': 'min'
        }
    
    df_norm = pd.DataFrame(index=df.index)
    for col, sentit in cols_objectius.items():
        if col not in df.columns:
            continue
        if sentit == 'min':
            df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
        else:
            df_norm[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min() + 1e-10)
    
    distancia = sum(df_norm[col]**2 for col in df_norm.columns) ** 0.5
    return distancia.idxmin()

# =============================================================================
# 1. EXECUTAR ESCENARIS DE COMPROMÍS PER CADA FRONT
# =============================================================================

resultats_compromis = {}
nivells_compromis = {}

for llindar, front in fronts_saturacio.items():
    print(f"\nProcessant front amb saturació = {etiquetes_saturacio[llindar]}...")
    
    # Afegir llindar específic a la configuració
    config_iter = config_base.copy()
    config_iter['llindar_desal_max'] = llindar    

    # Trobar escenari de compromís
    idx_compr = calcular_idx_compromis(front)
    row = front.loc[idx_compr]
  
    # Executar escenari
    result = executar_escenari(row, config_iter)
    
    nom = f'Sat_{etiquetes_saturacio[llindar]}'
    resultats_compromis[llindar] = result
    nivells_compromis[llindar] = result['level_final']
    
    print(f"  → min_level: {result['level_final'].min():.1f}%, "
          f"mean_level: {result['level_final'].mean():.1f}%")

print("\n✓ Tots els escenaris de compromís executats!")

# =============================================================================
# GRÀFIC 1: PERFILS TEMPORALS DELS ESCENARIS DE COMPROMÍS
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 10,
})

colors_saturacio = {
    1: '#d62728',  # Emergència - vermell
    2: '#ff7f0e',  # Excepcionalitat - taronja
    3: '#2ca02c',  # Alerta - verd
    4: '#1f77b4',  # Prealerta - blau
}

line_styles = {
    1: '-',
    2: '--',
    3: '-.',
    4: ':',
}

fig, ax = plt.subplots(figsize=(14, 7))

# Dades empíriques
empirical_data = (datos.df_pct_int_h * 100)['2016-01-01':]
ax.plot(empirical_data.index, empirical_data.values, 
        color='black', linewidth=2.5, linestyle='-',
        label='Dades empíriques', zorder=10)

# Escenaris de compromís per cada llindar de saturació
for llindar in sorted(nivells_compromis.keys()):
    series = nivells_compromis[llindar]
    ax.plot(series.index, series.values, 
            linewidth=1.8, 
            linestyle=line_styles[llindar], 
            color=colors_saturacio[llindar],
            alpha=0.85, 
            label=f'Compromís - Sat. {etiquetes_saturacio[llindar]}', 
            zorder=5)

ax.set_xlabel('Data', fontweight='bold')
ax.set_ylabel('Nivell embassaments conques internes (%)', fontweight='bold')
ax.set_title('Perfils de nivell segons llindar de saturació de dessalinització\n(Escenaris de compromís)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.legend(loc='lower left', frameon=True, framealpha=0.95, edgecolor='black')

plt.tight_layout()
plt.show()

# =============================================================================
# GRÀFIC 2: MITJANES DEL FRONT AMB RANG INTERQUARTÍLIC
# =============================================================================

# Calcular estadístiques per cada front
estadistiques_fronts = []

for llindar, front in fronts_saturacio.items():
    stats = {
        'llindar': llindar,
        'etiqueta': etiquetes_saturacio[llindar],
        'min_level_mean': front['min_level'].mean(),
        'min_level_q25': front['min_level'].quantile(0.25),
        'min_level_q75': front['min_level'].quantile(0.75),
        'mean_level_mean': front['mean_level'].mean(),
        'mean_level_q25': front['mean_level'].quantile(0.25),
        'mean_level_q75': front['mean_level'].quantile(0.75),
    }
    estadistiques_fronts.append(stats)

df_stats = pd.DataFrame(estadistiques_fronts)

# Gràfic
fig, ax = plt.subplots(figsize=(10, 6))

x = df_stats['llindar']
x_labels = df_stats['etiqueta']

# Nivell mínim (mitjana + rang interquartílic)
ax.plot(x, df_stats['min_level_mean'], 'o-', color='#d62728', 
        linewidth=2, markersize=8, label='Nivell mínim (mitjana)')
ax.fill_between(x, df_stats['min_level_q25'], df_stats['min_level_q75'], 
                color='#d62728', alpha=0.2, label='Nivell mínim (P25-P75)')

# Nivell mitjà (mitjana + rang interquartílic)
ax.plot(x, df_stats['mean_level_mean'], 's-', color='#1f77b4', 
        linewidth=2, markersize=8, label='Nivell mitjà (mitjana)')
ax.fill_between(x, df_stats['mean_level_q25'], df_stats['mean_level_q75'], 
                color='#1f77b4', alpha=0.2, label='Nivell mitjà (P25-P75)')

ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Llindar de saturació dessalinització', fontweight='bold')
ax.set_ylabel('Nivell dels embassaments (%)', fontweight='bold')
ax.set_title('Seguretat hídrica del front de Pareto segons llindar de saturació', 
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='best', frameon=True, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, 100)

plt.tight_layout()
plt.show()

# =============================================================================
# GRÀFIC 3: VALORS DE L'ESCENARI DE COMPROMÍS PER CADA LLINDAR
# =============================================================================

# Extreure mètriques dels escenaris de compromís
metriques_compromis = []

for llindar, result in resultats_compromis.items():
    metriques_compromis.append({
        'llindar': llindar,
        'etiqueta': etiquetes_saturacio[llindar],
        'min_level': result['level_final'].min(),
        'mean_level': result['level_final'].mean(),
    })

df_compr = pd.DataFrame(metriques_compromis)

# Gràfic
fig, ax = plt.subplots(figsize=(10, 6))

x = df_compr['llindar']
x_labels = df_compr['etiqueta']

# Nivell mínim
ax.plot(x, df_compr['min_level'], 'o-', color='#d62728', 
        linewidth=2.5, markersize=10, label='Nivell mínim')

# Nivell mitjà
ax.plot(x, df_compr['mean_level'], 's-', color='#1f77b4', 
        linewidth=2.5, markersize=10, label='Nivell mitjà')

ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Llindar de saturació dessalinització', fontweight='bold')
ax.set_ylabel('Nivell embassaments (%)', fontweight='bold')
ax.set_title('Seguretat hídrica de l\'escenari de compromís segons llindar de saturació', 
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='best', frameon=True, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, 100)

# Afegir valors numèrics als punts
for i, row in df_compr.iterrows():
    ax.annotate(f'{row["min_level"]:.1f}%', 
                (row['llindar'], row['min_level']), 
                textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    ax.annotate(f'{row["mean_level"]:.1f}%', 
                (row['llindar'], row['mean_level']), 
                textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)

plt.tight_layout()
plt.show()

# =============================================================================
# TAULA RESUM (opcional)
# =============================================================================

print("\n" + "=" * 80)
print(" TAULA RESUM: ESCENARIS DE COMPROMÍS PER LLINDAR DE SATURACIÓ")
print("=" * 80)

df_resum = df_compr[['etiqueta', 'min_level', 'mean_level']].copy()
df_resum.columns = ['Llindar Saturació', 'Nivell Mínim (%)', 'Nivell Mitjà (%)']
df_resum = df_resum.round(1)
print(df_resum.to_string(index=False))


#%%

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
# import os
# os.environ['OMP_NUM_THREADS'] = '6'
# =============================================================================
# 1. PREPARAR DADES
# =============================================================================
# Filtrar solucions amb alta resiliència
df_resilient = df_rgs_2040[
    (df_rgs_2040['min_level'] > 30) &  # Ajustar llindar segons necessitat
    (df_rgs_2040['mean_level'] > 60)
].copy()

print(f"Solucions resilients: {len(df_resilient)}")

# Variables de decisió per clustering
# features = ['potencia_solar', 'potencia_eolica', 'potencia_baterias', 
#             'max_desalation', 'midpoint_estimation', 'overflow_threshold_pct']

# Variables que realment varien al teu experiment
features = ['max_desalation','max_regen', 'midpoint_estimation', 'overflow_threshold',
            'min_run_hours', 'llindar_desal_max', 'llindar_regen_max',
            'L_emergencia', 'L_excepcionalitat', 'L_alerta', 'L_prealerta']

X = df_resilient[features].values

# Escalar (important per clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# 2. CLUSTERING AMB K-MEANS
# =============================================================================
# Determinar K òptim amb elbow method
inertias = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Visualitzar elbow
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertias, 'bo-')
ax.set_xlabel('Nombre de clusters (K)')
ax.set_ylabel('Inèrcia')
ax.set_title('Elbow Method')
plt.show()

# Aplicar K-means amb K escollit
K = 4  # Ajustar segons elbow
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df_resilient['cluster'] = kmeans.fit_predict(X_scaled)

# =============================================================================
# 3. ANALITZAR CLUSTERS
# =============================================================================
# Característiques mitjanes per cluster
cluster_summary = df_resilient.groupby('cluster').agg({
    'max_desalation': 'mean',
    'max_regen': 'mean',
    'midpoint_estimation': 'mean',
    'overflow_threshold': 'mean',
    'llindar_desal_max': 'mean',
    'llindar_regen_max': 'mean',
    'overflow_threshold': 'mean',
    'L_emergencia': 'mean',
    'L_excepcionalitat': 'mean',
    'L_alerta': 'mean',
    'L_prealerta': 'mean',
    # Variables resposta (per comparar)
    'restriction_days':'mean',
    'min_level': 'mean',
    'mean_level': 'mean',
    'gas_imports': 'mean',
    'total_costs': 'mean',
}).round(1)

print("\n=== RESUM CLUSTERS ===")
print(cluster_summary)

# =============================================================================
# 4. VISUALITZACIÓ 2D
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Solar vs Eòlica
scatter1 = axes[0].scatter(
    df_resilient['potencia_solar'] / 1000, 
    df_resilient['potencia_eolica'] / 1000,
    c=df_resilient['cluster'], 
    cmap='tab10', 
    alpha=0.6, 
    s=30
)
axes[0].set_xlabel('Potència solar (GW)')
axes[0].set_ylabel('Potència eòlica (GW)')
axes[0].set_title('Clusters: Solar vs Eòlica')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# Bateries vs Dessalinització
scatter2 = axes[1].scatter(
    df_resilient['potencia_baterias'] / 1000, 
    df_resilient['max_desalation'],
    c=df_resilient['cluster'], 
    cmap='tab10', 
    alpha=0.6, 
    s=30
)
axes[1].set_xlabel('Potència bateries (GW)')
axes[1].set_ylabel('Capacitat dessalinització (MW)')
axes[1].set_title('Clusters: Bateries vs Dessalinització')
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.savefig('clusters_resiliencia.png', dpi=150)
plt.show()

# =============================================================================
# 5. IDENTIFICAR ESTRATÈGIES
# =============================================================================
print("\n=== INTERPRETACIÓ ESTRATÈGIES ===")
for i in range(K):
    cluster_data = df_resilient[df_resilient['cluster'] == i]
    print(f"\nCluster {i} ({len(cluster_data)} solucions):")
    print(f"  Solar: {cluster_data['potencia_solar'].mean()/1000:.1f} GW")
    print(f"  Eòlica: {cluster_data['potencia_eolica'].mean()/1000:.1f} GW")
    print(f"  Bateries: {cluster_data['potencia_baterias'].mean()/1000:.1f} GW")
    print(f"  Dessalinització: {cluster_data['max_desalation'].mean():.0f} MW")
    print(f"  → min_level: {cluster_data['min_level'].mean():.1f}%")
    print(f"  → mean_level: {cluster_data['mean_level'].mean():.1f}%")
    
    
    
    
# from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=10)
df_resilient['cluster'] = dbscan.fit_predict(X_scaled)

# -1 = soroll (outliers)
print(f"Clusters trobats: {df_resilient['cluster'].nunique() - 1}")
print(f"Outliers: {(df_resilient['cluster'] == -1).sum()}")


#%%

# Top 5% més resilients
top_resilient = df_rgs_2040.nlargest(int(len(df_rgs_2040) * 0.05), 'min_level')

print("Característiques de les solucions més resilients:")
print(top_resilient[features].describe().round(1))


# Quines variables influeixen més en la resiliència?
correlacions = df_rgs_2040[features + ['min_level', 'mean_level']].corr()[['min_level', 'mean_level']]
print(correlacions.sort_values('min_level', ascending=False))


from sklearn.ensemble import RandomForestRegressor

X = df_rgs_2040[features]
y = df_rgs_2040['restriction_days']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("Importància de variables per min_level:")
print(importance)


# Comparar configuracions segons nivell de resiliència
df_rgs_2040['resiliencia_cat'] = pd.cut(df_rgs_2040['min_level'], 
                                        bins=[0, 30, 50, 70, 100], 
                                        labels=['Baixa', 'Mitjana', 'Alta', 'Molt alta'])

test = df_rgs_2040.groupby('resiliencia_cat')[features].mean().round(1)


#%%

fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance)))
bars = ax.barh(range(len(importance)), importance.values, color=colors)

# Afegir text dins o fora de les barres
for i, (bar, name, val) in enumerate(zip(bars, importance.index, importance.values)):
    if val > importance.max() * 0.3:  # Si la barra és prou llarga, text dins
        ax.text(val - 0.01, i, name, ha='right', va='center', fontsize=10, color='white', fontweight='bold')
    else:  # Si no, text fora
        ax.text(val + 0.01, i, name, ha='left', va='center', fontsize=10, color='black')

ax.set_yticks([])  # Amagar etiquetes Y (ja estan a les barres)
ax.set_xlabel('Importància', fontweight='bold')
# ax.set_title('Importància de variables per a la resiliència hídrica (min_level)', fontweight='bold')
ax.set_title('Importància de variables per a l\'impacte social (restriction_days)', fontweight='bold')
ax.set_xlim(0, importance.max() * 1.15)

plt.tight_layout()
plt.savefig('feature_importance_resiliencia.png', dpi=150, bbox_inches='tight')
plt.show()