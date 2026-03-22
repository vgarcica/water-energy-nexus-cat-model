# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 20:39:39 2026

@author: Víctor García Carrasco
"""

# Paràmetres fixes segons escenari 2030
config_base = {
    'nucleares_activas': [True, True, True],
    'potencia_cogeneracion': 542,
    'potencia_autoconsumo': 2181,
    'llindar_desal_max': 1,
    'demanda_electrica': 1.25,
    'max_desalation': 64,
    'max_regen': 0.250,
}

def executar_escenari(config_base):
    """Executa un escenari i retorna el resultat complet."""
    
    params = {
        'potencia_solar': 4971,
        'potencia_eolica': 6234,
        'potencia_baterias': 2234,
        
        # 'potencia_solar': 2453,
        # 'potencia_eolica': 6634,
        # 'potencia_baterias': 509,
              
        'min_run_hours': 6,
        'max_desalation': config_base.get('max_desalation', 64),
        'max_regen': 0.250,
        'midpoint_estimation': 95,
        'overflow_threshold': 95,
        'seasonal_phase': 0.0,
        'seasonal_amplitude': 0.0,
        'regen_base_pct': 0.5,
        'llindar_regen_max': 1,
        'derivada_nivell': 0.0,
        'x1_base_eme': 16,
        'x2_gap_exc': 9,
        'x3_gap_ale': config_base.get('x3_gap_ale', 20),
        # 'x3_gap_ale': 15,
        'x4_gap_pre': 20,
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
        llindar_activacio_desal_max=config_base['llindar_desal_max'],
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


# row = front_nsga.loc[idx_compromis]
result = executar_escenari(config_base)

#%%




# Paràmetres fixes segons escenari 2030
config_base = {
    'nucleares_activas': [False, False, False],
    # 'nucleares_activas': [True, True, True],
    'potencia_cogeneracion': 122.4,
    'potencia_autoconsumo': 5000,
    'llindar_desal_max': 1,
    'demanda_electrica': 1.5,
    'max_desalation': 64,
    'max_regen': 0.250,
}

def executar_escenari(config_base):
    """Executa un escenari i retorna el resultat complet."""
    
    params = {
        'potencia_solar': 17431,
        'potencia_eolica': 18439,
        'potencia_baterias': 4034,
        
             
        'min_run_hours': 12,
        'max_desalation': config_base.get('max_desalation', 64),
        'max_regen': 0.250,
        'midpoint_estimation': 70,
        'overflow_threshold': 80,
        'seasonal_phase': 0.0,
        'seasonal_amplitude': 0.0,
        'regen_base_pct': 0.5,
        'llindar_regen_max': 1,
        'derivada_nivell': 0.0,
        'x1_base_eme': 16,
        'x2_gap_exc': 19,
        'x3_gap_ale': 20,
        'x4_gap_pre': 15,
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
        llindar_activacio_desal_max=config_base['llindar_desal_max'],
        llindar_activacio_regen_max=params['llindar_regen_max'],
        k_deriv=params['derivada_nivell'],
        umbrales_sequia=umbrales,
        nucleares_activas=config_base['nucleares_activas'],
        potencia_cogeneracion=config_base['potencia_cogeneracion'],
        demanda_electrica=config_base['demanda_electrica'],
        CF_eolica_obj = None,
        usar_CF_automatic = True,
    )
    
    return result


# row = front_nsga.loc[idx_compromis]
result = executar_escenari(config_base)


result['hydro_metrics']['Restricciones escenario (días)']
result['hydro_metrics']['Llenado mínimo (%)']
result['hydro_metrics']['Llenado promedio (%)']
result['desalacion']
result['regeneracion']
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

# Crear figura
fig, ax = plt.subplots(figsize=(14, 7))

# Dades empíriques (línia destacada)
empirical_data = (datos.df_pct_int_h * 100)['2016-01-01':]
# empirical_data = hydro_base_level
ax.plot(empirical_data.index, empirical_data.values, 
        color='black', linewidth=1.75, linestyle='-',
        label='Dades històriques', zorder=10)

ax.plot(result['level_final'].index, result['level_final'].values, 
        color='red', linewidth=1.75, linestyle='--',
        label='Solució amb oportunisme fins prealerta', zorder=10)


# # Escenaris
# for i, (nom, config) in enumerate(escenaris_plot.items()):
#     if nom in nivells_escenaris:
#         series = nivells_escenaris[nom]
#         ax.plot(series.index, series.values, 
#                 linewidth=1.8, 
#                 linestyle=config['style'], 
#                 color=config['color'],
#                 alpha=0.85, 
#                 label=config['label'], 
#                 zorder=5-i)

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

# Configuració base
config_base = {
    'nucleares_activas': [True, True, True],
    'potencia_cogeneracion': 542,
    'potencia_autoconsumo': 2181,
    'demanda_electrica': 1.25,
    'max_desalation': 64,
    'max_regen': 0.250,
}

# Escenaris amb diferents llindars
escenaris_plot = {
    'llindar_1': {'llindar_desal_max': 1},
    'llindar_2': {'llindar_desal_max': 2},
    'llindar_3': {'llindar_desal_max': 3},
}

# Executar
resultats_saturacio = {}
nivells_saturacio = {}

for nom, config_extra in escenaris_plot.items():
    print(f"Executant escenari {nom}...")
    
    # Combinar config_base amb variació
    config = {**config_base, **config_extra}
    
    result = executar_escenari(config)
    
    resultats_saturacio[nom] = result
    nivells_saturacio[nom] = result['level_final']
    
    print(f"  → min_level: {result['level_final'].min():.1f}%, "
          f"mean_level: {result['level_final'].mean():.1f}%")

print("\n✓ Tots els escenaris executats!")



# Configuració dels escenaris a graficar
escenaris_plot = {
    'llindar_1': {'label': 'Llindar desal. = 1 (Prealerta)', 'style': '-', 'color': '#2ca02c'},
    'llindar_2': {'label': 'Llindar desal. = 2 (Alerta)', 'style': '--', 'color': '#1f77b4'},
    'llindar_3': {'label': 'Llindar desal. = 3 (Excepcionalitat)', 'style': ':', 'color': '#d62728'},
}


nivells_escenaris = nivells_saturacio

#%%

# =============================================================================
# CONFIGURACIÓ BASE (comuna a tots els escenaris)
# =============================================================================
config_base = {
    'nucleares_activas': [False, False, False],
    'potencia_cogeneracion': 122.4,
    'potencia_autoconsumo': 5000,
    'demanda_electrica': 1.5,
    'max_regen': 0.250,
    # 'max_regen': 0.173,
    # Potències renovables fixes
    'potencia_solar': 17431,
    'potencia_eolica': 18439,
    'potencia_baterias': 4034,
}

# =============================================================================
# DEFINICIÓ DELS 4 ESCENARIS
# =============================================================================
escenaris = {
    'dur_64': {
        'max_desalation': 64,
        'llindar_desal_max': 1,      # Prealerta
        'midpoint_estimation': 70,
        'min_run_hours': 12,
        'overflow_threshold': 90,
        'llindar_regen_max': 1,      # Prealerta
        # Llindars reforçats: Eme=16, Exc=35, Ale=55, Pre=70
        'x1_base_eme': 16,
        'x2_gap_exc': 19,            # 35-16
        'x3_gap_ale': 20,            # 50-35
        'x4_gap_pre': 15,            # 70-50
    },
    'dur_96': {
        'max_desalation': 96,
        'llindar_desal_max': 1,
        'midpoint_estimation': 70,
        'min_run_hours': 12,
        'overflow_threshold': 90,
        'llindar_regen_max': 1,
        'x1_base_eme': 16,
        'x2_gap_exc': 19,
        'x3_gap_ale': 20,
        'x4_gap_pre': 15,
    },
    'suau_64': {
        'max_desalation': 64,
        'llindar_desal_max': 2,      # Alerta
        'midpoint_estimation': 80,
        'min_run_hours': 6,
        'overflow_threshold': 85,
        'llindar_regen_max': 1,
        # Llindars normals: Eme=16, Exc=25, Ale=40, Pre=60
        'x1_base_eme': 16,
        'x2_gap_exc': 9,             # 25-16
        'x3_gap_ale': 15,            # 40-25
        'x4_gap_pre': 20,            # 60-40
    },
    'suau_96': {
        'max_desalation': 96,
        'llindar_desal_max': 2,
        'midpoint_estimation': 80,
        'min_run_hours': 6,
        'overflow_threshold': 85,
        'llindar_regen_max': 1,
        'x1_base_eme': 16,
        'x2_gap_exc': 9,
        'x3_gap_ale': 15,
        'x4_gap_pre': 20,
    },
    'eficient_96': {
        'max_desalation': 96,
        'llindar_desal_max': 2,
        'midpoint_estimation': 80,
        'min_run_hours': 6,
        'overflow_threshold': 85,
        'llindar_regen_max': 1,
        'x1_base_eme': 16,
        'x2_gap_exc': 9,
        'x3_gap_ale': 5,
        'x4_gap_pre': 20,
    },
    'actual_32': {
        'max_desalation': 32,
        'llindar_desal_max': 1,
        'midpoint_estimation': 80,
        'min_run_hours': 12,
        'overflow_threshold': 90,
        'llindar_regen_max': 1,
        'x1_base_eme': 16,
        'x2_gap_exc': 9,
        'x3_gap_ale': 15,
        'x4_gap_pre': 20,
        'max_regen': 0.155, #0.173,
        'nucleares_activas': [True, True, True],
        'potencia_cogeneracion': 546,
        'potencia_autoconsumo': 1381,
        'demanda_electrica': 1,
        'potencia_solar': 400,
        'potencia_eolica': 1400,
        'potencia_baterias': 500,        
    }, 
    

}

# =============================================================================
# FUNCIÓ EXECUTAR ESCENARI
# =============================================================================
def executar_escenari(config_base, escenari_params):
    """Executa un escenari combinant config_base amb paràmetres específics."""
    
    params = {**config_base, **escenari_params}
    
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
        
        # # ESCENARI HIDROLOGIC HISTORIC
        # df_energia_turbinada_mensual_internes=datos.energia_turbinada_mensual_internes,
        # df_energia_turbinada_mensual_ebre=datos.energia_turbinada_mensual_ebre,
        # df_nivel_si=hydro_base_level,
        # ESCENARI HIDROLOGIC PERSONALITZAT (descomentar si cal)
        df_energia_turbinada_mensual_internes=energia_turbinada_esc_int,
        df_energia_turbinada_mensual_ebre=energia_turbinada_esc_ebre,
        df_nivel_si=hydro_esc4_level,
        
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
        potencia_autoconsumo=params['potencia_autoconsumo'],
        min_run_hours=params['min_run_hours'],
        max_desalation=params['max_desalation'],
        midpoint_estimation=params['midpoint_estimation'],
        overflow_threshold_pct=params['overflow_threshold'],
        seasonal_phase_months=0.0,
        seasonal_desal_amplitude=0.0,
        max_regen=params['max_regen'],
        regen_base_pct=0.5,
        llindar_activacio_desal_max=params['llindar_desal_max'],
        llindar_activacio_regen_max=params['llindar_regen_max'],
        k_deriv=0.0,
        umbrales_sequia=umbrales,
        nucleares_activas=params['nucleares_activas'],
        potencia_cogeneracion=params['potencia_cogeneracion'],
        demanda_electrica=params['demanda_electrica'],
        CF_eolica_obj=None,
        usar_CF_automatic=True,
    )
    
    return result

# =============================================================================
# EXECUTAR TOTS ELS ESCENARIS
# =============================================================================
resultats = {}
nivells = {}

for nom, params in escenaris.items():
    print(f"Executant escenari {nom}...")
    result = executar_escenari(config_base, params)
    resultats[nom] = result
    nivells[nom] = result['level_final']
    print(f"  → min_level: {result['level_final'].min():.1f}%, "
          f"mean_level: {result['level_final'].mean():.1f}%, "
          f"dies restricció: {result['hydro_metrics']['Restricciones escenario (días)']}")

print("\n✓ Tots els escenaris executats!")

# =============================================================================
# CONFIGURACIÓ GRÀFICA
# =============================================================================
escenaris_plot = {
    'dur_64': {'label': 'Seguretat (160 hm³/y)', 'style': '-', 'color': '#d62728'},
    'dur_96': {'label': 'Seguretat (240 hm³/y)', 'style': '--', 'color': '#d62728'},
    'suau_64': {'label': 'Oportunista (160 hm³/y)', 'style': '-', 'color': '#2ca02c'},
    'suau_96': {'label': 'Oportunista (240 hm³/y)', 'style': '--', 'color': '#2ca02c'},
    'eficient_96': {'label': 'Eficient (240 hm³/y)', 'style': '--', 'color': '#1f77b4'},
    'actual_32': {'label': 'Actual (80 hm³/y)', 'style': ':', 'color': '#1f77b4'},
}

# =============================================================================
# GRÀFIC
# =============================================================================
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

fig, ax = plt.subplots(figsize=(14, 7))

# Dades empíriques
empirical_data = (datos.df_pct_int_h * 100)['2016-01-01':]
ax.plot(empirical_data.index, empirical_data.values, 
        color='black', linewidth=1.75, linestyle='-',
        label='Dades històriques', zorder=10)

# Escenaris
for nom, config in escenaris_plot.items():
    if nom in nivells:
        series = nivells[nom]
        # Afegir dies restricció a la llegenda
        dies_restr = resultats[nom]['hydro_metrics']['Restricciones escenario (días)']
        ax.plot(series.index, series.values, 
                linewidth=1.8, 
                linestyle=config['style'], 
                color=config['color'],
                alpha=0.85, 
                # label=f"{config['label']} ({dies_restr} dies restr.)", 
                label=f"{config['label']}", 
                zorder=5)

# Personalització
ax.set_xlabel('Data', fontweight='bold')
ax.set_ylabel('Nivell embassaments conques internes (%)', fontweight='bold')
ax.set_title('Escenari Històric', 
# ax.set_title('Sequera Estival Recurrent',
# ax.set_title('Escenari Sec amb Torrencialitat', 
# ax.set_title('Sequera Plurianual Agreujada',              
# ax.set_title('Sequera Plurianual Extrema',              
#ax.set_title('Sequera Plurianual Extrema - (2023 duplicat i reduit al 50%)',              
             fontsize=16, fontweight='bold', pad=15)

ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

legend = ax.legend(loc='lower left', frameon=True, shadow=True, 
                   fancybox=True, framealpha=0.95, edgecolor='black',
                   ncol=2)
legend.get_frame().set_linewidth(1)

plt.tight_layout()
plt.savefig('comparacio_dur_suau.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# TAULA RESUM
# =============================================================================
resum = []
for nom in escenaris.keys():
    result = resultats[nom]
    resum.append({
        'Escenari': nom,
        'Nivell mínim (%)': round(result['level_final'].min(), 1),
        'Nivell mitjà (%)': round(result['level_final'].mean(), 1),
        'Restriccions (dies)': result['hydro_metrics']['Restricciones escenario (días)'],
        'Aigua dessalada (hm³)': round(result['desal_final_hm3'].sum(), 1),
        'Aigua regenerada (hm³)': round(result['regen_final'].sum(), 1),
    })

df_resum = pd.DataFrame(resum).set_index('Escenari')
print("\n" + "="*80)
print("TAULA RESUM D'ESCENARIS")
print("="*80)
print(df_resum.to_string())


# Format amb separadors
print("\n" + "="*90)
print(f"{'ESCENARI':<12} {'L_min (%)':<12} {'L_mean (%)':<12} {'Restr (dies)':<14} {'Restr (hm³)':<14} {'Desal (hm³)':<14} {'Regen (hm³)':<12}")
print("-"*90)
for nom in escenaris.keys():
    r = resultats[nom]
    print(f"{nom:<12} {r['level_final'].min():>10.1f} {r['level_final'].mean():>12.1f} "
          f"{r['hydro_metrics']['Restricciones escenario (días)']:>12} "
          f"{r['savings_final'].sum():>12.1f} {r['desal_final_hm3'].sum():>13.1f} {r['regen_final'].sum():>12.1f} {r['hydro_metrics']['Regen. i Dessal. en dèficit (MWh)'].sum()/1000:>12.1f}"
          f"{r['hydro_metrics']['Factor oportunista (%)']:>12.1f} {round(r['hydro_metrics']['Arrancades']/9):>8}")
print("="*90)


#%%
# Configurar estilo académico
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
})

fig, ax = plt.subplots(figsize=(14, 6))

# Dades històriques
historic = datos.dessalacio_diaria.resample('ME').sum()
ax.plot(historic.index, historic.values, 
        color='black', linewidth=2, linestyle='-',
        label='Dades històriques', zorder=10)

# Dades del model
model = resultats['actual_32']['desal_final_hm3'].resample('ME').sum()[:-1]
ax.plot(model.index, model.values, 
        color='#1f77b4', linewidth=1.8, linestyle='--',
        label='Model (80 hm³/y)', alpha=0.85, zorder=5)

# Personalització
ax.set_xlabel('Data', fontweight='bold')
ax.set_ylabel('Aigua dessalinitzada (hm³/mes)', fontweight='bold')
ax.set_title('Validació: Dessalinització històrica vs Model', fontweight='bold', pad=15)

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.legend(loc='upper left', frameon=True, shadow=True, framealpha=0.95)

# Estadístiques de comparació
print(f"Total històric: {historic.sum():.1f} hm³")
print(f"Total model: {model.sum():.1f} hm³")
print(f"Diferència: {((model.sum() / historic.sum()) - 1) * 100:.1f}%")

plt.tight_layout()
plt.savefig('validacio_dessalinitzacio.png', dpi=150, bbox_inches='tight')
plt.show()

#%%
# Configuració base
config_base = {
    'nucleares_activas': [True, True, True],
    'potencia_cogeneracion': 542,
    'potencia_autoconsumo': 2181,
    'demanda_electrica': 1.25,
    'max_desalation': 64,
    'max_regen': 0.250,
    'llindar_desal_max': 2,
}

# Escenaris amb diferents gaps d'alerta
escenaris_gap = {
    'gap_20': {'x3_gap_ale': 20},
    'gap_15': {'x3_gap_ale': 15},
    'gap_10': {'x3_gap_ale': 10},
}

# Executar
resultats_gap = {}
nivells_gap = {}
for nom, config_extra in escenaris_gap.items():
    print(f"Executant escenari {nom}...")
    config = {**config_base, **config_extra}
    result = executar_escenari(config)
    resultats_gap[nom] = result
    nivells_gap[nom] = result['level_final']
    print(f"  → min_level: {result['level_final'].min():.1f}%, "
          f"mean_level: {result['level_final'].mean():.1f}%", ""
          f"dies restricció: {result['hydro_metrics']['Restricciones escenario (días)']}")
print("\n✓ Tots els escenaris executats!")

# Configuració gràfic
escenaris_plot = {
    'gap_20': {'label': 'Gap alerta = 20% (L_alerta=45%)', 'style': '-', 'color': '#2ca02c'},
    'gap_15': {'label': 'Gap alerta = 15% (L_alerta=40%)', 'style': '--', 'color': '#1f77b4'},   
    'gap_10': {'label': 'Gap alerta = 10% (L_alerta=35%)', 'style': ':', 'color': '#d62728'},
}

escenaris_plot = {
    'gap_20': {'label': f"Gap alerta = 20% (L_alerta=45%) - {resultats_gap['gap_20']['hydro_metrics']['Restricciones escenario (días)']} dies", 'style': ':', 'color': '#d62728'},
    'gap_15': {'label': f"Gap alerta = 15% (L_alerta=40%) - {resultats_gap['gap_15']['hydro_metrics']['Restricciones escenario (días)']} dies", 'style': '--', 'color': '#1f77b4'},   
    'gap_10': {'label': f"Gap alerta = 10% (L_alerta=35%) - {resultats_gap['gap_10']['hydro_metrics']['Restricciones escenario (días)']} dies", 'style': '-', 'color': '#2ca02c'},
}

nivells_escenaris = nivells_gap

#%%

# Configuració base
config_base = {
    'nucleares_activas': [True, True, True],
    'potencia_cogeneracion': 542,
    'potencia_autoconsumo': 2181,
    'demanda_electrica': 1.25,
    'max_regen': 0.250,
    'llindar_desal_max': 2,
}

# Escenaris amb diferents capacitats de dessalinització
escenaris_desal = {
    'desal_32': {'max_desalation': 32},
    'desal_64': {'max_desalation': 64},
    'desal_96': {'max_desalation': 96},
}

# Executar
resultats_desal = {}
nivells_desal = {}
for nom, config_extra in escenaris_desal.items():
    print(f"Executant escenari {nom}...")
    config = {**config_base, **config_extra}
    result = executar_escenari(config)
    resultats_desal[nom] = result
    nivells_desal[nom] = result['level_final']
    print(f"  → min_level: {result['level_final'].min():.1f}%, "
          f"mean_level: {result['level_final'].mean():.1f}%")
print("\n✓ Tots els escenaris executats!")

# Configuració gràfic
escenaris_plot = {
    'desal_32': {'label': 'Capacitat desal. = 32 MW (80hm$^3$ anuals)', 'style': ':', 'color': '#d62728'},
    'desal_64': {'label': 'Capacitat desal. = 64 MW (160hm$^3$ anuals)', 'style': '--', 'color': '#1f77b4'},
    'desal_96': {'label': 'Capacitat desal. = 96 MW (240hm$^3$ anuals)', 'style': '-', 'color': '#2ca02c'},
}

nivells_escenaris = nivells_desal

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

# Configuración de parámetros
PARAMS = {
    'max_desal': 80,  # hm³
    'llindar_alerta': 45,  # %
    'llindar_prealerta': 65,  # % (si aplica)
    'min_run_hours': 6  # h (si aplica)
}


# Crear figura
fig, ax = plt.subplots(figsize=(14, 7))

# Dades empíriques (línia destacada)
empirical_data = (datos.df_pct_int_h * 100)['2016-01-01':]
ax.plot(empirical_data.index, empirical_data.values, 
        color='black', linewidth=1.75, linestyle='-',
        label='Dades històriques', zorder=10)

# Escenaris
for nom, config in escenaris_plot.items():
    if nom in nivells_escenaris:
        series = nivells_escenaris[nom]
        ax.plot(series.index, series.values, 
                linewidth=1.8, 
                linestyle=config['style'], 
                color=config['color'],
                alpha=0.85, 
                label=config['label'], 
                zorder=5)

# Personalització
ax.set_xlabel('Data', fontweight='bold')
ax.set_ylabel('Nivell embassaments conques internes (%)', fontweight='bold')
ax.set_title(f'Efecte del llindar de saturació en la dessalinització\n'
             f'(Màxim anual: 80 hm$^3$, Llindar d\'alerta: 45%)', 
             fontsize=16, fontweight='bold', pad=15)
ax.set_title('Efecte del llindar d\'alerta\n'
             f'(Màxim anual: 80 hm$^3$, Llindar de saturació: Alerta)', 
             fontsize=16, fontweight='bold', pad=15)
# ax.set_title('Efecte de la capacitat de dessalinització\n'
#              f'(Llindar de saturació: Alerta, Llindar d\'alerta: 45%)', 
#              fontsize=16, fontweight='bold', pad=15)

# # Subtítulo con parámetros
# param_str = (f"Màxim dessalinització: {PARAMS['max_desal']} hm$^3$/any | "
#              f"Llindar d'alerta: {PARAMS['llindar_alerta']}%")
# fig.suptitle(param_str, y=0.92, fontsize=12, style='italic', color='gray')

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
plt.savefig('comparacio_llindars_desal.png', dpi=150, bbox_inches='tight')
plt.show()


#%%

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# 1. Filtrar dades
df_filtered = df_rgs_2050[df_rgs_2050['potencia_baterias'] == 4000].copy()
print(f"Punts després del filtre: {len(df_filtered)}")

# 2. Crear graella regular
xi = np.linspace(df_filtered['potencia_solar'].min(), df_filtered['potencia_solar'].max(), 100)
yi = np.linspace(df_filtered['potencia_eolica'].min(), df_filtered['potencia_eolica'].max(), 100)
Xi, Yi = np.meshgrid(xi, yi)

# 3. Interpolar
Zi = griddata(
    points=(df_filtered['potencia_solar'].values, df_filtered['potencia_eolica'].values),
    # values=df_filtered['gas_imports'].values - df_filtered['deficits_total'].values,
    values=df_filtered['total_costs'].values,
    xi=(Xi, Yi),
    method='nearest'
)

# 4. Visualitzar
fig, ax = plt.subplots(figsize=(10, 8))
heatmap = ax.pcolormesh(Xi / 1000, Yi / 1000, Zi / 9e6, cmap='viridis_r', shading='auto')
plt.colorbar(heatmap, label='Gas + Imports (TWh/y)')

ax.set_xlabel('Potència solar (GW)', fontweight='bold')
ax.set_ylabel('Potència eòlica (GW)', fontweight='bold')
ax.set_title('Residu tèrmic vs capacitat renovable (Bateries = 4 GW)', fontweight='bold')

plt.tight_layout()
plt.savefig('heatmap_gas_imports.png', dpi=150, bbox_inches='tight')
plt.show()

#%%
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# =============================================================================
# CONFIGURACIÓ
# =============================================================================
bateries_gw = 4  # GW
tolerancia_gw = 0.5  # ± tolerància per al filtre

x_range = (0, 20)  # GW solar
y_range = (5, 20)  # GW eòlica


# Coordenades del punt (en MW)
proencat_solar =  22431 - 5000# MW
proencat_eolica = 18439  # MW

# =============================================================================
# FILTRATGE I INTERPOLACIÓ
# =============================================================================
# df_filtered = df_rgs_2050[
#     (df_rgs_2050['potencia_baterias'] >= (bateries_gw - tolerancia_gw) * 1000) & 
#     (df_rgs_2050['potencia_baterias'] <= (bateries_gw + tolerancia_gw) * 1000)
# ].copy()
df_filtered = df_rgs_2050[df_rgs_2050['potencia_baterias'] == bateries_gw * 1000].copy()

print(f"Punts després del filtre: {len(df_filtered)}")

# Crear graella regular
xi = np.linspace(x_range[0] * 1000, x_range[1] * 1000, 100)
yi = np.linspace(y_range[0] * 1000, y_range[1] * 1000, 100)
Xi, Yi = np.meshgrid(xi, yi)

# Interpolar
Zi = griddata(
    points=(df_filtered['potencia_solar'].values, df_filtered['potencia_eolica'].values),
    values=df_filtered['gas_imports'].values - df_filtered['deficits_total'].values,
    xi=(Xi, Yi),
    method='linear'
)

# Omplir NaN amb nearest
Zi_nearest = griddata(
    points=(df_filtered['potencia_solar'].values, df_filtered['potencia_eolica'].values),
    values=df_filtered['gas_imports'].values - df_filtered['deficits_total'].values,
    xi=(Xi, Yi),
    method='nearest'
)

# Combinar: cubic on existeix, nearest als extrems
Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)

# =============================================================================
# VISUALITZACIÓ
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
heatmap = ax.pcolormesh(Xi / 1000, Yi / 1000, Zi / 9e6, cmap='viridis_r', shading='auto')
plt.colorbar(heatmap, label='Residu tèrmic (TWh/y)')

# Afegir punt al gràfic (recorda el +5 GW de l'offset a l'eix X)
ax.scatter((proencat_solar / 1000), proencat_eolica / 1000, 
           color='red', s=100, marker='x', edgecolor='white', linewidth=1.5, zorder=10)

ax.annotate('PROENCAT 2040', 
            xy=((proencat_solar / 1000), proencat_eolica / 1000),
            xytext=(-100, 10), textcoords='offset points',
            fontsize=10, fontweight='bold', color='black',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))


ax.set_xlabel('Potència solar (GW) (Autoconsum = 5 GW)', fontweight='bold')
ax.set_ylabel('Potència eòlica (GW)', fontweight='bold')
ax.set_title(f'Residu tèrmic vs capacitat renovable (Bateries = {bateries_gw} GW)', fontweight='bold')

ax.set_xlim(x_range)
ax.set_ylim(y_range)

from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x + 5)}'))

# Després de crear el gràfic, afegeix:
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xticks(range(x_range[0], x_range[1] + 1, 5))  # Cada 5 GW
ax.set_yticks(range(y_range[0], y_range[1] + 1, 5))

plt.tight_layout()
plt.savefig(f'heatmap_gas_imports_bat{bateries_gw}GW.png', dpi=150, bbox_inches='tight')
plt.show()
