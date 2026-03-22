# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 16:51:09 2026

@author: Víctor García Carrasco
"""

# Diccionari de metadades per cada variable
VARIABLES_CONFIG = {
    'total_costs':      {'unitat': 'M€/y',  'escala': 1e-6/9,  'label': 'Costos'},
    'gas_imports':      {'unitat': 'TWh/y', 'escala': 1e-6/9,  'label': 'Residu tèrmic'},
    'min_level':        {'unitat': '%',   'escala': 1,     'label': 'Nivell mínim'},
    'mean_level':       {'unitat': '%',   'escala': 1,     'label': 'Nivell mitjà'},
    'squared_dev':      {'unitat': '%²',  'escala': 1,     'label': 'Desviació quadràtica mitjana'},
    'restriction_days': {'unitat': 'dies','escala': 1,     'label': 'Dies restricció'},
    'restriction_ratio': {'unitat': 'dies/hm³','escala': 1,     'label': 'Ratio'},
    'restriction_savings': {'unitat': 'hm³','escala': 1,     'label': 'Aigua estalviada'},    
    'max_desalation':   {'unitat': 'MW',  'escala': 1,     'label': 'Potència dessalinització'},
    'desal_cf':         {'unitat': '%',   'escala': 1,     'label': 'FC dessalinització'},
    'desal_hm3':        {'unitat': 'hm³', 'escala': 1,     'label': 'Volum dessalat'},
    'spillage_hm3':     {'unitat': 'hm³', 'escala': 1,     'label': 'Vessaments'},
    'surpluses_total':  {'unitat': 'TWh', 'escala': 1e-6,  'label': 'Excedents'},
    'potencia_solar':   {'unitat': 'GW',  'escala': 1e-3,  'label': 'Potència solar'},
    'potencia_eolica':  {'unitat': 'GW',  'escala': 1e-3,  'label': 'Potència eòlica'},
    'regen_hm3':        {'unitat': 'hm³', 'escala': 1,     'label': 'Volum regenerat'},
    'regen_max':        {'unitat': 'hm³/dia', 'escala': 1,     'label': 'Capacitat regeneració'},
    'llindar_desal_max':        {'unitat': '%', 'escala': 1,     'label': 'Saturació operativa dessalinització'},
    'overflow_threshold':        {'unitat': '%', 'escala': 1,     'label': 'Llindar sobreeiximent'},
    'midpoint_estimation': {'unitat': '%', 'escala': 1,     'label': 'Inflexió funció dessalinització'},
    'L_prealerta': {'unitat': '%', 'escala': 1,     'label': 'Llindar de prealerta'},
    'L_alerta': {'unitat': '%', 'escala': 1,     'label': 'Llindar de alerta'},
}

def get_label(var):
    """Retorna etiqueta amb unitats."""
    cfg = VARIABLES_CONFIG.get(var, {'label': var, 'unitat': ''})
    return f"{cfg['label']} ({cfg['unitat']})"

def get_scaled(df, var):
    """Retorna sèrie escalada."""
    cfg = VARIABLES_CONFIG.get(var, {'escala': 1})
    return df[var] * cfg['escala']


def plot_pareto_front_2d(df_all, df_pareto, obj1, obj2, dir1, dir2):
    """Plot 2D amb unitats i escalat automàtic."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Escalar dades
    x_all = get_scaled(df_all, obj1)
    y_all = get_scaled(df_all, obj2)
    x_pareto = get_scaled(df_pareto, obj1)
    y_pareto = get_scaled(df_pareto, obj2)
    
    # Plot
    ax.scatter(x_all, y_all, c='lightgray', alpha=0.3, s=10, label='Tots')
    ax.scatter(x_pareto, y_pareto, c='red', alpha=0.7, s=30, label='Pareto')
    
    # Etiquetes amb unitats
    ax.set_xlabel(get_label(obj1))
    ax.set_ylabel(get_label(obj2))
    
    # Títol
    ax.set_title(f'{get_label(obj1)} vs {get_label(obj2)}')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_pareto_front_3d(df_all, df_pareto, obj1, obj2, obj3, dir1, dir2, dir3):
    """Plot 3D amb unitats i escalat automàtic."""
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Escalar dades
    x_all = get_scaled(df_all, obj1)
    y_all = get_scaled(df_all, obj2)
    z_all = get_scaled(df_all, obj3)
    x_pareto = get_scaled(df_pareto, obj1)
    y_pareto = get_scaled(df_pareto, obj2)
    z_pareto = get_scaled(df_pareto, obj3)
    
    # Plot
    ax.scatter(x_all, y_all, z_all, c='lightgray', alpha=0.1, s=5, label='Tots')
    ax.scatter(x_pareto, y_pareto, z_pareto, c='red', alpha=0.7, s=20, label='Pareto')
    
    # Etiquetes amb unitats
    ax.set_xlabel(get_label(obj1))
    ax.set_ylabel(get_label(obj2))
    ax.set_zlabel(get_label(obj3))
    
    ax.set_title(f'Front de Pareto 3D')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_pareto_3d_plotly(df_all, df_pareto, obj1, obj2, obj3, dir1, dir2, dir3):
    """Plot 3D interactiu amb Plotly, unitats i escalat automàtic."""
    import plotly.graph_objects as go
    
    # Escalar dades
    x_pareto = get_scaled(df_pareto, obj1)
    y_pareto = get_scaled(df_pareto, obj2)
    z_pareto = get_scaled(df_pareto, obj3)
    
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x_pareto, y=y_pareto, z=z_pareto,
            mode='markers',
            marker=dict(size=5, color=z_pareto, colorscale='Viridis', opacity=0.8),
            text=[f'{obj1}: {x:.2f}<br>{obj2}: {y:.2f}<br>{obj3}: {z:.2f}' 
                  for x, y, z in zip(x_pareto, y_pareto, z_pareto)],
            hoverinfo='text'
        )
    ])
    
    fig.update_layout(
        title='Front de Pareto 3D (interactiu)',
        scene=dict(
            xaxis_title=get_label(obj1),
            yaxis_title=get_label(obj2),
            zaxis_title=get_label(obj3),
        ),
        width=900,
        height=700,
    )
    
    fig.show()

def extract_pareto_energy_scenarios(df_results):
    """
    Extrae soluciones no dominadas para el problema del nexe aigua-energia.
    """
    
    scenarios_2D = {
        # =====================================================================
        # TRADE-OFFS PRINCIPALS (2D)
        # =====================================================================
        
        # --- Cost vs Seguretat ---
        "cost_vs_nivell_min": {
            "objectives": ["total_costs", "min_level"],
            "directions": ["min", "max"],
            "description": "Cost vs nivell mínim"
        },
        "cost_vs_nivell_mitja": {
            "objectives": ["total_costs", "mean_level"],
            "directions": ["min", "max"],
            "description": "Cost vs nivell mitjà"
        },
        "cost_vs_msd": {
            "objectives": ["total_costs", "squared_dev"],
            "directions": ["min", "min"],
            "description": "Cost vs desviació quadràtica mitjana (MSD)"
        },
        
        "cost_vs_restriccions": {
            "objectives": ["total_costs", "restriction_savings"],
            "directions": ["min", "min"],
            "description": "Cost vs restriccions"
        },
        
        # --- Descarbonització ---
        "residu_vs_nivell_min": {
            "objectives": ["gas_imports", "min_level"],
            "directions": ["min", "max"],
            "description": "Residu tèrmic vs nivell mínim"
        },
        # "regen_vs_maxregen": {
        #     "objectives": ["regen_hm3", "max_regen"],
        #     "directions": ["min", "min"],
        #     "description": "Regeneració vs Capacitat de regeneració"
        # },
        
        "residu_vs_cost": {
            "objectives": ["total_costs","gas_imports"],
            "directions": ["min", "min"],
            "description": "Cost vs residu tèrmic"
        },
        
        # --- Dessalinització ---
        "desal_vs_nivell_min": {
            "objectives": ["max_desalation", "min_level"],
            "directions": ["min", "max"],
            "description": "Potència dessalinització vs nivell mínim"
        },
        # "desal_vs_nivell_mean": {
        #     "objectives": ["max_desalation", "mean_level"],
        #     "directions": ["min", "max"],
        #     "description": "Potència dessalinització vs nivell mitjà"
        # },
        "regen_vs_nivell_min": {
            "objectives": ["max_regen", "min_level"],
            "directions": ["min", "max"],
            "description": "Capacitat regeneració vs nivell mínim"
        },
        "solar_vs_nivell_min": {
            "objectives": ["potencia_solar", "min_level"],
            "directions": ["min", "max"],
            "description": "Solar vs nivell mínim"
        },
        "eolica_vs_nivell_min": {
            "objectives": ["potencia_eolica", "min_level"],
            "directions": ["min", "max"],
            "description": "Eòlica vs nivell mínim"
        },
       

        "saturacio_desal_vs_nivell_mean": {
            "objectives": ["llindar_desal_max", "mean_level"],
            "directions": ["min", "max"],
            "description": "Cost vs saturació desal."
        },
        
        
        "sobreeiximent_vs_nivell_mean": {
            "objectives": ["overflow_threshold", "mean_level"],
            "directions": ["max", "max"],
            "description": "Seguretat hídrica vs nivell de sobreeiximent"
        },
        
        "sobreeiximent_vs_desal_cf": {
            "objectives": ["overflow_threshold", "desal_cf"],
            "directions": ["max", "max"],
            "description": "Factor de capacitat vs nivell de sobreeiximent"
        },

        "alerta_vs_savings": {
            "objectives": ["L_alerta", "restriction_days"],
            "directions": ["max", "min"],
            "description": "Restriccions vs Llindar d''alerta"
        },         
        
        "inflexio_vs_nivell_mean": {
            "objectives": ["midpoint_estimation", "mean_level"],
            "directions": ["min", "max"],
            "description": "Seguretat hídrica vs inflexió funció dessalinització"
        },
        
        "inflexio_vs_desal_cf": {
            "objectives": ["midpoint_estimation", "desal_cf"],
            "directions": ["min", "max"],
            "description": "Factor de capacitat vs inflexió funció dessalinització"
        },
        
        "prealerta_vs_nivell_mean": {
            "objectives": ["L_prealerta", "mean_level"],
            "directions": ["min", "max"],
            "description": "Seguretat hídrica vs llindar de prealerta"
        },

        "alerta_vs_nivell_mean": {
            "objectives": ["L_alerta", "mean_level"],
            "directions": ["min", "max"],
            "description": "Seguretat hídrica vs llindar de alerta"
        },
        
        # "desal_vs_spillage": {
        #     "objectives": ["max_desalation", "spillage_hm3"],
        #     "directions": ["max", "min"],
        #     "description": "Potència dessalinització vs vessaments"
        # },        
        
        # "desal_vs_restriccions": {
        #     "objectives": ["max_desalation", "restriction_days"],
        #     "directions": ["min", "min"],
        #     "description": "Potència dessalinització vs restriccions"
        # },
        # "desal_cf_vs_maxdesal": {
        #     "objectives": ["desal_cf", ",max_desalation"],
        #     "directions": ["max", "min"],
        #     "description": "Factor capacitat vs Potència dessalinització"
        # },        
        # "desal_cf_vs_cost": {
        #     "objectives": ["desal_cf", "total_costs"],
        #     "directions": ["max", "min"],
        #     "description": "Factor capacitat dessalinització vs cost"
        # },
        # "desal_cf_vs_volum": {
        #     "objectives": ["desal_cf", "desal_hm3"],
        #     "directions": ["max", "max"],
        #     "description": "Factor de capacitat vs volum dessalat"
        # },
    }
        
        # --- Robustesa hídrica (2D) ---
        # "nivell_min_vs_mitja": {
        #     "objectives": ["min_level", "mean_level"],
        #     "directions": ["max", "max"],
        #     "description": "Nivell mínim vs mitjà"
        # },
        # "nivell_min_vs_msd": {
        #     "objectives": ["min_level", "squared_dev"],
        #     "directions": ["max", "min"],
        #     "description": "Nivell mínim vs variabilitat"
        # },
        # "restriccions_vs_vessaments": {
        #     "objectives": ["restriction_days", "spillage_hm3"],
        #     "directions": ["min", "min"],
        #     "description": "Restriccions vs vessaments"
        # },
        
    scenarios_3D = {        
        # =====================================================================
        # TRADE-OFFS MULTIOBJECTIU (3D)
        # =====================================================================
        
        # --- Triangle principal del nexe ---
        # "nexe_principal": {
        #     "objectives": ["total_costs", "squared_dev", "gas_imports"],
        #     "directions": ["min", "min", "min"],
        #     "description": "Cost vs seguretat vs residu tèrmic"
        # },
        # "nexe_nivell_min": {
        #     "objectives": ["total_costs", "mean_level", "gas_imports"],
        #     "directions": ["min", "max", "min"],
        #     "description": "Cost vs nivell mitjà vs residu tèrmic"
        # },
        
        "nexe_nivell_mean": {
            "objectives": ["total_costs",  "gas_imports", "mean_level"],
            "directions": ["min", "min", "max"],
            "description": "Cost vs nivell mitjà vs residu tèrmic"
        },        
        
        "nexe_msd": {
            "objectives": ["total_costs",  "gas_imports", "squared_dev"],
            "directions": ["min", "min", "min"],
            "description": "Cost vs nivell mitjà vs residu tèrmic"
        },           
        
        # --- Seguretat completa ---
        # "seguretat_completa": {
        #     "objectives": ["min_level", "restriction_days", "spillage_hm3"],
        #     "directions": ["max", "min", "min"],
        #     "description": "Nivell mínim vs restriccions vs vessaments"
        # },
        # "robustesa_hidrica": {
        #     "objectives": ["min_level", "mean_level", "squared_dev"],
        #     "directions": ["max", "max", "min"],
        #     "description": "Nivell mínim vs mitjà vs variabilitat"
        # },
        
        #-----------------------------------------------
        
        # # --- Sinergies renovables-aigua ---
        # "renovables_vs_seguretat": {
        #     "objectives": ["gas_imports", "min_level", "surpluses_total"],
        #     "directions": ["min", "max", "min"],
        #     "description": "Residu tèrmic vs seguretat vs excedents"
        # },
        
        # # --- Eficiència dessalinització ---
        # "eficiencia_desal": {
        #     "objectives": ["desal_cf", "desal_hm3", "total_costs"],
        #     "directions": ["max", "max", "min"],
        #     "description": "FC dessalinització vs volum vs cost"
        # },
        
        # # --- Triple bottom line ---
        # "triple_sostenibilitat": {
        #     "objectives": ["total_costs", "desal_hm3", "restriction_days"],
        #     "directions": ["min", "min", "min"],
        #     "description": "Cost vs volum dessalat vs impacte social"
        # },
        
        "seguretat_hidrica1_renw": {
            "objectives": ["potencia_solar", "potencia_eolica", "mean_level"],
            "directions": ["min", "min", "max"],
            "description": "Inversió solar vs eòlica vs nivell mitjà"
        },
        
        # "seguretat_hidrica2_renw": {
        #     "objectives": ["potencia_solar", "potencia_eolica", "min_level"],
        #     "directions": ["min", "min", "max"],
        #     "description": "Inversió solar vs eòlica vs nivell mitjà"
        # },        
        # # # --- Inversió renovable ---
        # "inversio_renovable": {
        #     "objectives": ["potencia_solar", "potencia_eolica", "gas_imports"],
        #     "directions": ["min", "min", "min"],
        #     "description": "Inversió solar vs eòlica vs residu tèrmic"
        # },
    }
    
    scenarios = scenarios_3D
    results = {}
    
    for name, config in scenarios.items():
        # Verificar que les columnes existeixen
        if not all(obj in df_results.columns for obj in config["objectives"]):
            print(f"⚠ Saltant '{name}': columnes no trobades")
            continue
        
        pareto_solutions = extract_pareto_front(
            df_results, 
            config["objectives"], 
            config["directions"]
        )
        
        n_obj = len(config["objectives"])
        print(f"[{n_obj}D] {config['description']}: {len(pareto_solutions)} solucions Pareto ({(1 - len(pareto_solutions)/len(df_results))*100:.1f}% reducció)")
        
        results[name] = {
            "pareto_df": pareto_solutions,
            "config": config,
            "reduction_pct": (1 - len(pareto_solutions)/len(df_results))*100
        }
    
    return results


def plot_pareto_panel_2d(df_all, pareto_results, scenarios_2d=None, ncols=3, figsize_per_plot=(5, 4)):
    """
    Panell de gràfics 2D per múltiples fronts de Pareto.
    
    Args:
        df_all: DataFrame amb tots els escenaris
        pareto_results: Diccionari retornat per extract_pareto_energy_scenarios
        scenarios_2d: Llista de noms d'escenaris a mostrar (None = tots els 2D)
        ncols: Nombre de columnes del panell
        figsize_per_plot: Mida de cada subplot
    """
    
    # Filtrar escenaris 2D
    if scenarios_2d is None:
        scenarios_2d = [name for name, data in pareto_results.items() 
                       if len(data['config']['objectives']) == 2]
    
    n_plots = len(scenarios_2d)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, 
                             figsize=(figsize_per_plot[0]*ncols, figsize_per_plot[1]*nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, name in enumerate(scenarios_2d):
        if name not in pareto_results:
            continue
            
        ax = axes[idx]
        config = pareto_results[name]['config']
        df_pareto = pareto_results[name]['pareto_df']
        
        obj1, obj2 = config['objectives']
        dir1, dir2 = config['directions']
        
        # Escalar dades
        x_all = get_scaled(df_all, obj1)
        y_all = get_scaled(df_all, obj2)
        x_pareto = get_scaled(df_pareto, obj1)
        y_pareto = get_scaled(df_pareto, obj2)
        
        # Plot
        ax.scatter(x_all, y_all, c='lightgray', alpha=0.2, s=5, rasterized=True)
        ax.scatter(x_pareto, y_pareto, c='red', alpha=0.7, s=15)
        
        # Etiquetes
        ax.set_xlabel(get_label(obj1), fontsize=9)
        ax.set_ylabel(get_label(obj2), fontsize=9)
        ax.set_title(config['description'], fontsize=10, fontweight='bold')
        
        # Indicar direccions òptimes amb fletxes
        ax.annotate('', xy=(0.05, 0.95 if dir2 == 'max' else 0.05), 
                   xytext=(0.05, 0.05 if dir2 == 'max' else 0.95),
                   xycoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        ax.annotate('', xy=(0.95 if dir1 == 'max' else 0.05, 0.05), 
                   xytext=(0.05 if dir1 == 'max' else 0.95, 0.05),
                   xycoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        
        # Info
        ax.text(0.98, 0.98, f'n={len(df_pareto)}', transform=ax.transAxes, 
               fontsize=8, ha='right', va='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Amagar eixos sobrants
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('pareto_panel_2d.png', dpi=150, bbox_inches='tight')
    plt.show()
    
def plot_pareto_panel_3d(df_all, pareto_results, scenarios_3d=None, ncols=2, figsize_per_plot=(6, 5)):
    """
    Panell de gràfics 3D per múltiples fronts de Pareto.
    
    Args:
        df_all: DataFrame amb tots els escenaris
        pareto_results: Diccionari retornat per extract_pareto_energy_scenarios
        scenarios_3d: Llista de noms d'escenaris a mostrar (None = tots els 3D)
        ncols: Nombre de columnes del panell
        figsize_per_plot: Mida de cada subplot
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Filtrar escenaris 3D
    if scenarios_3d is None:
        scenarios_3d = [name for name, data in pareto_results.items() 
                       if len(data['config']['objectives']) == 3]
    
    n_plots = len(scenarios_3d)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig = plt.figure(figsize=(figsize_per_plot[0]*ncols, figsize_per_plot[1]*nrows))
    
    for idx, name in enumerate(scenarios_3d):
        if name not in pareto_results:
            continue
        
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        config = pareto_results[name]['config']
        df_pareto = pareto_results[name]['pareto_df']
        
        obj1, obj2, obj3 = config['objectives']
        
        # Escalar dades (només Pareto per claredat)
        x_pareto = get_scaled(df_pareto, obj1)
        y_pareto = get_scaled(df_pareto, obj2)
        z_pareto = get_scaled(df_pareto, obj3)
        
        # Color basat en el tercer objectiu
        scatter = ax.scatter(x_pareto, y_pareto, z_pareto, 
                            c=z_pareto, cmap='viridis', alpha=0.7, s=15)
        
        # Etiquetes
        ax.set_xlabel(get_label(obj1), fontsize=8, labelpad=10)
        ax.set_ylabel(get_label(obj2), fontsize=8, labelpad=10)
        ax.set_zlabel(get_label(obj3), fontsize=8, labelpad=10)
        ax.set_title(f"{config['description']}\n(n={len(df_pareto)})", fontsize=9, fontweight='bold')
        
        # Ajustar angle de visió
        ax.view_init(elev=20, azim=45)
        
        # Reduir mida dels ticks
        ax.tick_params(axis='both', which='major', labelsize=7)
    
    plt.tight_layout()
    plt.savefig('pareto_panel_3d.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_pareto_panel_3d_projections(df_all, pareto_results, scenario_name, figsize=(15, 4)):
    """
    Mostra un escenari 3D amb les seves 3 projeccions 2D.
    
    Args:
        df_all: DataFrame amb tots els escenaris
        pareto_results: Diccionari retornat per extract_pareto_energy_scenarios
        scenario_name: Nom de l'escenari 3D a mostrar
        figsize: Mida de la figura
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    config = pareto_results[scenario_name]['config']
    df_pareto = pareto_results[scenario_name]['pareto_df']
    
    obj1, obj2, obj3 = config['objectives']
    
    # Escalar
    x = get_scaled(df_pareto, obj1)
    y = get_scaled(df_pareto, obj2)
    z = get_scaled(df_pareto, obj3)
    
    fig = plt.figure(figsize=figsize)
    
    # # Vista 3D
    # ax1 = fig.add_subplot(141, projection='3d')
    # ax1.scatter(x, y, z, c=z, cmap='viridis', alpha=0.7, s=15)
    # ax1.set_xlabel(get_label(obj1), fontsize=8)
    # ax1.set_ylabel(get_label(obj2), fontsize=8)
    # ax1.set_zlabel(get_label(obj3), fontsize=8)
    # ax1.set_title('Vista 3D', fontsize=10)
    # ax1.view_init(elev=20, azim=45)
    
    # Projecció XY
    ax1 = fig.add_subplot(142)
    ax1.scatter(x, y, c=z, cmap='viridis', alpha=0.7, s=15)
    ax1.set_xlabel(get_label(obj1), fontsize=9)
    ax1.set_ylabel(get_label(obj2), fontsize=9)
    ax1.set_title(f'{obj1} vs {obj2}', fontsize=10)
    
    # Projecció XZ
    ax2 = fig.add_subplot(143)
    ax2.scatter(x, z, c=y, cmap='viridis', alpha=0.7, s=15)
    ax2.set_xlabel(get_label(obj1), fontsize=9)
    ax2.set_ylabel(get_label(obj3), fontsize=9)
    ax2.set_title(f'{obj1} vs {obj3}', fontsize=10)
    
    # Projecció YZ
    ax3 = fig.add_subplot(144)
    ax3.scatter(y, z, c=x, cmap='viridis', alpha=0.7, s=15)
    ax3.set_xlabel(get_label(obj2), fontsize=9)
    ax3.set_ylabel(get_label(obj3), fontsize=9)
    ax3.set_title(f'{obj2} vs {obj3}', fontsize=10)
    
    plt.suptitle(config['description'], fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'pareto_3d_{scenario_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

#%%
%%time 
df_rgs = pd.read_parquet('df_rgs_50k_8v_E2.parquet')
df_rgs = pd.read_parquet('df_rgs_50k_14v_E2b.parquet')
df_rgs = pd.read_parquet('df_rgs_50k_14v_E2030.parquet')
df_rgs = pd.read_parquet('df_rgs_50k_7v_E1a.parquet')
# df_rgs = pd.read_parquet('df_rgs_50k_7v_old.parquet')


pareto_results = extract_pareto_energy_scenarios(df_rgs)
pareto_results = extract_pareto_energy_scenarios(df_tests)

#%%

# Panell 2D (tots els escenaris 2D)
# plot_pareto_panel_2d(df_tests, pareto_results, ncols=3)
plot_pareto_panel_2d(df_rgs, pareto_results, ncols=2)
# Panell 2D (selecció)

plot_pareto_panel_2d(df_rgs, pareto_results, 
                     scenarios_2d=['cost_vs_nivell_min','cost_vs_nivell_mitja','cost_vs_restriccions','residu_vs_cost'],
                     ncols=2)

plot_pareto_panel_2d(df_rgs, pareto_results, 
                     scenarios_2d=['desal_vs_nivell_min','regen_vs_nivell_min','solar_vs_nivell_min','eolica_vs_nivell_min'],
                     ncols=2)

# plot_pareto_panel_2d(df_rgs, pareto_results, 
#                      # scenarios_2d=['saturacio_desal_cost'],
#                      scenarios_2d=['sobreeiximent_vs_nivell_mean','inflexio_vs_nivell_mean','prealerta_vs_nivell_mean','alerta_vs_nivell_mean'],
#                      ncols=2)

plot_pareto_panel_2d(df_rgs, pareto_results, 
                     scenarios_2d=['sobreeiximent_vs_nivell_mean','inflexio_vs_nivell_mean','sobreeiximent_vs_desal_cf','inflexio_vs_desal_cf'],
                     ncols=2)


# plot_pareto_panel_2d(df_rgs, pareto_results, 
#                      scenarios_2d=['desal_vs_nivell_min','cost_vs_nivell_mitja','cost_vs_restriccions','residu_vs_cost'],
#                      ncols=2)






plot_pareto_panel_2d(df_rgs, pareto_results, 
                     scenarios_2d=['cost_vs_nivell_min'],
                     ncols=1)

plot_pareto_panel_2d(df_rgs, pareto_results, 
                     scenarios_2d=['cost_vs_nivell_mitja', 'cost_vs_msd'],
                     ncols=2)

plot_pareto_panel_2d(df_rgs, pareto_results, 
                     scenarios_2d=['inflexio_vs_nivell_mean'],
                     ncols=1)

plot_pareto_panel_2d(df_rgs, pareto_results, 
                     scenarios_2d=['desal_vs_nivell_min'],
                     ncols=1)

plot_pareto_panel_2d(df_rgs, pareto_results, 
                     scenarios_2d=['prealerta_vs_nivell mean'],
                     ncols=1)



# Panell 2D (selecció)
subset = df_rgs[df_rgs.llindar_desal_max>3]
# subset = df_rgs
pareto_results = extract_pareto_energy_scenarios(subset)
plot_pareto_panel_2d(subset, pareto_results, 
                     scenarios_2d=['cost_vs_nivell_min','cost_vs_nivell_mitja'],
                     ncols=2)

plot_pareto_panel_2d(df_tests, pareto_results, 
                     scenarios_2d=['cost_vs_nivell_min', 'residu_vs_cost', 'desal_vs_nivell_min'],
                     ncols=3)

# Panell 3D (tots els escenaris 3D)
plot_pareto_panel_3d(df_rgs, pareto_results,['seguretat_hidrica1_renw'], ncols=2)

# Detall d'un escenari 3D amb projeccions
plot_pareto_panel_3d_projections(df_rgs, pareto_results, 'nexe_nivell_min')
plot_pareto_panel_3d_projections(df_rgs, pareto_results, 'seguretat_hidrica_renw')


#%%

"""
Funció per crear histogrames de variables del nexe aigua-energia.
Afegir al teu script de visualitzacions.
"""

def plot_histograms(df, variables, ncols=3, figsize_per_plot=(4, 3), 
                    bins=50, color='steelblue', alpha=0.7,
                    show_stats=True, density=False):
    """
    Crea histogrames per una o múltiples variables amb escalat automàtic.
    
    Args:
        df: DataFrame amb les dades
        variables: String (una variable) o llista de variables
        ncols: Nombre de columnes del panell (si múltiples variables)
        figsize_per_plot: Mida de cada subplot
        bins: Nombre de bins o seqüència de bins
        color: Color dels histogrames
        alpha: Transparència
        show_stats: Mostrar estadístiques (mitjana, std, min, max)
        density: Si True, normalitza l'histograma (àrea = 1)
    
    Returns:
        fig: Figura de matplotlib
    
    Exemple:
        # Una sola variable
        plot_histograms(df_rgs, 'total_costs')
        
        # Múltiples variables
        plot_histograms(df_rgs, ['total_costs', 'gas_imports', 'min_level'])
        
        # Amb opcions
        plot_histograms(df_rgs, ['total_costs', 'min_level'], 
                       bins=30, color='coral', show_stats=True)
    """
    
    # Convertir a llista si és una sola variable
    if isinstance(variables, str):
        variables = [variables]
    
    # Filtrar variables que existeixen
    variables = [v for v in variables if v in df.columns]
    if not variables:
        print("⚠ Cap de les variables especificades existeix al DataFrame")
        return None
    
    n_plots = len(variables)
    
    # Configurar layout
    if n_plots == 1:
        fig, ax = plt.subplots(figsize=figsize_per_plot)
        axes = [ax]
    else:
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, 
                                 figsize=(figsize_per_plot[0]*ncols, 
                                         figsize_per_plot[1]*nrows))
        axes = axes.flatten()
    
    for idx, var in enumerate(variables):
        ax = axes[idx]
        
        # Obtenir dades escalades
        data = get_scaled(df, var)
        
        # Histograma
        ax.hist(data, bins=bins, color=color, alpha=alpha, 
                edgecolor='white', linewidth=0.5, density=density)
        
        # Etiqueta amb unitats
        ax.set_xlabel(get_label(var), fontsize=9)
        ax.set_ylabel('Densitat' if density else 'Freqüència', fontsize=9)
        
        # Títol
        cfg = VARIABLES_CONFIG.get(var, {'label': var})
        ax.set_title(cfg['label'], fontsize=10, fontweight='bold')
        
        # Estadístiques
        if show_stats:
            stats_text = (f'μ = {data.mean():.2f}\n'
                         f'σ = {data.std():.2f}\n'
                         f'min = {data.min():.2f}\n'
                         f'max = {data.max():.2f}')
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=7, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Línia vertical per la mitjana
        ax.axvline(data.mean(), color='red', linestyle='--', 
                  linewidth=1.5, alpha=0.8, label='Mitjana')
    
    # Amagar eixos sobrants
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('histograms.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # return fig


def plot_histogram_comparison(df_list, labels, variable, 
                              bins=50, alpha=0.5, figsize=(8, 5)):
    """
    Compara histogrames de la mateixa variable per diferents DataFrames.
    Útil per comparar distribucions entre escenaris o mètodes.
    
    Args:
        df_list: Llista de DataFrames
        labels: Llista d'etiquetes per cada DataFrame
        variable: Variable a comparar
        bins: Nombre de bins
        alpha: Transparència
        figsize: Mida de la figura
    
    Exemple:
        plot_histogram_comparison(
            [df_rgs, df_nsga], 
            ['Random Grid Search', 'NSGA-II'],
            'total_costs'
        )
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_list)))
    
    for df, label, color in zip(df_list, labels, colors):
        if variable not in df.columns:
            print(f"⚠ Variable '{variable}' no trobada a {label}")
            continue
        
        data = get_scaled(df, variable)
        ax.hist(data, bins=bins, alpha=alpha, label=label, 
                color=color, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel(get_label(variable), fontsize=10)
    ax.set_ylabel('Freqüència', fontsize=10)
    
    cfg = VARIABLES_CONFIG.get(variable, {'label': variable})
    ax.set_title(f"Comparació: {cfg['label']}", fontsize=11, fontweight='bold')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'histogram_comparison_{variable}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # return fig

#%%

# plot_histograms(df_rgs, 'total_costs')
plot_histograms(df_rgs, ['mean_level', 'squared_dev'], ncols=2, bins=40, color='coral')

plot_histograms(df_rgs, ['min_level', 'mean_level', 'squared_dev'], ncols=3, bins=40, color='coral')
plot_histograms(df_rgs, ['restriction_days', 'restriction_savings'], ncols=2, bins=40, color='coral')

df_rgs['restriction_ratio'] = (df_rgs.restriction_days / df_rgs.restriction_savings)
plot_histograms(df_rgs, ['restriction_days', 'restriction_savings', 'restriction_ratio'], ncols=3, bins=40, color='coral')



plot_histograms(df_rgs, ['total_costs', 'gas_imports', 'min_level', 'mean_level'], 
                   ncols=2, bins=40, color='coral')

plot_histograms(df_rgs, 'total_costs')

plot_histograms(df_rgs, 'gas_imports')
#%%

# =============================================================================
# EXEMPLES D'ÚS
# =============================================================================

if __name__ == "__main__":
    # Exemple amb dades simulades
    import pandas as pd
    
    np.random.seed(42)
    n = 1000
    
    df_test = pd.DataFrame({
        'total_costs': np.random.normal(2e9, 5e8, n),
        'gas_imports': np.random.normal(5e6, 1e6, n),
        'min_level': np.random.beta(2, 5, n) * 100,
        'mean_level': np.random.beta(5, 2, n) * 100,
        'restriction_days': np.random.poisson(20, n),
    })
    
    # Una variable
    print("=== Histograma d'una variable ===")
    plot_histograms(df_test, 'total_costs')
    
    # Múltiples variables
    print("\n=== Panell d'histogrames ===")
    plot_histograms(df_test, ['total_costs', 'gas_imports', 'min_level', 'mean_level'], 
                   ncols=2, bins=40, color='coral')
