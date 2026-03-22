# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 01:33:09 2026

@author: Víctor García Carrasco
"""

def comparar_fronts_pareto(front_rgs, front_nsga, objectives, directions):
    """
    Compara dos fronts de Pareto amb mètriques estàndard.
    
    Args:
        front_rgs: DataFrame amb solucions Pareto del RGS
        front_nsga: DataFrame amb solucions Pareto del NSGA-II
        objectives: Llista de noms de columnes dels objectius
        directions: Llista de direccions ('min' o 'max')
    """
    
    # Extreure matrius de fitness
    fits_rgs = front_rgs[objectives].values
    fits_nsga = front_nsga[objectives].values
    
    # Normalitzar per comparació (min-max sobre ambdós fronts)
    all_fits = np.vstack([fits_rgs, fits_nsga])
    mins = all_fits.min(axis=0)
    maxs = all_fits.max(axis=0)
    
    fits_rgs_norm = (fits_rgs - mins) / (maxs - mins + 1e-10)
    fits_nsga_norm = (fits_nsga - mins) / (maxs - mins + 1e-10)
    
    # Invertir objectius a maximitzar per tenir tot a minimitzar
    for i, d in enumerate(directions):
        if d == 'max':
            fits_rgs_norm[:, i] = 1 - fits_rgs_norm[:, i]
            fits_nsga_norm[:, i] = 1 - fits_nsga_norm[:, i]
    
    resultats = {}
    
    # --- MÈTRICA 1: Hypervolume ---
    from scipy.spatial import ConvexHull
    
    def hypervolume_2d(front, ref_point):
        """Hypervolume per 2D (extensible a 3D amb llibreries)."""
        sorted_front = front[front[:, 0].argsort()]
        hv = 0
        prev_x = 0
        for point in sorted_front:
            hv += (ref_point[0] - prev_x) * (ref_point[1] - point[1])
            prev_x = point[0]
        return max(0, hv)
    
    ref_point = np.ones(len(objectives)) * 1.1  # Punt de referència
    
    # Per 3D, usar pymoo o pygmo
    try:
        from pymoo.indicators.hv import HV
        ind = HV(ref_point=ref_point)
        resultats['HV_RGS'] = ind(fits_rgs_norm)
        resultats['HV_NSGA'] = ind(fits_nsga_norm)
        resultats['HV_ratio'] = resultats['HV_NSGA'] / resultats['HV_RGS']
    except ImportError:
        print("pymoo no disponible per calcular Hypervolume 3D")
        resultats['HV_RGS'] = None
        resultats['HV_NSGA'] = None
    
    # --- MÈTRICA 2: Spacing (uniformitat) ---
    def spacing(front):
        """Mesura la uniformitat de distribució del front."""
        n = len(front)
        if n < 2:
            return 0
        
        dists = []
        for i in range(n):
            dist_min = np.min([np.linalg.norm(front[i] - front[j]) 
                              for j in range(n) if i != j])
            dists.append(dist_min)
        
        d_mean = np.mean(dists)
        return np.sqrt(np.sum((np.array(dists) - d_mean)**2) / (n - 1))
    
    resultats['Spacing_RGS'] = spacing(fits_rgs_norm)
    resultats['Spacing_NSGA'] = spacing(fits_nsga_norm)
    
    # --- MÈTRICA 3: Spread (extensió) ---
    def spread(front):
        """Mesura l'extensió del front en cada dimensió."""
        return np.max(front, axis=0) - np.min(front, axis=0)
    
    resultats['Spread_RGS'] = spread(fits_rgs_norm)
    resultats['Spread_NSGA'] = spread(fits_nsga_norm)
    
    # --- MÈTRICA 4: Cobertura (C-metric) ---
    def cobertura(front_a, front_b):
        """Proporció de solucions de B dominades per almenys una de A."""
        count = 0
        for b in front_b:
            for a in front_a:
                if np.all(a <= b) and np.any(a < b):
                    count += 1
                    break
        return count / len(front_b)
    
    resultats['C_NSGA_domina_RGS'] = cobertura(fits_nsga_norm, fits_rgs_norm)
    resultats['C_RGS_domina_NSGA'] = cobertura(fits_rgs_norm, fits_nsga_norm)
    
    # --- MÈTRICA 5: Distància generacional (GD) ---
    def generational_distance(front_a, front_ref):
        """Distància mitjana de A al front de referència."""
        dists = []
        for a in front_a:
            dist_min = np.min([np.linalg.norm(a - r) for r in front_ref])
            dists.append(dist_min)
        return np.mean(dists)
    
    # Usar la unió com a referència aproximada
    front_unio = np.vstack([fits_rgs_norm, fits_nsga_norm])
    
    resultats['GD_RGS'] = generational_distance(fits_rgs_norm, front_unio)
    resultats['GD_NSGA'] = generational_distance(fits_nsga_norm, front_unio)
    
    # --- RESUM ---
    print("\n" + "="*60)
    print("COMPARACIÓ DE FRONTS DE PARETO")
    print("="*60)
    print(f"\nNombre de solucions:")
    print(f"  RGS:   {len(front_rgs)}")
    print(f"  NSGA:  {len(front_nsga)}")
    
    if resultats['HV_RGS']:
        print(f"\nHypervolume (més alt = millor):")
        print(f"  RGS:   {resultats['HV_RGS']:.4f}")
        print(f"  NSGA:  {resultats['HV_NSGA']:.4f}")
        print(f"  Ratio: {resultats['HV_ratio']:.2f}")
    
    print(f"\nSpacing (més baix = més uniforme):")
    print(f"  RGS:   {resultats['Spacing_RGS']:.4f}")
    print(f"  NSGA:  {resultats['Spacing_NSGA']:.4f}")
    
    # print(f"\nSpread (més alt = més diversitat):")
    # print(f"  RGS:   {resultats['Spread_RGS']:.4f}")
    # print(f"  NSGA:  {resultats['Spread_NSGA']:.4f}")    
    
    print(f"\nCobertura (proporció dominada):")
    print(f"  NSGA domina RGS: {resultats['C_NSGA_domina_RGS']:.1%}")
    print(f"  RGS domina NSGA: {resultats['C_RGS_domina_NSGA']:.1%}")
    
    print(f"\nDistància Generacional (més baix = més proper al òptim):")
    print(f"  RGS:   {resultats['GD_RGS']:.4f}")
    print(f"  NSGA:  {resultats['GD_NSGA']:.4f}")
    
    return resultats

def plot_comparacio_fronts_3d(front_rgs, front_nsga, objectives, labels=None):
    """
    Visualització 3D comparativa dels dos fronts de Pareto.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if labels is None:
        labels = objectives
    
    fits_rgs = front_rgs[objectives].values
    fits_nsga = front_nsga[objectives].values
    
    # --- FIGURA 1: Superposició 3D ---
    fig = plt.figure(figsize=(20, 5))
    
    # Vista 1: Superposats
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(fits_nsga[:, 0], fits_nsga[:, 1], fits_nsga[:, 2], 
                c='red', alpha=0.7, s=5, label=f'NSGA-II (n={len(fits_nsga)})')
    ax1.scatter(fits_rgs[:, 0], fits_rgs[:, 1], fits_rgs[:, 2], 
                c='blue', alpha=0.5, s=5, label=f'RGS (n={len(fits_rgs)})')
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax1.set_zlabel(labels[2])
    ax1.set_title('Superposició')

    # Cambiar ángulo de vista (elevación, azimut)
    ax1.view_init(elev=45, azim=20)  # Ajusta elev y azim según necesites    
    # Solución 4: Ajustar márgenes de la figura completa
    ax1.zaxis.labelpad = -25
    
    ax1.legend()
    
    # # Vista 2: Només RGS
    # ax2 = fig.add_subplot(132, projection='3d')
    # ax2.scatter(fits_rgs[:, 0], fits_rgs[:, 1], fits_rgs[:, 2], 
    #             c='blue', alpha=0.6, s=20)
    # ax2.set_xlabel(labels[0])
    # ax2.set_ylabel(labels[1])
    # ax2.set_zlabel(labels[2])
    # ax2.set_title(f'RGS (n={len(fits_rgs)})')
    
    # # Vista 3: Només NSGA
    # ax3 = fig.add_subplot(133, projection='3d')
    # ax3.scatter(fits_nsga[:, 0], fits_nsga[:, 1], fits_nsga[:, 2], 
    #             c='red', alpha=0.6, s=30)
    # ax3.set_xlabel(labels[0])
    # ax3.set_ylabel(labels[1])
    # ax3.set_zlabel(labels[2])
    # ax3.set_title(f'NSGA-II (n={len(fits_nsga)})')
    
    # plt.tight_layout()
    plt.show()
    
    # --- FIGURA 2: Projeccions 2D ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    parelles = [(0, 1), (0, 2), (1, 2)]
    
    for ax, (i, j) in zip(axes, parelles):
        ax.scatter(fits_nsga[:, i], fits_nsga[:, j], 
                   c='red', alpha=0.6, s=25, label='NSGA-II')        
        ax.scatter(fits_rgs[:, i], fits_rgs[:, j], 
                   c='blue', alpha=0.4, s=15, label='RGS')
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.legend()
        ax.set_title(f'{labels[i]} vs {labels[j]}')
    
    plt.tight_layout()
    plt.show()
    
    # --- FIGURA 3: Distribucions marginals ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (i, obj) in zip(axes, enumerate(objectives)):
        ax.hist(fits_nsga[:, i], bins=20, alpha=0.5, density=True, label='NSGA-II', color='red')
        ax.hist(fits_rgs[:, i], bins=20, alpha=0.5, density=True, label='RGS', color='blue')
        ax.axvline(fits_rgs[:, i].mean(), color='blue', linestyle='--', linewidth=2)
        ax.axvline(fits_nsga[:, i].mean(), color='red', linestyle='-', linewidth=2)
        ax.set_xlabel(labels[i])
        ax.set_ylabel('Densitat')
        ax.legend()
        ax.set_title(f'Distribució {labels[i]}')
    
    plt.tight_layout()
    plt.show()

#%%
%%time
# Definir objectius
# objectives = ['gas_imports', 'squared_dev', 'total_costs']
objectives = ['total_costs','gas_imports', 'squared_dev']
directions = ['min', 'min', 'min']
# labels = ['Gas_Imports (TWh/y)', 'MSD', 'Costos (M€/y)']
labels = ['Costos (M€/y)','Gas_Imports (TWh/y)', 'MSD']

# labels = ['Emissions (TWh)', 'Nivell mínim (%)', 'Costos (M€)']

n_years = np.round((len(precomputed['hydro_level_int'])/(24*365)),0)
# Extreure fronts
front_rgs = extract_pareto_front(df_rgs, objectives, directions)
# front_rgs = pareto_results['costo_seguridad_dependencia']["pareto_df"].copy()
front_rgs.loc[:, ['gas_imports','total_costs']] /= (n_years * 1000000)
# front_rgs.loc[:, 'total_costs'] /= n_years
front_nsga = df_pareto_nsga.copy()
front_nsga = front_nsga.drop(['gas_imports','squared_dev'], axis=1)
# front_nsga = optimizador.hof_to_dataframe(hof)
# front_nsga.loc[:, 'total_costs'] /= n_years

# Comparació quantitativa
# front_rgs[['gas_imports','total_costs']] = front_rgs[['gas_imports','total_costs']]/1000000

front_nsga.rename(columns={'obj_costs': 'total_costs', 'obj_msqdev': 'squared_dev', 'obj_emissions': 'gas_imports'}, inplace=True)
front_nsga.loc[:, ['gas_imports','total_costs']] /= n_years

# DESCOMENTAR SI SE QUIERE HACER LA COMPARACION
resultats = comparar_fronts_pareto(front_rgs, front_nsga, objectives, directions)

# Comparació visual
plot_comparacio_fronts_3d(front_rgs, front_nsga, objectives, labels)


#%%
def calcular_solapament(min1, max1, min2, max2):
    overlap_min = max(min1, min2)
    overlap_max = min(max1, max2)
    if overlap_max <= overlap_min:
        return 0.0
    rang_total = max(max1, max2) - min(min1, min2)
    return 100 * (overlap_max - overlap_min) / rang_total


front_nsga.rename(columns={'llindar_activacio_desal_max': 'llindar_desal_max', 'llindar_activacio_regen_max': 'llindar_regen_max', 'overflow_threshold_pct': 'overflow_threshold' }, inplace=True)

# Variables a comparar
variables = [
    'potencia_solar', 'potencia_eolica', 'potencia_baterias',
    'max_desalation', 'min_run_hours', 'max_regen', 'overflow_threshold',
    'midpoint_estimation', 'llindar_desal_max',
    'llindar_regen_max', 'L_prealerta', 'L_alerta',
    'L_excepcionalitat', 'L_emergencia'
]

# Calcular taula
resultats = []
for var in variables:
    if var in front_rgs.columns and var in front_nsga.columns:
        min_rgs, max_rgs = front_rgs[var].min(), front_rgs[var].max()
        min_nsga, max_nsga = front_nsga[var].min(), front_nsga[var].max()
        solap = calcular_solapament(min_rgs, max_rgs, min_nsga, max_nsga)
        resultats.append({
            'Variable': var,
            'RGS Mín': min_rgs,
            'RGS Màx': max_rgs,
            'NSGA Mín': min_nsga,
            'NSGA Màx': max_nsga,
            'Solapament (%)': solap
        })

df_rangs = pd.DataFrame(resultats).round(1)
print(df_rangs.to_string(index=False))
print(df_rangs.to_string(index=False, columns=df_rangs.columns[1:]))
