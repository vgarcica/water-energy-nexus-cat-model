# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 12:29:14 2026

@author: tirki
"""

from EnerSimFunc import (generar_matriu_estalvi)

precomputed['hydro_level_int']

# # CONSUM

# # Consum base mensual [hm³/dia] - de gener a desembre
# consum_base_diari_estacional = np.array([
#     1.37, 1.37, 1.42, 1.74, 2.32, 3.03, 
#     3.51, 3.24, 2.36, 1.48, 1.37, 1.37
# ])


#  CABAL ECOLOGIC

# --- 1. DADES DE CABALS (Recuperades i processades de l'apartat anterior) ---
# Ordenats d'Octubre a Setembre (l/s)
raw_manteniment = np.array([3736, 3736, 3921, 3814, 3814, 3921, 4850, 4850, 3998, 3091, 3091, 3091])
raw_alerta      = np.array([3159, 3159, 3290, 3243, 3243, 3290, 4070, 4070, 3271, 2569, 2569, 2569])
raw_excep       = np.array([2450, 2450, 2483, 2464, 2464, 2483, 2528, 2528, 2492, 2401, 2401, 2401])

def processar_vector_cabals(vector_oct_sep):
    """Reordena a Gen-Des i converteix l/s a hm3/dia"""
    # Reordenar: Index 3 (Gener) fins al final + Index 0,1,2 (Oct, Nov, Des)
    vector_gen_des = np.concatenate((vector_oct_sep[3:], vector_oct_sep[:3]))
    factor_conversio = 0.0000864 # (l/s -> hm3/dia)
    return vector_gen_des * factor_conversio

# Vectors base en hm3/dia (Gener a Desembre)
vec_mant  = processar_vector_cabals(raw_manteniment)
vec_alert = processar_vector_cabals(raw_alerta)
vec_excep = processar_vector_cabals(raw_excep)

# --- 2. RUTINA PER CONSTRUIR LA MATRIU Q_ECO ---

def generar_matriu_q_eco():
    """
    Retorna una matriu (12 x 7) amb el cabal ecològic (hm3/dia)
    per a cada mes i fase de sequera.
    
    Mapeig assumit:
      - Normalitat (0), Prealerta (1) --> Manteniment
      - Alerta (2)                    --> Alerta
      - Excep (3) fins Emerg 3 (6)    --> Excepcionalitat (o pitjor)
    """
    matriu = np.zeros((12, 7), dtype=np.float64)
    
    # Assignació per columnes (Fases)
    # Fases 0 i 1: Manteniment
    matriu[:, 0] = vec_mant
    matriu[:, 1] = vec_mant
    
    # Fase 2: Alerta
    matriu[:, 2] = vec_alert
    
    # Fases 3, 4, 5, 6: Excepcionalitat (s'aplica el mínim a partir d'aquí)
    for fase in range(3, 7):
        matriu[:, fase] = vec_excep
        
    return matriu

# --- 3. RUTINA PER CONSTRUIR LA MATRIU Q_RES (VOLUM RESTRINGIT) ---

def generar_matriu_q_res(consum_base_mensual, umbrales=None, restriccions=None):
    """
    Calcula el volum absolut d'aigua estalviada/restringida (hm3/dia).
    
    Args:
        consum_base_mensual: array (12,) amb la demanda bruta (Q_dem)
        umbrales, restriccions: passats a la teva funció generar_matriu_estalvi
    
    Returns:
        matriu_q_res: np.ndarray (12, 7) en hm3/dia
        llindars: np.ndarray per al càlcul de fase
    """
    # 1. Obtenim la matriu de % d'estalvi (valors de 0 a 1, ex: 0.15)
    matriu_pct, llindars = generar_matriu_estalvi(umbrales, restriccions)
    
    # 2. Multipliquem la demanda base pel % d'estalvi
    # Utilitzem broadcasting: (12, 1) * (12, 7)
    matriu_q_res = consum_base_mensual[:, np.newaxis] * matriu_pct
    
    return matriu_q_res, llindars

# --- 4. EXECUCIÓ I VISUALITZACIÓ ---

# Consum base definit anteriorment
consum_base = np.array([1.37, 1.37, 1.42, 1.74, 2.32, 3.03, 3.51, 3.24, 2.36, 1.48, 1.37, 1.37])

# Generem les matrius
MATRIU_Q_ECO = generar_matriu_q_eco()
MATRIU_Q_RES, LLINDARS = generar_matriu_q_res(consum_base)

# Exemple de com quedaria l'equació a l'agost (mes 7, índex 7) en fase d'Emergència I (índex 4)
mes_idx = 7 # Agost
fase_idx = 4 # Emergencia I

q_eco_val = MATRIU_Q_ECO[mes_idx, fase_idx]
q_res_val = MATRIU_Q_RES[mes_idx, fase_idx]
q_dem_val = consum_base[mes_idx]

print(f"--- EXEMPLE AGOST (Emergència I) ---")
print(f"Demanda Base (Q_dem): {q_dem_val:.3f} hm3/dia")
print(f"Restricció   (Q_res): {q_res_val:.3f} hm3/dia (Això és el que NO se serveix)")
print(f"Cabal Eco    (Q_eco): {q_eco_val:.3f} hm3/dia")
print(f"------------------------------------")
print(f"Equació Outflow (sense comptar reg/des encara):")
print(f"max({q_eco_val:.3f}, {q_dem_val:.3f} - Q_des - Q_reg - {q_res_val:.3f})")


# precomputed['hydro_level_int'] * max_capacity_int - (  - datos.dessalacio_diaria)


#%%

# --- 1. DEFINICIÓ DE VARIABLES D'ENTRADA ---

# Capacitat màxima de les Conques Internes (aprox. 700 hm3, ajusta-ho al valor exacte)
CAPACITAT_MAX_HM3 = max_capacity_int #694.45  # Valor típic Ter-Llobregat + altres internes


# Fonts no convencionals (hm3/dia) - Variables definibles com demanaves
Q_des_total_dia = datos.dessalacio_diaria  # Substitueix pel valor o serie temporal si la tens
Q_reg_total_dia = datos.regeneracio_diaria  # Substitueix pel valor o serie temporal

# Recuperem la sèrie horària i fem una còpia per no alterar l'original
# precomputed['hydro_level_int'] ve en % (0-100)
df_balanc = pd.DataFrame({'nivell_pct': precomputed['hydro_level_int']})

# --- 2. CÀLCUL DE VOLUMS I DELTES (El motor del balanç) ---

# Convertim % a Volum (hm3)
df_balanc['volum_hm3'] = (df_balanc['nivell_pct'] / 100.0) * CAPACITAT_MAX_HM3

# Calculem la variació de volum respecte l'hora anterior (Delta V)
# Delta = V_t - V_{t-1}
# Si el volum puja, delta és positiu. Si baixa, negatiu.
df_balanc['delta_volum_hm3'] = df_balanc['volum_hm3'].diff().fillna(0)

# --- 3. DETERMINACIÓ DE LA FASE I EL MES (Vectoritzat) ---

# Índexs temporals
mesos_idx = df_balanc.index.month - 1  # 0=Gener, 11=Desembre

# Índexs de Fase de Sequera
# LLINDARS estava ordenat ascendent: [5.5, 11, 16, 25, 40, 60] (Exemple)
# np.searchsorted ens diu on cauria el valor.
# Si nivell = 70 (>60) -> idx_sorted = 6. 
# Si nivell = 4 (<5.5) -> idx_sorted = 0.
idx_sorted = np.searchsorted(LLINDARS, df_balanc['nivell_pct'])

# CORRECCIÓ D'ÍNDEX: 
# A les teves matrius (MATRIU_Q_ECO, MATRIU_Q_RES):
# Col 0 = Normalitat, Col 6 = Emergència 3.
# La relació és inversa: Si tenim molt nivell (idx_sorted 6), estem a Normalitat (0).
# Fórmula: Fase = 6 - idx_sorted
fases_idx = 6 - idx_sorted
fases_idx = np.clip(fases_idx, 0, 6) # Per seguretat, que no surti de rang

# --- 4. LOOKUP A LES MATRIUS (Lectura ràpida) ---

# Extraiem els valors corresponents a cada hora segons el seu mes i fase
# Dividim per 24 per passar de hm3/dia a hm3/hora
q_eco_hora = MATRIU_Q_ECO[mesos_idx, fases_idx] / 24.0
q_res_hora = MATRIU_Q_RES[mesos_idx, fases_idx] / 24.0
q_dem_hora = consum_base[mesos_idx] / 24.0

# 1. Alineem la sèrie diària amb l'índex horari del dataframe principal
#    'method="ffill"' repeteix el valor del dia per a les 24 hores corresponents.
# 2. Dividim per 24 per passar de hm³/dia a hm³/hora.

q_des_hora = Q_des_total_dia.reindex(df_balanc.index, method='ffill') / 24.0
q_reg_hora = Q_reg_total_dia.reindex(df_balanc.index, method='ffill') / 24.0

# --- 5. CÀLCUL DE L'OUTFLOW TEÒRIC ---

# Apliquem la fórmula de sortida: Out = max(Eco, Demanda - NoConvencional - Restriccions)
# Terme de demanda neta (el que volem treure de l'embassament per consumir)
demanda_neta_embassament = q_dem_hora - q_des_hora - q_reg_hora - q_res_hora

# L'Outflow real és el màxim entre el cabal ecològic i la demanda neta que queda per cobrir
outflow_total_hora = np.maximum(q_eco_hora, demanda_neta_embassament)

# --- 6. BALANÇ INVERS: APORTACIONS NATURALS ---

# Equació: Vol_final = Vol_inicial + Aportació - Sortida
# Per tant: Aportació = (Vol_final - Vol_inicial) + Sortida
# Aportació = Delta_Vol + Outflow
aportacio_bruta_hm3h = df_balanc['delta_volum_hm3'] + outflow_total_hora
# df_balanc['aportacio_natural_hm3h'] = df_balanc['delta_volum_hm3'] + outflow_total_hora

# --- 7. CORRECCIÓ PER VESSAMENTS IMPLÍCITS ---
# L'aportació natural no pot ser inferior al cabal ecològic mínim
# La diferència representa sortides no comptabilitzades (vessaments, etc.)

# Càlcul del vessament implícit (només quan aportació_bruta < q_eco)
vessament_implicit_hm3h = np.maximum(0, q_eco_hora - aportacio_bruta_hm3h)

# Assignació al dataframe
df_balanc['aportacio_bruta_hm3h'] = aportacio_bruta_hm3h  # Per diagnòstic
df_balanc['vessament_implicit_hm3h'] = vessament_implicit_hm3h
df_balanc['aportacio_natural_hm3h'] = aportacio_bruta_hm3h + vessament_implicit_hm3h


# Opcional: Eliminar valors negatius (soroll de mesura o evaporació no comptada)
# df_balanc['aportacio_natural_hm3h'] = df_balanc['aportacio_natural_hm3h'].clip(lower=0)

# --- RESULTAT ---
print("Càlcul d'aportacions completat.")
print(df_balanc[['nivell_pct', 'delta_volum_hm3', 'aportacio_natural_hm3h']].head())

# --- DIAGNÒSTIC ---
n_corregits = (vessament_implicit_hm3h > 0).sum()
pct_corregits = 100 * n_corregits / len(df_balanc)
print(f"Punts corregits per vessament implícit: {n_corregits} ({pct_corregits:.1f}%)")
print(f"Vessament implícit total: {vessament_implicit_hm3h.sum():.2f} hm³")


fig, ax1 = plt.subplots(figsize=(12, 6))

# Primer eix (esquerra)
color1 = 'tab:blue'
ax1.set_xlabel('Any')
ax1.set_ylabel('Vessament Implícit (hm³/h)', color=color1)
ax1.plot(df_balanc.index, df_balanc.vessament_implicit_hm3h, color=color1, linewidth=2)
# ax1.plot(df_balanc.resample('ME').sum().index, df_balanc.vessament_implicit_hm3h.resample('ME').sum(), color=color1, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

# Segon eix (dreta)
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Nivell Hidràulic (int)', color=color2)
ax2.plot(precomputed['hydro_level_int'].index, precomputed['hydro_level_int'], color=color2, linewidth=2, linestyle='--')

ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Comparació: Vessament vs Nivell Hidràulic')
fig.tight_layout()
plt.show()
#%%
def netejar_aportacions(s_inflows_horari):
    """
    1. Agrega a diari.
    2. Elimina negatius traslladant el volum al dia següent (conserva la massa).
    3. Retorna a horari dividint per 24.
    """
    # 1. Agregació Diària (Resample sumant)
    # Això ja elimina la majoria del soroll d'alta freqüència
    s_diari = s_inflows_horari.resample('D').sum()
    
    # 2. Algorisme de "Carry Forward" (Arrossegament)
    # Si tenim -5 hm3 avui, posem 0 i restem 5 hm3 al dia de demà.
    valors = s_diari.values
    for i in range(len(valors) - 1):
        if valors[i] < 0:
            # Passem el "deute" al dia següent
            valors[i+1] += valors[i] 
            # Posem el dia actual a 0
            valors[i] = 0
            
    # Cas particular: Si l'últim dia queda negatiu, el posem a 0 (error residual inevitable)
    if valors[-1] < 0:
        valors[-1] = 0
        
    # Reconstruïm la sèrie Pandas diària neta
    s_diari_neta = pd.Series(valors, index=s_diari.index)
    
    # 3. Desagregació a Horari
    # Reindexem a l'índex original horari i dividim per 24
    # ffill repeteix el valor diari 24 vegades, després dividim.
    s_horaria_final = s_diari_neta.reindex(s_inflows_horari.index, method='ffill') / 24.0
    
    return s_horaria_final

# --- ÚS ---
# Aplica-ho a la sèrie que hem calculat abans
# df_balanc['aportacio_natural_hm3h'] ve del càlcul anterior
inflows_clean = netejar_aportacions(df_balanc['aportacio_natural_hm3h'])

# Comprovació
print(f"Suma Total Original: {df_balanc['aportacio_natural_hm3h'].sum():.2f}")
print(f"Suma Total Neta:     {inflows_clean.sum():.2f}")
print(f"Minim valor (ha de ser >= 0): {inflows_clean.min()}")





#%%

import pandas as pd
import os
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓ ---
directori = 'internes'
codis_estacions = ['CI', 'ZC', 'WS', 'Z1', 'Z9'] # He corregit la coma que faltava entre Z1 i Z9
columna_interes = 'Precipitación acumulada'

llista_series = []

# --- 2. BUCLE DE LECTURA ---
print("Llegint fitxers d'estacions...")

for codi in codis_estacions:
    nom_fitxer = f"estacio_{codi}.csv"
    ruta_fitxer = os.path.join(directori, nom_fitxer)
    
    if os.path.exists(ruta_fitxer):
        try:
            # Llegim el CSV. 
            # ATENCIÓ: Ajusta 'sep' si els teus fitxers usen ';'
            # parse_dates intenta trobar la data automàticament.
            df = pd.read_csv(ruta_fitxer, sep=',', parse_dates=[0], dayfirst=True, index_col=0)
            
            # Assegurem que l'índex és datetime
            df.index = pd.to_datetime(df.index)
            
            # Extraiem només la columna de precipitació
            if columna_interes in df.columns:
                serie = df[columna_interes]
                serie.name = codi # Posem el codi com a nom de la columna
                llista_series.append(serie)
                print(f"  -> {codi}: Carregada correctament.")
            else:
                print(f"  -> {codi}: ALERTA - No s'ha trobat la columna '{columna_interes}'")
                
        except Exception as e:
            print(f"  -> {codi}: ERROR llegint el fitxer. {e}")
    else:
        print(f"  -> {codi}: El fitxer no existeix.")

# --- 3. FUSIÓ I CÀLCUL DEL PROMIG ---

if llista_series:
    # Unim totes les sèries en un sol DataFrame (alineat per data)
    df_precip = pd.concat(llista_series, axis=1)
    
    # Calculem el promig de les estacions disponibles per a cada dia
    # skipna=True fa que si una estació falla un dia, es faci el promig de les altres
    df_precip['Promig_Diari'] = df_precip.mean(axis=1, skipna=True)
    
    # Omplim possibles NaNs amb 0 només si volem assumir que no plou si no hi ha dades 
    # (Millor deixar NaNs si volem ser rigorosos, però per correlacions sovint es posa 0)
    df_precip['Promig_Diari'] = df_precip['Promig_Diari'].fillna(0)
    # s'eliminen els valors faltants (-1) per 0
    df_precip['Promig_Diari'][df_precip['Promig_Diari'] < 0] = 0

    print("\n--- Resultats ---")
    print(df_precip.head())
    
    # --- 4. VISUALITZACIÓ RÀPIDA ---
    plt.figure(figsize=(12, 6))
    plt.plot(df_precip.index, df_precip['Promig_Diari'], label='Precipitació Mitjana (mm)', color='blue', alpha=0.7)
    plt.title('Precipitació Acumulada Mitjana (Estacions Internes)')
    plt.ylabel('mm / dia')
    plt.xlabel('Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- 5. EXPORTACIÓ (Opcional) ---
    # df_precip['Promig_Diari'].to_csv('precipitacio_mitjana_conques.csv')
    
else:
    print("\nNo s'ha pogut carregar cap dada.")
    

#%%
# Configurar el estilo de seaborn para un aspecto más profesional
import seaborn as sn
sn.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def analitzar_best_lag(serie_pluja, serie_aportacions, max_dies=15):
    """
    Troba el desplaçament (lag) en dies que maximitza la correlació.
    """
    # Assegurem freqüència diària
    s_pluja = serie_pluja.resample('D').mean().fillna(0)
    s_aport = serie_aportacions.resample('D').mean() # Ja netejada
    
    correlacions = []
    lags = range(max_dies + 1)
    
    for lag in lags:
        # Desplacem la pluja 'lag' dies endavant per veure si coincideix amb l'aportació
        corr = s_pluja.shift(lag).corr(s_aport)
        correlacions.append(corr)
        print(correlacions)
    
    # Resultats
    best_lag = lags[correlacions.index(max(correlacions))]
    max_corr = max(correlacions)
    
    print(f"--- RESULTATS ANÀLISI DE LAG ---")
    print(f"Millor decalatge: {best_lag} dies")
    print(f"Correlació màxima: {max_corr:.4f}")
    
    # Gràfic
    plt.figure(figsize=(8, 4))
    plt.bar(lags, correlacions, color='skyblue')
    plt.axvline(best_lag, color='red', linestyle='--', label=f'Òptim: {best_lag} dies')
    plt.title('Precipitacions històriques vs Aportacions estimades')
    plt.xlabel('Dies de retard (Lag)')
    plt.ylabel('Coeficient de Correlació')
    plt.legend()
    plt.show()
    
    return best_lag

# --- ÚS ---
# Assumint que tens df_precip i inflows_clean dels passos anteriors
# lag_optim = analitzar_best_lag(df_precip['Promig_Diari'], inflows_clean)
lag_optim = analitzar_best_lag(df_precip['Promig_Diari'], df_balanc.aportacio_natural_hm3h)


#%%


# Suposem que tens un DataFrame 'df_total' amb columnes diaries/mensuals alineades:
# - 'Inflows': Aportacions netes (hm3)
# - 'Solar': Generació PV (MWh)
# - 'Wind': Generació Eòlica (MWh)
# - 'Hydro_Gen': Generació Hidràulica (MWh)

# 1. Creem un DataFrame conjunt (assegura't que tot té el mateix index temporal)
# Si les dades d'energia són horàries, passa-les a diàries o mensuals primer.
df_corr_analysis = pd.DataFrame({
    'Aportacions (hm3)': escenaris_simulacio['Historic'].resample('ME').sum(),
    # 'Eòlica (MWh)': results['energy_data']['Eòlica'].resample('ME').sum(), # Ajusta la ruta
    'Eòlica (MWh)': datos.df_sintetic['Eòlica_w'].resample('ME').sum(),
    # 'Solar (MWh)': results['energy_data']['Solar'].resample('ME').sum(),   # Ajusta la ruta
    'Solar (MWh)': datos.df_sintetic['Solar_w'].resample('ME').sum(),   # Ajusta la ruta
    # 'Hidro Gen (MWh)': hist_series.resample('ME').sum()                    # La sèrie històrica
    'Hydro (MWh)': datos.energia_turbinada_mensual_internes
}).dropna()


df_corr_analysis = pd.DataFrame({
    'Aportacions (hm3)': escenaris_simulacio['Historic'].resample('D').sum(),
    # 'Eòlica (MWh)': results['energy_data']['Eòlica'].resample('ME').sum(), # Ajusta la ruta
    'Eòlica (MWh)': datos.df_sintetic['Eòlica_w'].resample('D').sum(),
    # 'Solar (MWh)': results['energy_data']['Solar'].resample('ME').sum(),   # Ajusta la ruta
    'Solar (MWh)': datos.df_sintetic['Solar_w'].resample('D').sum(),   # Ajusta la ruta
    # 'Hidro Gen (MWh)': hist_series.resample('ME').sum()                    # La sèrie històrica
    'Hydro (MWh)': datos.energia_turbinada_mensual_internes
}).dropna()

# 2. Matriu de Correlació (Global)
plt.figure(figsize=(10, 8))
corr_matrix = df_corr_analysis.corr()
sn.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Matriu de Correlació Global (Mensual)')
plt.show()

# 3. Correlació Estacional (Molt important!)
# A vegades Eòlica i Pluja correlacionen a l'hivern però no a l'estiu.
df_corr_analysis['Mes'] = df_corr_analysis.index.month
corrs_mensuals = df_corr_analysis.groupby('Mes').apply(lambda x: x.corr()['Aportacions (hm3)'])

# Treiem la columna d'aportacions contra ella mateixa i la del Mes
corrs_mensuals = corrs_mensuals.drop(columns=['Aportacions (hm3)', 'Mes'], errors='ignore')

plt.figure(figsize=(12, 6))
sn.heatmap(corrs_mensuals.T, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1)
plt.title('Com es relacionen les renovables amb les aportacions segons el mes?')
plt.xlabel('Mes de l\'any')
plt.show()



# 4. Correlació Desestacionalitzada (Anomalies)
# Eliminem el component estacional restant la mitjana mensual de cada mes

def desestacionalitzar(serie):
    """Retorna les anomalies respecte a la mitjana de cada mes."""
    mitjanes_mensuals = serie.groupby(serie.index.month).transform('mean')
    return serie - mitjanes_mensuals

# Creem el DataFrame amb anomalies
df_anomalies = df_corr_analysis.drop(columns=['Mes']).apply(desestacionalitzar)

# Matriu de correlació de les anomalies
plt.figure(figsize=(10, 8))
corr_matrix_desest = df_anomalies.corr()
sn.heatmap(corr_matrix_desest, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Matriu de Correlació Desestacionalitzada (Anomalies Mensuals)')
plt.tight_layout()
plt.show()

# Opcional: Comparativa en una sola figura
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sn.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", ax=axes[0])
axes[0].set_title('Correlació Global')

sn.heatmap(corr_matrix_desest, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", ax=axes[1])
axes[1].set_title('Correlació Desestacionalitzada')

plt.tight_layout()
plt.show()

# Diagnòstic: quanta correlació "perd" cada variable?
print("\n--- Efecte de l'Estacionalitat ---")
for col in df_anomalies.columns:
    if col != 'Aportacions (hm3)':
        r_global = corr_matrix.loc['Aportacions (hm3)', col]
        r_desest = corr_matrix_desest.loc['Aportacions (hm3)', col]
        perdua = r_global - r_desest
        print(f"{col}: Global={r_global:.3f}, Desest={r_desest:.3f}, Δ={perdua:+.3f}")

#%%

def generar_escenaris_aportacions(serie_aportacions_base, serie_pluja_base, lag_optim=1):
    escenaris = {}
    
    # 1. Escenari BASE (La història tal qual)
    escenaris['Historic'] = serie_aportacions_base
    
    # 2. Escenari SEQUERA SEVERA (Reducció uniforme del 30%)
    escenaris['Sequera_Global'] = serie_aportacions_base * 0.7
    
    # 3. Escenari CANVI CLIMÀTIC (Menys aigua, més extrem)
    # Reduim un 40% a l'estiu/primavera, mantenim hivern
    # Això requereix iterar o usar màscares temporals
    s_cc = serie_aportacions_base.copy()
    mesos_secs = [4, 5, 6, 7, 8, 9] # Abr-Set
    
    # Creem una màscara per aplicar el factor només als mesos secs
    mask_secs = s_cc.index.month.isin(mesos_secs)
    s_cc[mask_secs] = s_cc[mask_secs] * 0.6  # -40% en mesos càlids
    s_cc[~mask_secs] = s_cc[~mask_secs] * 0.9 # -10% la resta
    
    escenaris['Canvi_Climatic'] = s_cc
    
    # 4. Escenari SINTÈTIC basat en PLUJA MODIFICADA
    # Imaginem que plou la meitat però molt concentrat (Dificil de modelar senzillament, 
    # però podem fer una aproximació simple amb la pluja)
    # Aquest el deixem com a idea futura si el model Rainfall-Runoff és robust.
    
    return escenaris

# ÚS
escenaris_simulacio = generar_escenaris_aportacions(inflows_clean, df_precip['Promig_Diari'])


# Visualitzem un any per veure diferències
plt.figure(figsize=(12,5))
any_visualitzar = '2022' # Tria un any que tinguis dades
plt.plot(escenaris_simulacio['Historic'][any_visualitzar], label='Històric', color='black', alpha=0.5)
plt.plot(escenaris_simulacio['Sequera_Global'][any_visualitzar], label='Sequera (-30%)', color='red', alpha=0.7)
plt.plot(escenaris_simulacio['Canvi_Climatic'][any_visualitzar], label='Canvi Climàtic (Estacional)', color='orange', linestyle='--')
plt.title(f'Comparativa Escenaris Aportacions ({any_visualitzar})')
plt.ylabel('hm3/h')
plt.legend()
plt.show()

#%%
def generar_escenaris_avancats(serie_aportacions_base):
    """
    Genera escenaris basats en projeccions ACA/TICCC i stress-test.
    Assumeix que serie_aportacions_base té un índex Datetime.
    """
    escenaris = {}
    
    # --- 1. BASE ---
    escenaris['Historic'] = serie_aportacions_base.copy()
    
    # --- 2. CANVI CLIMÀTIC ACA 2050 (Ajust estacional) ---
    # Objectiu: Reducció mitjana anual ~20%.
    # Estiu/Primavera molt castigats, Hivern menys.
    s_cc = serie_aportacions_base.copy()
    
    # Definim factors per mesos (Font: aproximació a projeccions regionals)
    # Hivern (12, 1, 2, 3): -10%
    # Transició (4, 5, 10, 11): -20%
    # Estiu pur (6, 7, 8, 9): -40%
    factors = {
        1: 0.95, 2: 0.95, 3: 0.95, 12: 0.95,       # Hivern
        4: 0.9, 5: 0.9, 10: 0.9, 11: 0.9,      # Primavera/Tardor
        6: 0.6, 7: 0.6, 8: 0.6, 9: 0.6         # Estiu
    }
    
    # Apliquem el factor segons el mes
    # Utilitzem map per velocitat
    factors_series = s_cc.index.month.map(factors)
    s_cc = s_cc * factors_series
    
    escenaris['Clima_2050'] = s_cc
    
    # --- 3. SEQUERA PERSISTENT (Stress-test temporal) ---
    # Simulem què passaria si la sequera 2021-2024 hagués començat un any abans.
    # Desplacem temporalment el patró de sequera mantenint la resta intacte.
    
    def generar_sequera_avancada(serie_base, data_inici_sequera='2021-07-01', anys_avanc=1):
        """
        Avança l'inici de la sequera observada un nombre d'anys determinat.
        
        Paràmetres:
        -----------
        serie_base : pd.Series
            Sèrie temporal d'aportacions amb índex DatetimeIndex.
        data_inici_sequera : str
            Data aproximada on comença el descens cap a la sequera.
        anys_avanc : int
            Quants anys volem avançar l'inici de la sequera.
        
        Retorna:
        --------
        pd.Series : Sèrie modificada amb la sequera avançada.
        """
        s_avancat = serie_base.copy()
        
        # Convertim a Timestamp
        data_inici = pd.Timestamp(data_inici_sequera)
        data_avancada = data_inici - pd.DateOffset(years=anys_avanc)
        
        # Extraiem les dades de sequera (des de data_inici fins al final)
        dades_sequera = serie_base.loc[data_inici:].values
        
        # Trobem l'índex on començar a sobreescriure
        mask_avancat = s_avancat.index >= data_avancada
        idx_avancat = np.where(mask_avancat)[0][0]
        
        # Calculem quants punts podem sobreescriure sense sortir de la sèrie
        n_disponible = len(s_avancat) - idx_avancat
        n_sequera = len(dades_sequera)
        n_a_copiar = min(n_disponible, n_sequera)
        
        # Sobreescrivim el període avançat amb les dades de sequera
        s_avancat.iloc[idx_avancat : idx_avancat + n_a_copiar] = dades_sequera[:n_a_copiar]
        
        return s_avancat
    
    # def generar_sequera_persistent(serie_base, data_inici_sequera='2021-07-01', anys_avanc=1):
    #     """
    #     Genera un escenari de sequera persistent avançant l'inici i repetint
    #     el penúltim any (el més sec) per omplir el buit temporal generat.
        
    #     Paràmetres:
    #     -----------
    #     serie_base : pd.Series
    #         Sèrie temporal d'aportacions amb índex DatetimeIndex.
    #     data_inici_sequera : str
    #         Data aproximada on comença el descens cap a la sequera.
    #     anys_avanc : int
    #         Quants anys volem avançar l'inici (i repeticions del penúltim any).
        
    #     Retorna:
    #     --------
    #     pd.Series : Sèrie modificada amb la sequera persistent.
    #     """
    #     s_persistent = serie_base.copy()
        
    #     # Convertim a Timestamp
    #     data_inici = pd.Timestamp(data_inici_sequera)
    #     data_avancada = data_inici - pd.DateOffset(years=anys_avanc)
        
    #     # Identifiquem anys clau (calendari)
    #     any_final = serie_base.index[-1].year
    #     any_penultim = any_final - 1
        
    #     # Màscares temporals
    #     mask_sequera = serie_base.index >= data_inici
    #     mask_ultim = serie_base.index.year == any_final
    #     mask_penultim = serie_base.index.year == any_penultim
        
    #     # Extraiem les parts: sequera sense últim any, penúltim any, últim any
    #     dades_sequera_sense_ultim = serie_base.loc[mask_sequera & ~mask_ultim].values
    #     dades_penultim = serie_base.loc[mask_penultim].values
    #     dades_ultim = serie_base.loc[mask_ultim].values
        
    #     # Construïm: [sequera sense últim] + [penúltim × anys_avanc] + [últim]
    #     dades_repetides = np.tile(dades_penultim, anys_avanc)
    #     dades_final = np.concatenate([dades_sequera_sense_ultim, dades_repetides, dades_ultim])
        
    #     # Trobem l'índex on començar a sobreescriure
    #     mask_avancat = s_persistent.index >= data_avancada
    #     idx_avancat = np.where(mask_avancat)[0][0]
        
    #     # Calculem quants punts podem sobreescriure sense sortir de la sèrie
    #     n_disponible = len(s_persistent) - idx_avancat
    #     n_a_copiar = min(n_disponible, len(dades_final))
        
    #     # Sobreescrivim el període avançat amb les dades construïdes
    #     s_persistent.iloc[idx_avancat : idx_avancat + n_a_copiar] = dades_final[:n_a_copiar]
        
    #     return s_persistent 
 
    # def generar_sequera_persistent(serie_base, data_inici_sequera='2021-07-01', anys_avanc=1, factor_reduccio=0.1):
    #    """
    #    Genera un escenari de sequera persistent avançant l'inici i repetint
    #    el penúltim any (el més sec) per omplir el buit temporal generat.
       
    #    Paràmetres:
    #    -----------
    #    serie_base : pd.Series
    #        Sèrie temporal d'aportacions amb índex DatetimeIndex.
    #    data_inici_sequera : str
    #        Data aproximada on comença el descens cap a la sequera.
    #    anys_avanc : int
    #        Quants anys volem avançar l'inici (i repeticions del penúltim any).
    #    factor_reduccio : float
    #        Factor multiplicador per les dades duplicades (< 1 = més sec).
    #        Ex: 0.8 = 20% menys aportacions, 0.5 = 50% menys.
       
    #    Retorna:
    #    --------
    #    pd.Series : Sèrie modificada amb la sequera persistent.
    #    """
    #    s_persistent = serie_base.copy()
       
    #    # Convertim a Timestamp
    #    data_inici = pd.Timestamp(data_inici_sequera)
    #    data_avancada = data_inici - pd.DateOffset(years=anys_avanc)
       
    #    # Identifiquem anys clau (calendari)
    #    any_final = serie_base.index[-1].year
    #    any_penultim = any_final - 1
       
    #    # Màscares temporals
    #    mask_sequera = serie_base.index >= data_inici
    #    mask_ultim = serie_base.index.year == any_final
    #    mask_penultim = serie_base.index.year == any_penultim
       
    #    # Extraiem les parts: sequera sense últim any, penúltim any, últim any
    #    dades_sequera_sense_ultim = serie_base.loc[mask_sequera & ~mask_ultim].values
    #    dades_penultim = serie_base.loc[mask_penultim].values
    #    dades_ultim = serie_base.loc[mask_ultim].values
       
    #    # Construïm: [sequera sense últim] + [penúltim × anys_avanc × factor] + [últim]
    #    dades_repetides = np.tile(dades_penultim, anys_avanc) * factor_reduccio  # ← CANVI
    #    dades_final = np.concatenate([dades_sequera_sense_ultim, dades_repetides, dades_ultim])
       
    #    # Trobem l'índex on començar a sobreescriure
    #    mask_avancat = s_persistent.index >= data_avancada
    #    idx_avancat = np.where(mask_avancat)[0][0]
       
    #    # Calculem quants punts podem sobreescriure sense sortir de la sèrie
    #    n_disponible = len(s_persistent) - idx_avancat
    #    n_a_copiar = min(n_disponible, len(dades_final))
       
    #    # Sobreescrivim el període avançat amb les dades construïdes
    #    s_persistent.iloc[idx_avancat : idx_avancat + n_a_copiar] = dades_final[:n_a_copiar]
       
    #    return s_persistent
   
    def generar_sequera_persistent(serie_base, data_inici_sequera='2021-07-01', anys_avanc=1, factors_reduccio=None):
        """
        Genera un escenari de sequera persistent avançant l'inici i repetint
        el penúltim any (el més sec) per omplir el buit temporal generat.
        
        Paràmetres:
        -----------
        serie_base : pd.Series
            Sèrie temporal d'aportacions amb índex DatetimeIndex.
        data_inici_sequera : str
            Data aproximada on comença el descens cap a la sequera.
        anys_avanc : int
            Quants anys volem avançar l'inici (i repeticions del penúltim any).
        factors_reduccio : dict o float, opcional
            - Si és dict: {any: factor} per cada any a modificar.
              Claus especials: 'duplicat' per l'any repetit, 'tots' per aplicar a tot.
              Ex: {2022: 0.8, 2023: 0.9, 'duplicat': 0.7}
            - Si és float: s'aplica només a les dades duplicades.
            - Si és None: no s'aplica cap reducció.
        
        Retorna:
        --------
        pd.Series : Sèrie modificada amb la sequera persistent.
        """
        s_persistent = serie_base.copy()
        
        # Normalitzar factors_reduccio
        if factors_reduccio is None:
            factors = {}
        elif isinstance(factors_reduccio, (int, float)):
            factors = {'duplicat': factors_reduccio}
        else:
            factors = factors_reduccio.copy()
        
        # Convertim a Timestamp
        data_inici = pd.Timestamp(data_inici_sequera)
        data_avancada = data_inici - pd.DateOffset(years=anys_avanc)
        
        # Identifiquem anys clau (calendari)
        any_final = serie_base.index[-1].year
        any_penultim = any_final - 1
        
        # Màscares temporals
        mask_sequera = serie_base.index >= data_inici
        mask_ultim = serie_base.index.year == any_final
        mask_penultim = serie_base.index.year == any_penultim
        
        # Extraiem les parts
        dades_sequera_sense_ultim = serie_base.loc[mask_sequera & ~mask_ultim].copy()
        dades_penultim = serie_base.loc[mask_penultim].values.copy()
        dades_ultim = serie_base.loc[mask_ultim].values.copy()
        
        # Aplicar factors als anys de la sequera original
        for any_obj in dades_sequera_sense_ultim.index.year.unique():
            factor = factors.get(any_obj, factors.get('tots', 1.0))
            if factor != 1.0:
                mask_any = dades_sequera_sense_ultim.index.year == any_obj
                dades_sequera_sense_ultim.loc[mask_any] *= factor
        
        # Aplicar factor a l'últim any
        factor_ultim = factors.get(any_final, factors.get('tots', 1.0))
        dades_ultim = dades_ultim * factor_ultim
        
        # Construïm dades duplicades amb el seu factor
        factor_duplicat = factors.get('duplicat', factors.get(any_penultim, factors.get('tots', 1.0)))
        dades_repetides = np.tile(dades_penultim, anys_avanc) * factor_duplicat
        
        # Concatenar
        dades_final = np.concatenate([dades_sequera_sense_ultim.values, dades_repetides, dades_ultim])
        
        # Trobem l'índex on començar a sobreescriure
        mask_avancat = s_persistent.index >= data_avancada
        idx_avancat = np.where(mask_avancat)[0][0]
        
        # Calculem quants punts podem sobreescriure sense sortir de la sèrie
        n_disponible = len(s_persistent) - idx_avancat
        n_a_copiar = min(n_disponible, len(dades_final))
        
        # Sobreescrivim el període avançat amb les dades construïdes
        s_persistent.iloc[idx_avancat : idx_avancat + n_a_copiar] = dades_final[:n_a_copiar]
        
        return s_persistent    
 
    def generar_sequera_persistent_v2(serie_base, data_inici='2020-07-01', data_fi='2021-07-01', factor_reduccio=0.6):
        """
        Genera un escenari de sequera persistent aplicant una reducció
        a un període específic, mantenint la resta de la sèrie intacta.
        
        Paràmetres:
        -----------
        serie_base : pd.Series
            Sèrie temporal d'aportacions amb índex DatetimeIndex.
        data_inici : str
            Data d'inici del període a modificar.
        data_fi : str
            Data de fi del període a modificar.
        factor_reduccio : float
            Factor multiplicatiu (0.6 = reducció del 40%).
        
        Retorna:
        --------
        pd.Series : Sèrie modificada amb el període reduït.
        """
        s_modificat = serie_base.copy()
        
        # Màscara del període a modificar
        mask_periode = (s_modificat.index >= data_inici) & (s_modificat.index < data_fi)
        
        # Aplicar reducció
        s_modificat.loc[mask_periode] = s_modificat.loc[mask_periode] * factor_reduccio
        
        return s_modificat    

    # --- 5. SEQUERA ANTICIPADA (Reducció pre-sequera) ---
    escenaris['Sequera_Anticipada'] = generar_sequera_persistent_v2(
        serie_aportacions_base,
        data_inici='2020-07-01',
        data_fi='2021-07-01',
        factor_reduccio=0.85  # 40% menys d'aportacions
    )
        
    # Apliquem la funció
    escenaris['Sequera_Extensa'] = generar_sequera_persistent(
        serie_aportacions_base, 
        data_inici_sequera='2021-07-01',  # Ajusta segons el gràfic exacte
        anys_avanc=1,
        factors_reduccio={
        'duplicat': 0.5, # 50% menys l'any duplicat
        2023: 0.5,
        2024: 0.5
        })


    # --- 4. TORRENCIALITAT (Menys dies de pluja, més intensitat) ---
    s_torr = serie_aportacions_base.copy()
    
    # Llindar del 90% (el 10% dels dies més plujosos)
    llindar_alt = s_torr.quantile(0.90)
    
    # Màscares
    mask_pluja_baixa = s_torr < llindar_alt
    mask_pluja_alta = s_torr >= llindar_alt
    
    # Lògica:
    # El cabal base i pluges petites es redueixen un 30% (més dies secs)
    s_torr[mask_pluja_baixa] = s_torr[mask_pluja_baixa] * 0.6
    
    # Els pics forts augmenten un 20%
    s_torr[mask_pluja_alta] = s_torr[mask_pluja_alta] * 1.2
    
    # Opcional: Re-normalitzar perquè la mitjana anual baixi un 15% globalment?
    # De moment ho deixem així, ja que reduir el 90% dels dies ja baixarà molt la mitjana.
    
    escenaris['Torrencialitat'] = s_torr
    
    return escenaris

escenaris_simulacio = generar_escenaris_avancats(df_balanc.aportacio_natural_hm3h)


# Visualitzem un any per veure diferències
plt.figure(figsize=(12,5))
any_visualitzar = '2022' # Tria un any que tinguis dades
plt.plot(escenaris_simulacio['Historic'][any_visualitzar], label='Històric', color='black', alpha=0.5)
plt.plot(escenaris_simulacio['Clima_2050'][any_visualitzar], label='Clima 2050', color='red', alpha=0.7)
# plt.plot(escenaris_simulacio['Sequera_Extensa'][any_visualitzar], label='Sequera Extesa', color='orange', alpha=1)
# plt.plot(escenaris_simulacio['Sequera_Anticipada'][any_visualitzar], label='Sequera Extesa', color='orange', alpha=1)
plt.plot(escenaris_simulacio['Torrencialitat'][any_visualitzar], label='Torrencialitat', color='green', linestyle='--')
plt.title(f'Comparativa Escenaris Aportacions ({any_visualitzar})')
plt.ylabel('hm3/h')
plt.legend()
plt.show()

escenaris_simulacio['Clima_2050'].sum()/escenaris_simulacio['Historic'].sum()
escenaris_simulacio['Torrencialitat'].sum()/escenaris_simulacio['Historic'].sum()

escenaris_simulacio['Sequera_Extensa'].sum()/escenaris_simulacio['Historic'].sum()

#%%

def reconstruir_volum_escenari(
    volum_historic,           # pd.Series: V_hist sense intervencions
    aportacions_historiques,  # pd.Series: I_hist
    aportacions_escenari,     # pd.Series: I_esc
    capacitat_max,            # float: hm³
    outflow_base=None         # pd.Series opcional: si no es passa, es calcula implícitament
):
    """
    Reconstrueix la sèrie de volums d'embassament per a un escenari pertorbat.
    
    Mètode: Aplica la diferència d'aportacions de manera acumulativa,
    respectant els límits físics del sistema.
    """
    
    # Pertorbació d'aportacions
    delta_aportacions = aportacions_escenari - aportacions_historiques
    
    # Opció simple: pertorbació acumulada amb clipping
    delta_acumulat_hm3 = delta_aportacions.cumsum()
    
    # Convertir a percentatge
    delta_acumulat_pct = delta_acumulat_hm3 * (100.0 / capacitat_max)    
    
    volum_escenari = volum_historic + delta_acumulat_pct

    # Aplicar límits físics
    # volum_escenari = volum_escenari_brut.clip(lower=0, upper=capacitat_max)
   
   
    return volum_escenari


# 1-escenaris_simulacio['Sequera_Extensa'].sum()/escenaris_simulacio['Historic'].sum()
print(1-escenaris_simulacio['Sequera_Anticipada'].sum()/escenaris_simulacio['Historic'].sum())

# reconstruir_volum_escenari(hydro_base_level,escenaris_simulacio['Historic'],escenaris_simulacio['Clima_2050'],max_capacity_int)
# reconstruir_volum_escenari(hydro_base_level,escenaris_simulacio['Historic'],escenaris_simulacio['Torrencialitat'],max_capacity_int).plot()
reconstruir_volum_escenari(hydro_base_level,escenaris_simulacio['Historic'],escenaris_simulacio['Sequera_Extensa'],max_capacity_int).plot()
reconstruir_volum_escenari(hydro_base_level,escenaris_simulacio['Historic'],escenaris_simulacio['Sequera_Anticipada'],max_capacity_int).plot()

hydro_esc1_level = reconstruir_volum_escenari(hydro_base_level,escenaris_simulacio['Historic'],escenaris_simulacio['Sequera_Extensa'],max_capacity_int).dropna()
hydro_esc2_level = reconstruir_volum_escenari(hydro_base_level,escenaris_simulacio['Historic'],escenaris_simulacio['Sequera_Anticipada'],max_capacity_int).dropna()
# hydro_esc2_level.plot()
#%%

def calibrar_elasticitat(energia_mensual, aportacions):
    """
    Estima l'exponent β mitjançant regressió log-log.
    """
    aport_mensual = aportacions.resample('ME').sum()
    
    # Alineem
    idx = energia_mensual.index.intersection(aport_mensual.index)
    E = energia_mensual.loc[idx]
    I = aport_mensual.loc[idx]
    
    # Filtrem zeros
    mask = (E > 0) & (I > 0)
    E, I = E[mask], I[mask]
    
    # Regressió log-log: log(E) = α + β·log(I)
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.log(I), np.log(E)
    )
    
    print(f"Elasticitat estimada: β = {slope:.3f}")
    print(f"R² = {r_value**2:.3f}, p-value = {p_value:.2e}")
    
    return slope

# Ús:
beta_calibrat = calibrar_elasticitat(
    datos.energia_turbinada_mensual_internes, 
    df_balanc['aportacio_natural_hm3h']
)

#%%

def calibrar_gamma(nivell_internes, nivell_ebre):
    """
    Estima γ a partir de la correlació entre els canvis de nivell
    de les dues conques.
    
    γ alt → sèries independents
    γ baix → sèries molt acoblades
    """
    # Canvis mensuals de nivell (més robust que nivells absoluts)
    delta_int = nivell_internes.resample('ME').mean().diff().dropna()
    delta_ebre = nivell_ebre.resample('ME').mean().diff().dropna()
    
    # Alineem
    idx = delta_int.index.intersection(delta_ebre.index)
    
    # Correlació de Pearson
    correlacio = delta_int.loc[idx].corr(delta_ebre.loc[idx])
    
    # γ = 1 - |correlació| 
    # (si correlació alta → γ baix, Ebre segueix internes)
    gamma = 1 - abs(correlacio)
    
    print(f"Correlació Δnivells: {correlacio:.3f}")
    print(f"γ inferit: {gamma:.3f}")
    
    return gamma

# Ús:
gamma_calibrat = calibrar_gamma(
    precomputed['hydro_level_int'], 
    precomputed['hydro_level_ebro']
)



#%%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generar_sintetic_controlat(serie_historica, n_anys_simulats=1, llavor=None, limit_sigma=2.5):
    """
    Genera sèries sintètiques amb dos controls de seguretat:
    1. Truncament del soroll gaussià (evita valors > 2.5 sigmes).
    2. Límit absolut basat en el màxim històric.
    """
    if llavor:
        np.random.seed(llavor)

    # Dades de referència per al límit físic
    MAX_HISTORIC = serie_historica.max()
    
    # 1. Transformació Logarítmica
    log_serie = np.log(serie_historica + 1e-6)

    # 2. Captura d'estadístiques (Amb una finestra més suau per estabilitzar)
    dayofyear = log_serie.index.dayofyear
    
    # Suavitzat una mica més agressiu (window=30 dies) per evitar pics de variança puntuals
    mitjanes = log_serie.groupby(dayofyear).mean().rolling(window=30, center=True, min_periods=1).mean()
    stds = log_serie.groupby(dayofyear).std().rolling(window=30, center=True, min_periods=1).mean()

    # Emplenem possibles forats als extrems (gener/desembre) degut al rolling
    mitjanes = mitjanes.fillna(method='bfill').fillna(method='ffill')
    stds = stds.fillna(method='bfill').fillna(method='ffill')

    # Convertim a arrays per accés ràpid (índex 0..365)
    # Atenció: dayofyear va de 1 a 366. Farem un array de 367 per indexar directe.
    mu_arr = np.zeros(367)
    sigma_arr = np.zeros(367)
    mu_arr[mitjanes.index] = mitjanes.values
    sigma_arr[stds.index] = stds.values

    # # 3. Càlcul d'Autocorrelació (Lag-1) global
    # # (Simplificació robusta: usem una rho global)
    # anomalies = (log_serie - mitjanes.loc[dayofyear]) / stds.loc[dayofyear]
    # rho = anomalies.autocorr(lag=1)
    
    # 3. CÀLCUL D'ANOMALIES (CORREGIT)
    # Extraiem els valors com a arrays numpy per evitar errors d'índexs
    vals_mu = mitjanes.loc[dayofyear].values
    vals_sigma = stds.loc[dayofyear].values
    
    # Operació vectoritzada sense índexs
    anomalies_values = (log_serie.values - vals_mu) / vals_sigma
    
    # Convertim a Sèrie per poder usar .autocorr()
    anomalies = pd.Series(anomalies_values)
    
    rho = anomalies.autocorr(lag=1)
    
    # Si la correlació és massa alta (>0.9), el model es pot "encallar" en valors alts. 
    # La limitem una mica per seguretat.
    rho = min(rho, 0.85) 

    # 4. Simulació
    dates_sim = pd.date_range(start=serie_historica.index[0], periods=n_anys_simulats*365, freq='D')
    n_steps = len(dates_sim)
    z_sintetic = np.zeros(n_steps)
    
    # Inicialització
    z_sintetic[0] = 0 
    
    std_soroll = np.sqrt(1 - rho**2)

    for t in range(1, n_steps):
        # Generem soroll aleatori
        noise = np.random.normal(0, std_soroll)
        
        # --- CORRECCIÓ 1: TRUNCAMENT ESTADÍSTIC ---
        # Si el soroll és molt extrem (ex: 4 sigmes), el retallem.
        # Això evita que l'exponencial es dispari.
        noise = np.clip(noise, -limit_sigma * std_soroll, limit_sigma * std_soroll)
        
        # Càlcul AR(1)
        val_proposat = rho * z_sintetic[t-1] + noise
        
        # Opcional: També podem limitar el valor Z acumulat perquè no derivi massa
        val_proposat = np.clip(val_proposat, -3.0, 3.0) 
        
        z_sintetic[t] = val_proposat

    # 5. Reconstrucció
    doy_sim = dates_sim.dayofyear
    
    # Recuperem mu i sigma per cada dia simulat
    mus_t = mu_arr[doy_sim]
    sigmas_t = sigma_arr[doy_sim]
    
    log_val_sim = z_sintetic * sigmas_t + mus_t
    serie_final = np.exp(log_val_sim) - 1e-6
    
    # --- CORRECCIÓ 2: LÍMIT FÍSIC (Hard Cap) ---
    # Cap valor pot superar el màxim històric vist.
    serie_final = np.minimum(serie_final, MAX_HISTORIC)
    
    # Evitar negatius
    serie_final = np.maximum(serie_final, 0)
    
    return pd.Series(serie_final, index=dates_sim, name="Sintètic_Acotat")

# --- PROVA ---
# inflows_clean ve dels passos anteriors
sintetica = generar_sintetic_controlat(inflows_clean, n_anys_simulats=5, limit_sigma=2.0)

print(f"Max Històric: {inflows_clean.max():.2f}")
print(f"Max Sintètic: {sintetica.max():.2f}")

plt.figure(figsize=(12,5))
plt.plot(sintetica.iloc[:730], label='Sintètic (2 anys)', color='green', alpha=0.8)
plt.axhline(y=inflows_clean.max(), color='red', linestyle='--', label='Sostre Històric')
plt.title("Generació Sintètica Acotada")
plt.legend()
plt.show()

# --- ÚS ---
# Suposem que 'inflows_clean' és la teva sèrie històrica neta
serie_sintetica = generar_sintetic_thomas_fiering(inflows_clean, n_anys_simulats=3)

# --- VALIDACIÓ VISUAL ---
plt.figure(figsize=(14, 6))

# 1. Hidrograma
plt.subplot(1, 2, 1)
plt.plot(inflows_clean.resample('D').last().iloc[:365*3].values, label='Any Històric (Exemple)', alpha=0.7)
plt.plot(serie_sintetica.iloc[:365*3].values, label='Any Sintètic', alpha=0.7, color='red')
plt.title("Comparativa Hidrogrames (1 any)")
plt.ylabel("Aportació (hm3/h)")
plt.legend()

# 2. Distribució (Histograma) - Per veure si capturem la 'cua' (Gamma)
plt.subplot(1, 2, 2)
sns.kdeplot(inflows_clean.resample('D').last(), label='Històric', fill=True)
sns.kdeplot(serie_sintetica, label='Sintètic', fill=True, color='red')
plt.title("Comparativa de Distribucions (Densitat)")
plt.xlim(0, inflows_clean.quantile(0.99)) # Tallem els extrems molt bèsties per veure-ho bé
plt.legend()

plt.tight_layout()
plt.show()