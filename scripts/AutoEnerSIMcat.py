# -*- coding: utf-8 -*-
"""
Mòdul de Generació d'Escenaris Sintètics - Versió Optimitzada
=============================================================

Aquest mòdul conté la funció principal de simulació d'escenaris amb
optimitzacions de rendiment per a execucions múltiples.

Millores respecte a la versió original:
- Precomputació de dades invariants (extracció autoconsum, reescalat, etc.)
- Retrocompatibilitat completa (funciona igual si no es passa precomputed)

Autor: Víctor García Carrasco
Data: 2024
"""

import pandas as pd
import numpy as np
from typing import Optional
import time


# =============================================================================
# FUNCIÓ DE PRECOMPUTACIÓ (EXECUTAR UNA SOLA VEGADA)
# =============================================================================

def precomputar_dades_base(
    df_demanda: pd.Series,
    df_solar: pd.Series,
    df_eolica: pd.Series,
    df_cogeneracion: pd.Series,
    df_autoconsum: pd.Series,
    df_potencia_historica: pd.DataFrame,
    df_capacidad_internes: pd.DataFrame,
    df_capacidad_ebre: pd.DataFrame,
    potencia_cogeneracio_max: float,
    start_date: str = '2016-01-01',
    end_date: str = '2024-12-31',
    pr_autoconsum: float = 0.75,
    verbose: bool = True
) -> dict:
    """
    Precalcula les dades invariants que no depenen dels paràmetres de l'escenari.
    
    Aquesta funció s'ha d'executar UNA SOLA VEGADA abans de llançar múltiples
    escenaris. Els resultats es passen a generar_escenario_sintetico() mitjançant
    el paràmetre 'precomputed'.
    
    Dades que es precalculen:
    - Sèries de renovables reescalades a potència vigent (sense pesos)
    - Cogeneració amb correcció de tendència aplicada
    - Demanda neta (sense autoconsum històric)
    - Nivells dels embassaments reindexats a freqüència horària
    
    :param df_demanda: Sèrie horària de demanda elèctrica
    :param df_solar: Sèrie horària de generació solar
    :param df_eolica: Sèrie horària de generació eòlica
    :param df_cogeneracion: Sèrie horària de cogeneració
    :param df_autoconsum: Sèrie d'autoconsum fotovoltaic
    :param df_potencia_historica: DataFrame amb l'evolució de potència instal·lada
    :param df_capacidad_internes: Nivells dels embassaments de les conques internes
    :param df_capacidad_ebre: Nivells dels embassaments de la conca de l'Ebre
    :param potencia_cogeneracio_max: Potència màxima de cogeneració per al clipping
    :param start_date: Data d'inici del període d'anàlisi
    :param end_date: Data de fi del període d'anàlisi
    :param pr_autoconsum: Performance ratio per a l'autoconsum (default: 0.75)
    :param verbose: Si True, mostra informació del procés
    
    :return: Diccionari amb totes les dades precomputades
    """
    
    if verbose:
        print("=" * 60)
        print("PRECOMPUTACIÓ DE DADES BASE")
        print("=" * 60)
        t_start = time.time()
    
    # -------------------------------------------------------------------------
    # 1. REESCALAT DE RENOVABLES A POTÈNCIA VIGENT (sense pesos)
    # -------------------------------------------------------------------------
    if verbose:
        print("  [1/4] Reescalant renovables a potència vigent...")
    
    # Solar (Fotovoltaica + Termosolar)
    # NOTA: Usem .loc[:end_date].iloc[-1] per obtenir el valor més recent <= end_date
    potencia_solar_total = df_potencia_historica[['Fotovoltaica', 'Termosolar']].sum(axis=1)
    potencia_solar_ref = potencia_solar_total.loc[:end_date].iloc[-1]
    ratio_solar = (potencia_solar_ref / potencia_solar_total).resample('h').ffill()
    solar_base = (df_solar * ratio_solar)[start_date:end_date]
    
    # Eòlica
    potencia_eolica_ref = df_potencia_historica['Eòlica'].loc[:end_date].iloc[-1]
    ratio_eolica = (potencia_eolica_ref / df_potencia_historica['Eòlica']).resample('h').ffill()
    eolica_base = (df_eolica * ratio_eolica)[start_date:end_date]
    
    # -------------------------------------------------------------------------
    # 2. COGENERACIÓ AMB CORRECCIÓ DE TENDÈNCIA
    # -------------------------------------------------------------------------
    if verbose:
        print("  [2/4] Aplicant correcció de tendència a cogeneració...")
    
    potencia_cogen_ref = df_potencia_historica['Cogeneració'].loc[:end_date].iloc[-1]
    ratio_cogen = (potencia_cogen_ref / df_potencia_historica['Cogeneració']).resample('h').ffill()
    cogen_reescalada = df_cogeneracion * ratio_cogen
    
    # Calcular tendència suau (mitjana mòbil centrada)
    tendencia = cogen_reescalada.rolling(
        window=365 * 24, 
        center=True, 
        min_periods=1
    ).mean()
    
    # Definir regió de referència (2024) → factor = 1
    mask_ref = (cogen_reescalada.index.year >= 2024)
    
    # Calcular nivell de referència = mitjana de la tendència en 2024
    nivel_ref = tendencia.loc[mask_ref].mean()
    
    # Factor variable: només aplica fora de la regió de referència
    factor = pd.Series(1.0, index=cogen_reescalada.index)
    factor.loc[~mask_ref] = nivel_ref / tendencia.loc[~mask_ref]
    
    # Aplicar factor i fer clipping
    cogen_base = (cogen_reescalada * factor).clip(upper=potencia_cogeneracio_max)
    cogen_base = cogen_base[start_date:end_date]
    
    # -------------------------------------------------------------------------
    # 3. EXTRACCIÓ DE L'AUTOCONSUM HISTÒRIC
    # -------------------------------------------------------------------------
    if verbose:
        print("  [3/4] Extraient autoconsum històric de la demanda...")
    
    demanda_slice = df_demanda[start_date:end_date]
    autoconsum_slice = df_autoconsum[start_date:end_date]
    
    # Cridem la funció d'extracció amb solar_base (pes=1)
    demanda_neta, autoconsum_historic = extraer_autoconsumo(
        demanda_slice, 
        solar_base, 
        autoconsum_slice, 
        pr=pr_autoconsum
    )
    
    # -------------------------------------------------------------------------
    # 4. NIVELLS DELS EMBASSAMENTS (REINDEX A HORARI)
    # -------------------------------------------------------------------------
    if verbose:
        print("  [4/4] Reindexant nivells d'embassaments a freqüència horària...")
    
    idx_horari = demanda_neta.index
    
    # Multiplicar per 100 per tenir percentatge i reindexar
    hydro_level_int = (100 * df_capacidad_internes.squeeze()).reindex(
        idx_horari, method='ffill'
    )
    hydro_level_ebro = (100 * df_capacidad_ebre.squeeze()).reindex(
        idx_horari, method='ffill'
    )
    
    # -------------------------------------------------------------------------
    # EMPAQUETAR RESULTATS
    # -------------------------------------------------------------------------
    precomputed = {
        # Sèries base (sense pesos aplicats)
        'solar_base': solar_base,
        'eolica_base': eolica_base,
        'cogen_base': cogen_base,
        
        # Demanda processada
        'demanda_neta': demanda_neta,
        'autoconsum_historic': autoconsum_historic,
        
        # Nivells hidràulics
        'hydro_level_int': hydro_level_int,
        'hydro_level_ebro': hydro_level_ebro,
        
        # Metadades
        'start_date': start_date,
        'end_date': end_date,
        'pr_autoconsum': pr_autoconsum
    }
    
    if verbose:
        t_elapsed = time.time() - t_start
        print("-" * 60)
        print(f"  Precomputació completada en {t_elapsed:.2f} segons")
        print(f"  Període: {start_date} a {end_date}")
        print(f"  Registres horaris: {len(demanda_neta):,}")
        print("=" * 60)
    
    return precomputed


# =============================================================================
# FUNCIÓ PRINCIPAL DE GENERACIÓ D'ESCENARIS (VERSIÓ OPTIMITZADA)
# =============================================================================

def generar_escenario_sintetico(
    # --- DADES D'ENTRADA (DataFrames i Series) ---
    df_demanda: pd.Series,
    df_nucleares_base: pd.DataFrame,
    df_cogeneracion: pd.Series,
    df_solar: pd.Series,
    df_eolica: pd.Series,
    df_autoconsum: pd.Series,
    df_potencia_historica: pd.DataFrame,
    df_capacidad_internes: pd.DataFrame,
    df_capacidad_ebre: pd.DataFrame,
    energia_turbinada_mensual_internes: pd.Series,
    energia_turbinada_mensual_ebre: pd.Series,
        
    # --- PARÀMETRES DE CONFIGURACIÓ DE L'ESCENARI ---
    nucleares_activas: list = [True, True, True],
    pesos: dict = {'solar': 1, 'wind': 1, 'dem': 1, 'cog': 1, 'auto': 1},
    baterias_config: list = [500, 2000, 0.8, 0],
    max_salto_hidro_pct: float = 5.0,
    optimizador_hidro: str = 'rapido',
       
    # --- PARÀMETRES FÍSICS DEL MODEL ---
    potencia_max_hidro: dict = None,
    sensibilidad_hidro: dict = None,
    capacidad_max_hidro: dict = None,
    umbral_overflow_pct: dict = {'ebro': 75.0, 'int': 75.0},
    
    # --- RANGO DE FECHAS ---
    start_date: str = '2016-01-01',
    end_date: str = '2024-12-31',
    
    # --- NOU: DADES PRECOMPUTADES (OPCIONAL) ---
    precomputed: Optional[dict] = None
    
) -> tuple[pd.DataFrame, dict]:
    """
    Genera un escenari energètic sintètic complet a partir d'un conjunt de paràmetres.

    Aquesta funció encapsula tot el procés:
    1. Construcció del DataFrame base segons la configuració nuclear i de pesos.
    2. Simulació de la generació hidràulica amb lògica de suavitzat.
    3. Simulació de l'emmagatzematge amb bateries/bombeig.
    4. Càlcul de la generació tèrmica residual (Gas+Imports).
    5. Càlcul d'un conjunt de mètriques clau de l'escenari.

    OPTIMITZACIÓ: Si es passa el paràmetre 'precomputed' (resultat de 
    precomputar_dades_base()), s'eviten càlculs redundants i s'accelera
    significativament l'execució en bucles d'escenaris múltiples.

    :param precomputed: Diccionari amb dades precomputades (opcional).
                        Si és None, es calculen dins la funció (comportament original).
    :param [altres params]: Veure docstring original.
    
    :return: Tupla (df_escenari, metriques)
    """

    # =========================================================================
    # ETAPA 1: CONFIGURACIÓ NUCLEAR
    # =========================================================================
    nombres_nucleares = ['Asco1', 'Asco2', 'Vandellos2']
    nucleares_a_usar = [nombre for nombre, activo in zip(nombres_nucleares, nucleares_activas) if activo]
    
    if nucleares_a_usar:
        df_nuclear_total = df_nucleares_base[nucleares_a_usar].sum(axis=1).resample('h').ffill()
    else:
        df_nuclear_total = pd.Series(0, index=df_demanda.index).resample('h').ffill()

    # =========================================================================
    # ETAPA 2: CONSTRUCCIÓ I REESCALAT DEL DATAFRAME INICIAL
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # BRANCA OPTIMITZADA: Usar dades precomputades si estan disponibles
    # -------------------------------------------------------------------------
    if precomputed is not None:
        # Extreure dades precomputades
        solar_base = precomputed['solar_base']
        eolica_base = precomputed['eolica_base']
        cogen_base = precomputed['cogen_base']
        demanda_neta = precomputed['demanda_neta']
        autoconsum_historic = precomputed['autoconsum_historic']
        hydro_level_int = precomputed['hydro_level_int']
        hydro_level_ebro = precomputed['hydro_level_ebro']
        pr_autoconsum = precomputed.get('pr_autoconsum', 0.75)
        
        # Construir DataFrame aplicant només els pesos
        df_sintetic = pd.DataFrame({
            'Demanda': demanda_neta.copy(),
            'Nuclear': df_nuclear_total[start_date:end_date]
        })
        
        df_sintetic['Solar_w'] = pesos['solar'] * solar_base
        df_sintetic['Eòlica_w'] = pesos['wind'] * eolica_base
        df_sintetic['Cogen_w'] = pesos['cog'] * cogen_base
        
        df_sintetic = df_sintetic.dropna()
        
        # Aplicar pes a demanda i reinsertar autoconsum amb nova capacitat
        df_sintetic['Demanda'] = pesos['dem'] * df_sintetic['Demanda']
        potencia_autoconsum_ref = df_potencia_historica['Autoconsum'].loc[:end_date].iloc[-1]
        df_sintetic['Demanda'], autoconsum_estimat = insertar_autoconsumo(
            df_sintetic['Demanda'],
            df_sintetic['Solar_w'], 
            pesos['auto'] * potencia_autoconsum_ref, 
            pr=pr_autoconsum
        )
        
        # Afegir nivells hidràulics (ja precomputats)
        df_sintetic['Hydro_Level_int'] = hydro_level_int.reindex(df_sintetic.index)
        df_sintetic['Hydro_Level_ebro'] = hydro_level_ebro.reindex(df_sintetic.index)
    
    # -------------------------------------------------------------------------
    # BRANCA ORIGINAL: Calcular tot des de zero (retrocompatibilitat)
    # -------------------------------------------------------------------------
    else:
        df_sintetic = pd.concat([
            df_demanda[start_date:end_date],
            df_nuclear_total[start_date:end_date]
        ], axis=1)
        df_sintetic.columns = ['Demanda', 'Nuclear']

        # Reescalat de renovables i cogeneració
        # NOTA: Usem .loc[:end_date].iloc[-1] per obtenir el valor més recent <= end_date
        potencia_solar_total = df_potencia_historica[['Fotovoltaica', 'Termosolar']].sum(axis=1)
        potencia_solar_ref = potencia_solar_total.loc[:end_date].iloc[-1]
        solar_reescalada = df_solar * (
            potencia_solar_ref / potencia_solar_total
        ).resample('h').ffill()
        
        potencia_eolica_ref = df_potencia_historica['Eòlica'].loc[:end_date].iloc[-1]
        eolica_reescalada = df_eolica * (
            potencia_eolica_ref / df_potencia_historica['Eòlica']
        ).resample('h').ffill()
        
        potencia_cogen_ref = df_potencia_historica['Cogeneració'].loc[:end_date].iloc[-1]
        cogen_reescalada = df_cogeneracion * (
            potencia_cogen_ref / df_potencia_historica['Cogeneració']
        ).resample('h').ffill()
        
        # Correcció de tendència de cogeneració
        tendencia = cogen_reescalada.rolling(window=365 * 24, center=True, min_periods=1).mean()
        mask_ref = (cogen_reescalada.index.year >= 2024)
        nivel_ref = tendencia.loc[mask_ref].mean()
        factor = pd.Series(1.0, index=cogen_reescalada.index)
        factor.loc[~mask_ref] = nivel_ref / tendencia.loc[~mask_ref]
        cogen_reescalada = cogen_reescalada * factor
        cogen_reescalada = cogen_reescalada.clip(upper=potencia_cogen_ref)

        df_sintetic['Solar_w'] = pesos['solar'] * solar_reescalada
        df_sintetic['Eòlica_w'] = pesos['wind'] * eolica_reescalada
        df_sintetic['Cogen_w'] = pesos['cog'] * cogen_reescalada
        
        df_sintetic = df_sintetic.dropna()

        # Recàlcul de la demanda considerant autoconsum
        df_sintetic.Demanda, autoconsum_historic = extraer_autoconsumo(
            df_sintetic.Demanda,
            df_sintetic.Solar_w, 
            df_autoconsum[start_date:end_date], 
            pr=0.75
        )
        df_sintetic.Demanda = pesos['dem'] * df_sintetic.Demanda
        potencia_autoconsum_ref = df_potencia_historica['Autoconsum'].loc[:end_date].iloc[-1]
        df_sintetic.Demanda, autoconsum_estimat = insertar_autoconsumo(
            df_sintetic.Demanda,
            df_sintetic.Solar_w, 
            pesos['auto'] * potencia_autoconsum_ref, 
            pr=0.75
        )
        
        # Nivells hidràulics
        hydro_hourly_int = df_capacidad_internes
        hydro_hourly_ebre = df_capacidad_ebre
        df_sintetic['Hydro_Level_int'] = 100 * hydro_hourly_int.reindex(df_sintetic.index, method='ffill')
        df_sintetic['Hydro_Level_ebro'] = 100 * hydro_hourly_ebre.reindex(df_sintetic.index, method='ffill')

    # =========================================================================
    # ETAPA 3: CÀLCUL DEL GAP INICIAL
    # =========================================================================
    df_sintetic['gap'] = (
        pesos['dem'] * df_sintetic['Demanda'] - 
        df_sintetic['Nuclear'] - 
        df_sintetic['Solar_w'] - 
        df_sintetic['Eòlica_w'] - 
        df_sintetic['Cogen_w']
    )
    
    df_sintetic = df_sintetic.dropna()
    
    # =========================================================================
    # ETAPA 4: SIMULACIÓ DE LA GENERACIÓ HIDRÀULICA
    # =========================================================================
    puntos_opt = 300 if optimizador_hidro == 'robusto' else 0
    
    df_sintetic = calcular_generacion_hidraulica(
        df_sintetic=df_sintetic,
        energia_turbinada_mensual_internes=energia_turbinada_mensual_internes,
        energia_turbinada_mensual_ebre=energia_turbinada_mensual_ebre,
        potencia_max_int=potencia_max_hidro['int'],
        potencia_max_ebro=potencia_max_hidro['ebro'],
        sensibility_int=sensibilidad_hidro['int'],
        sensibility_ebro=sensibilidad_hidro['ebro'],
        max_capacity_int=capacidad_max_hidro['int'],
        max_capacity_ebro=capacidad_max_hidro['ebro'],
        level_overflow_pct_int=umbral_overflow_pct['int'],
        level_overflow_pct_ebro=umbral_overflow_pct['ebro'],
        max_salto_pct_mensual=max_salto_hidro_pct,
        puntos_optimizacion=puntos_opt
    )
    df_sintetic.rename(columns={'Hidràulica': 'Hidráulica'}, inplace=True)

    df_sintetic.hydro_int = np.clip(df_sintetic.hydro_int, 0, potencia_max_hidro['int'])
    df_sintetic.hydro_ebro = np.clip(df_sintetic.hydro_ebro, 0, potencia_max_hidro['ebro'])
    df_sintetic.Hidráulica = df_sintetic.hydro_int + df_sintetic.hydro_ebro

    # =========================================================================
    # ETAPA 5: SIMULACIÓ DE BATERIES/BOMBEIG
    # =========================================================================
    # Gap residual sense fonts no renovables (per mètriques)
    df_sintetic['gap0'] = (
        pesos['dem'] * df_sintetic['Demanda'] - 
        df_sintetic['Solar_w'] - 
        df_sintetic['Eòlica_w'] - 
        df_sintetic['Hidráulica']
    )
    
    # Gap residual després de hidràulica
    df_sintetic['gap'] = (
        pesos['dem'] * df_sintetic['Demanda'] - 
        df_sintetic['Nuclear'] - 
        df_sintetic['Solar_w'] - 
        df_sintetic['Eòlica_w'] - 
        df_sintetic['Cogen_w'] - 
        df_sintetic['Hidráulica']
    )
      
    capacity, power = battery(baterias_config, df_sintetic['gap'])
    df_sintetic['gap'] = df_sintetic['gap'] - power
    df_sintetic['gap0'] = df_sintetic['gap0'] - power
    
    df_sintetic['Bateries'] = pd.Series(power, index=df_sintetic.index)
    df_sintetic.loc[df_sintetic['Bateries'] < 0, 'Bateries'] = 0
    df_sintetic['Càrrega'] = pd.Series(power * (-1), index=df_sintetic.index)
    df_sintetic.loc[df_sintetic['Càrrega'] < 0, 'Càrrega'] = 0
    df_sintetic['Càrrega'] = df_sintetic['Càrrega'] + df_sintetic['Demanda'] * pesos['dem']
    df_sintetic['SOC'] = pd.Series(capacity, index=df_sintetic.index)
    
    # =========================================================================
    # ETAPA 6: CÀLCUL DE LA GENERACIÓ RESIDUAL (TÈRMICA + IMPORTS)
    # =========================================================================
    df_sintetic['Gas+Imports'] = (
        df_sintetic['Demanda'] * pesos['dem'] -
        (df_sintetic['Nuclear'] + df_sintetic['Solar_w'] + 
         df_sintetic['Eòlica_w'] + df_sintetic['Cogen_w'] + 
         df_sintetic['Hidráulica'] + df_sintetic['Bateries'])
    )
    df_sintetic['Gas+Imports'] = df_sintetic['Gas+Imports'].clip(lower=0)
    
    # =========================================================================
    # ETAPA 7: CÀLCUL DE MÈTRIQUES FINALS
    # =========================================================================
    sample = df_sintetic.rename(columns={
        'Cogen_w': 'Cogeneració', 'Eòlica_w': 'Eòlica', 'Solar_w': 'Solar'
    })[['Demanda', 'Nuclear', 'Cogeneració', 'Eòlica', 'Solar', 'Hidráulica', 
        'Bateries', 'Gas+Imports', 'Càrrega', 'gap0']]
    
    total_demand = sample['Demanda'].sum() * pesos['dem']
    
    if total_demand < 1e-6:
        return df_sintetic, {}

    sample.loc[:, 'Total'] = (
        sample['Gas+Imports'] + sample['Cogeneració'] + sample['Nuclear'] + 
        sample['Solar'] + sample['Eòlica'] + sample['Hidráulica']
    )

    w_dem = pesos['dem']
    
    metrics = {
        'Wind %': sample['Eòlica'].sum() * 100 / total_demand,
        'Solar %': sample['Solar'].sum() * 100 / total_demand,
        'Autoconsum': autoconsum_estimat.sum() * 100 / total_demand,        
        'Hydro %': sample['Hidráulica'].sum() * 100 / total_demand,
        'Nuclear %': sample['Nuclear'].sum() * 100 / total_demand,
        'Cogeneració': sample['Cogeneració'].sum() * 100 / total_demand,
        'Cicles + Import.': sample['Gas+Imports'].sum() * 100 / total_demand,        
        'Batteries %': sample['Bateries'].sum() * 100 / total_demand,
        'Fossil+Imports %': (sample['Gas+Imports'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Low-carbon %': (sample[['Eòlica', 'Solar', 'Hidráulica', 'Nuclear']].sum().sum()) * 100 / total_demand,
        'Renewables %': sample[['Eòlica', 'Solar', 'Hidráulica']].sum().sum() * 100 / total_demand,
        'Ren.-coverage': 100 - (sample['Gas+Imports'] + sample['Nuclear'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Ren.cov-B': round((1 - sum(sample.gap0[sample.gap0 > 0]) / sum(w_dem * sample.Demanda)) * 100, 1),
        'Clean-coverage %': 100 - (sample['Gas+Imports'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Surpluses': ((sample['Total'] + sample['Bateries'] - sample['Càrrega']).sum()) * 100 / total_demand
    }

    df_sintetic.rename(columns={'Hidráulica': 'Hidràulica'}, inplace=True)

    return df_sintetic, {k: round(v, 2) for k, v in metrics.items()}


# =============================================================================
# FUNCIONS AUXILIARS (STUBS - SUBSTITUIR PER LES TEVES IMPLEMENTACIONS)
# =============================================================================
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


generacion_v2 = pd.read_excel(
    "generacion_v3.xlsx",
    index_col="fecha",
    parse_dates=True
)

# 1) Bins de Hydro_Level
bins = np.arange(0, 101, 10)   # cada 10 %
labels = (bins[:-1] + bins[1:]) / 2
df = generacion_v2[['Hydro_Level','Hidráulica']].dropna()
df['Hidráulica'] = df['Hidráulica']/17095
df['bin'] = pd.cut(df['Hydro_Level'], bins=bins, labels=labels)

# 2) Percentil 1% en cada bin
min_por_bin = df.groupby('bin', observed=False)['Hidráulica'].quantile(0.01).dropna()
x = min_por_bin.index.astype(float)
y = min_por_bin.values

# 1) Crea un interpolador lineal que permita extrapolar
f_hydro_min = interp1d(
    x, y,
    # kind='cubic',
    # kind = 'linear',
    kind = 'quadratic',
    fill_value='extrapolate',    # permite valores fuera de [x.min(), x.max()]
    assume_sorted=True
)

# 2) Función de consulta, con clipping de dominio a [40,100]
def hydro_min_for_level(level_pct):
    lvl = np.clip(level_pct, 35, 100)
    return float(f_hydro_min(lvl)/24)

def extraer_autoconsumo(demanda, fotovoltaica, autoconsumo, pr=0.75):
    """
    Extrae el autoconsumo de la demanda usando el perfil de generación FV.
    
    Parámetros:
    - demanda y fotovoltaica: series horaria de 'Demanda' y 'Fotovoltaica'
    - autoconsumo: serie horaria (potencia instalada de autoconsumo) [MW]
    - pr: performance ratio (por defecto 0.75)
    
    Retorna:
    - demanda_neta: demanda corregida por autoconsumo
    - autoconsumo_h: producción horaria de autoconsumo [MW]
    """

    # Crear perfil FV normalizado (forma horaria del recurso)
    perfil_fv = fotovoltaica
    perfil_fv_norm = perfil_fv / perfil_fv.max()

    # Producción estimada de autoconsumo
    autoconsumo_h = autoconsumo * perfil_fv_norm * pr

    # Alinear índices
    autoconsumo_h = autoconsumo_h.reindex(demanda.index)

    # Rellenar huecos:
    autoconsumo_h = autoconsumo_h.ffill().fillna(0)

    # La demanda observada es menor por el autoconsumo → se resta el autoconsumo
    # Para obtener la demanda "real" (sin autoconsumo), se suma:
    demanda_neta = demanda + autoconsumo_h
    demanda_neta = demanda_neta.clip(lower=0)

    return demanda_neta, autoconsumo_h


def insertar_autoconsumo(demanda, fotovoltaica, autoconsumo, pr=0.75):
    """
    Inserta el efecto del autoconsumo en la demanda corregida (Demanda_w).
    Reduce la demanda neta según el perfil fotovoltaico.

    Parámetros:
    - generacion: DataFrame con columnas 'Demanda_w' y 'Fotovoltaica'
    - autoconsumo: serie o valor (MW) de potencia instalada de autoconsumo
    - pr: performance ratio (por defecto 0.75)

    Retorna:
    - demanda_con_autoconsumo: demanda con autoconsumo reinsertado
    - autoconsumo_h: producción horaria de autoconsumo [MW]
    """

    # Crear perfil FV normalizado (forma horaria del recurso)
    perfil_fv = fotovoltaica
    perfil_fv_norm = perfil_fv / perfil_fv.max()

    # Si autoconsumo es un escalar, convertirlo en serie alineada
    if np.isscalar(autoconsumo):
        autoconsumo = pd.Series(autoconsumo, index=demanda.index)

    # Producción de autoconsumo
    autoconsumo_h = autoconsumo * perfil_fv_norm * pr

    # Alinear índices y rellenar huecos
    autoconsumo_h = autoconsumo_h.reindex(demanda.index)
    autoconsumo_h = autoconsumo_h.ffill().fillna(0)

    # Insertar efecto: la demanda observada es menor por el autoconsumo
    demanda_con_autoconsumo = demanda - autoconsumo_h
    demanda_con_autoconsumo = demanda_con_autoconsumo.clip(lower=0)

    return demanda_con_autoconsumo, autoconsumo_h


def battery(b, gap):
    """
    Simula el funcionamiento de una batería en respuesta a excedentes o déficits energéticos.
    
    Parámetros:
    b (list): Características de la batería
        b[0]: Potencia máxima de carga/descarga (kW)
        b[1]: Capacidad total de la batería (kWh)
        b[2]: Eficiencia de carga (%)
        b[-1]: Estado de carga actual (kWh)
    gap (array): Diferencia entre generación y demanda en cada intervalo (kW)
                 Valores negativos indican excedente, positivos indican déficit
    
    Retorna:
    b_t (array): Estado de carga de la batería en cada intervalo
    power_t (array): Potencia suministrada/absorbida por la batería en cada intervalo
    """
    
    b_t = np.zeros(len(gap))  # Array para almacenar el estado de carga en cada intervalo
    power_t = np.zeros(len(gap))  # Array para almacenar la potencia de la batería en cada intervalo

    max_charge_discharge = b[0]
    capacity_total = b[1]
    efficiency_charge = b[2]
    state_of_charge = b[-1]

    for k, g in enumerate(gap):
        power = np.sign(g) * min(max_charge_discharge, abs(g))

        if power < 0:  # Fase de carga (hay excedente de energía)
            power = -min(abs(power), (capacity_total - state_of_charge) / efficiency_charge)
            state_of_charge -= power * efficiency_charge
        elif power > 0:  # Fase de descarga (hay déficit de energía)
            power = min(power, state_of_charge)
            state_of_charge -= power

        # Registrar el nuevo estado de carga y la potencia suministrada/absorbida
        b_t[k] = state_of_charge
        power_t[k] = power
        
    return b_t, power_t

def _objetivo_optimizador(potencia_max, gap_slice, f_cuenca, hydro_min, energia_obj):
    """
    Función de coste para ser minimizada por scipy.optimize.minimize_scalar.
    Calcula el error absoluto entre la generación simulada y la energía objetivo.
    """
    # Calcula la generación total para una potencia máxima dada
    generacion_total = np.clip(gap_slice * f_cuenca, hydro_min, potencia_max).sum()
    # generacion_total = np.sum(np.clip(gap_slice * f_cuenca, hydro_min, potencia_max))
    # Devuelve el error absoluto a minimizar
    return abs(generacion_total - energia_obj)

# Función para redistribuir exceso (delta positivo), permitiendo usar gaps negativos si hace falta
def reajustar_por_overload_numpy(
    hidraulica_actual: np.ndarray, 
    gap_mes: np.ndarray, 
    energia_a_repartir: float, 
    potencia_max: float
) -> np.ndarray:
    """
    Versión NumPy-optimizada para repartir energía excedente.
    Reparte una cantidad de energía tratando primero de llenar la capacidad
    en horas con gap > 0, y luego de forma uniforme si es necesario.

    :param hidraulica_actual: Array NumPy con la generación hidráulica actual del mes.
    :param gap_mes: Array NumPy con los valores de 'gap' para el mes.
    :param energia_a_repartir: Exceso de energía a repartir (MWh).
    :param potencia_max: Potencia máxima por hora.
    :return: Array NumPy con la generación ajustada.
    """
    # Creamos una copia para no modificar el array original fuera de la función
    hid_ajustada = hidraulica_actual.copy()
    energia_restante = energia_a_repartir

    # Iteramos un número fijo de veces para asegurar la convergencia sin bucles infinitos
    for _ in range(10):  # 10 iteraciones suelen ser suficientes para converger
        if energia_restante < 1e-6:
            break

        # 1. Identificar horas candidatas (aquellas con capacidad disponible)
        # Usamos una máscara booleana, que es mucho más eficiente
        capacidad_restante_total = potencia_max - hid_ajustada
        mascara_candidatas = capacidad_restante_total > 1e-6

        # Si no hay ninguna hora con capacidad, salimos
        if not mascara_candidatas.any():
            break
            
        # Extraemos los valores solo de las horas candidatas para los cálculos
        gaps_candidatos = gap_mes[mascara_candidatas]
        cap_rest_candidatas = capacidad_restante_total[mascara_candidatas]

        # 2. Calcular los pesos para repartir la energía
        # Priorizamos horas con gap positivo
        gaps_positivos = np.clip(gaps_candidatos, 0, None)
        total_pos = gaps_positivos.sum()

        if total_pos > 1e-6:
            # Pesos proporcionales a los gaps positivos
            pesos = gaps_positivos / total_pos
        else:
            # Si no hay gaps positivos, reparto uniforme entre todas las candidatas
            num_candidatas = len(gaps_candidatos)
            pesos = np.ones(num_candidatas) / num_candidatas if num_candidatas > 0 else np.array([])
        
        if pesos.size == 0:
            break

        # 3. Calcular cuánta energía añadir en esta iteración
        # Intentamos repartir la energía restante según los pesos
        intento_reparto = energia_restante * pesos
        
        # La cantidad a añadir está limitada por la capacidad restante de cada hora
        energia_anadir = np.minimum(intento_reparto, cap_rest_candidatas)

        # 4. Actualizar la generación y la energía restante
        # Usamos la máscara booleana para actualizar solo las horas candidatas en el array original
        hid_ajustada[mascara_candidatas] += energia_anadir
        
        energia_restante -= energia_anadir.sum()

    return hid_ajustada

def suavizar_excedente_numpy(
    hidraulica_actual: np.ndarray, 
    energia_a_repartir: float, 
    hydro_min: float,
    potencia_max: float
) -> np.ndarray:
    """
    Versión robusta que reparte energía excedente usando un algoritmo de 
    "nivelación por agua" restringido, para suavizar la generación.

    Distribuye la 'energia_a_repartir' de forma iterativa, añadiendo energía
    a las horas con menor generación actual hasta agotarla, sin superar nunca
    la potencia máxima de cada hora.

    :param hidraulica_actual: Array NumPy con la generación hidráulica actual.
    :param energia_a_repartir: Exceso de energía EXACTO a repartir (MWh).
    :param hydro_min: Mínimo técnico de generación. Se usa como suelo.
    :param potencia_max: Potencia máxima por hora. Se usa como techo.
    :return: Array NumPy con la generación ajustada y suavizada.
    """
    hid_ajustada = hidraulica_actual.copy()
    energia_restante = energia_a_repartir

    # Asegurarnos de que la base de partida ya cumple el mínimo técnico
    hid_ajustada = np.maximum(hid_ajustada, hydro_min)
    
    # Iteramos para distribuir la energía progresivamente
    for _ in range(20): # Más iteraciones para un ajuste más fino
        if energia_restante < 1e-6:
            break

        # 1. Candidatas: horas que no han alcanzado la potencia máxima
        capacidad_restante = potencia_max - hid_ajustada
        mascara_candidatas = capacidad_restante > 1e-6
        
        if not mascara_candidatas.any():
            break

        # 2. Calcular cuánta energía repartir por hora candidata en esta iteración
        num_candidatas = mascara_candidatas.sum()
        energia_por_candidata = energia_restante / num_candidatas
        
        # 3. Identificar el "nivel del agua" actual de las candidatas
        # Es el nivel de la hora más llena entre las candidatas. No podemos 
        # añadir energía por encima de este nivel sin antes haber llenado
        # todas las horas hasta él.
        nivel_actual_candidatas = hid_ajustada[mascara_candidatas]
        
        # 4. Llenar hasta el siguiente nivel
        # La cantidad a añadir a cada hora es la mínima entre:
        # a) La energía disponible por candidata.
        # b) Lo que le falta para llegar a la siguiente hora más llena.
        # c) Su capacidad restante.
        
        # Para simplificar y hacerlo robusto, repartimos una pequeña fracción
        # de la energía por candidata, asegurando que el llenado es progresivo.
        intento_anadir = np.full(num_candidatas, energia_por_candidata)
        
        # Limitamos por la capacidad restante
        energia_anadir = np.minimum(intento_anadir, capacidad_restante[mascara_candidatas])

        # 5. Aplicar el ajuste y actualizar
        hid_ajustada[mascara_candidatas] += energia_anadir
        energia_restante -= energia_anadir.sum()
        
        # Opcional: Para una nivelación perfecta, se podría recalcular el nivel 
        # ideal en cada paso, pero este método iterativo es más estable.

    return hid_ajustada

def reducir_generacion_numpy(
    hidraulica_actual: np.ndarray, 
    energia_a_reducir: float, 
    hydro_min: float
) -> np.ndarray:
    """
    Versión NumPy para reducir la generación hidráulica de forma distribuida.
    Reduce la generación en las horas que superan el mínimo técnico.

    :param hidraulica_actual: Array NumPy con la generación actual.
    :param energia_a_reducir: Cantidad de energía a reducir (MWh).
    :param hydro_min: Mínimo técnico de generación para esa hora.
    :return: Array NumPy con la generación ajustada.
    """
    hid_ajustada = hidraulica_actual.copy()
    energia_restante_a_reducir = energia_a_reducir

    for _ in range(10):
        if energia_restante_a_reducir < 1e-6:
            break

        # Candidatas: horas cuya generación es superior al mínimo técnico
        capacidad_de_reduccion = hid_ajustada - hydro_min
        mascara_candidatas = capacidad_de_reduccion > 1e-6

        if not mascara_candidatas.any():
            break
        
        # Reparto uniforme entre todas las horas que pueden reducirse
        num_candidatas = mascara_candidatas.sum()
        reduccion_por_hora = energia_restante_a_reducir / num_candidatas
        
        # La cantidad a reducir está limitada por la capacidad de reducción de cada hora
        reduccion_real = np.minimum(reduccion_por_hora, capacidad_de_reduccion[mascara_candidatas])
        
        # Aplicamos la reducción
        hid_ajustada[mascara_candidatas] -= reduccion_real
        energia_restante_a_reducir -= reduccion_real.sum()
        
    return hid_ajustada

def procesar_cuenca(
    gap_slice: np.ndarray,
    level_np: np.ndarray,
    start_idx: int,
    end_idx: int,
    energia_mes: float,
    hydro_storage: float,
    f_cuenca: float,
    potencia_max: float,
    sensibility: float,
    max_capacity: float,
    level_overflow_pct: float,
    max_delta_mwh: float,
    hydro_min_for_level,
    puntos_optimizacion: int,
    reajustar_por_overload_numpy,
    suavizar_excedente_numpy,
    reducir_generacion_numpy,
    _objetivo_optimizador,
    minimize_scalar   
) -> tuple:
    """
    Procesa una cuenca hidráulica individual para evitar duplicación de código.
    
    Returns:
        tuple: (hid_mes, hydro_storage_actualizado, nivel_actualizado)
    """
    # ### PASO 1: OPTIMIZACIÓN IDEAL ###
    # # Calcular hydro_min para cada hora del mes
    # niveles_horarios = level_np[start_idx:end_idx]
    # hydro_min_array = np.array([hydro_min_for_level(nivel) * potencia_max for nivel in niveles_horarios])    
    
    nivel_prom = level_np[start_idx:end_idx].mean()
    hydro_min = hydro_min_for_level(nivel_prom) * potencia_max
    energia_obj = energia_mes + hydro_storage
    
    if puntos_optimizacion > 0:
        potencias = np.linspace(hydro_min, potencia_max, puntos_optimizacion)
        generaciones = [np.clip(gap_slice * f_cuenca, hydro_min, p).sum() for p in potencias]
        deltas_posibles = energia_mes - np.array(generaciones)
        error_obj = np.abs(deltas_posibles + hydro_storage)
        hid_max_optimo = potencias[np.argmin(error_obj)]
    else:
        res = minimize_scalar(_objetivo_optimizador, 
                             bounds=(hydro_min, potencia_max), 
                             args=(gap_slice, f_cuenca, hydro_min, energia_obj))
        hid_max_optimo = res.x

    hid_mes = np.clip(gap_slice * f_cuenca, hydro_min, hid_max_optimo)
    delta = energia_mes - hid_mes.sum()

    # ### PASO 2: APLICAR POLÍTICA DE OVERLOAD (AJUSTE CONDICIONAL) ###
    if nivel_prom >= level_overflow_pct and delta > 0:
        hid_mes = reajustar_por_overload_numpy(hid_mes, gap_slice, delta, potencia_max)
        # hid_mes = suavizar_excedente_numpy(hid_mes, delta, hydro_min, potencia_max)
        delta = energia_mes - hid_mes.sum()  # Recalcular delta tras el ajuste

    # ### PASO 3: FORZAR LÍMITE DE SUAVIZADO (RESTRICCIÓN DURA) ###
    if delta > max_delta_mwh:
        energia_a_anadir = delta - max_delta_mwh
        # hid_mes = reajustar_por_overload_numpy(hid_mes, gap_slice, energia_a_anadir, potencia_max)
        hid_mes = suavizar_excedente_numpy(hid_mes, energia_a_anadir, hydro_min, potencia_max)
    elif delta < -max_delta_mwh:
        energia_a_reducir = abs(delta) - max_delta_mwh
        hid_mes = reducir_generacion_numpy(hid_mes, energia_a_reducir, hydro_min) 
    
    # ### PASO 4: CÁLCULO FINAL Y ACTUALIZACIÓN DE ESTADO ###
    delta_final = energia_mes - hid_mes.sum()
    
    hydro_storage += delta_final
    pct = (delta_final / sensibility / max_capacity) * 100
    level_np[start_idx:] += pct
    
    # ### PASO 5: SEGURO ANTI-OVERFLOW (>100%) ###
    nivel_max = level_np[start_idx:].max()
    if nivel_max > 100:
        pct_exceso = nivel_max - 100
        energia_exceso = (pct_exceso / 100 * max_capacity) * sensibility
        hydro_storage -= energia_exceso
        level_np[start_idx:] -= pct_exceso
    
    return hid_mes, hydro_storage, level_np


def calcular_generacion_hidraulica(
    # --- Parámetros de Entrada ---
    df_sintetic: pd.DataFrame,
    energia_turbinada_mensual_internes: pd.Series,
    energia_turbinada_mensual_ebre: pd.Series,
    potencia_max_int: float,
    potencia_max_ebro: float,
    sensibility_int: float,
    sensibility_ebro: float,
    max_capacity_int: float,
    max_capacity_ebro: float,
    level_overflow_pct_int: float,
    level_overflow_pct_ebro: float,
    # --- Parámetros de Control ---
    max_salto_pct_mensual: float = 10.0,
    puntos_optimizacion: int = 0
) -> pd.DataFrame:
    """
    Calcula la generación hidráulica con un límite de variación mensual forzado y simétrico.

    --- DESCRIPCIÓN ---
    Esta función implementa una simulación hidrológica robusta y eficiente usando NumPy.
    Su característica principal es un mecanismo de suavizado que limita la variación mensual
    del nivel de los embalses, evitando oscilaciones bruscas.
    
    Permite elegir entre dos métodos de optimización:
    1. Búsqueda Manual (si puntos_optimizacion > 0): Método determinista y robusto.
    2. Minimize Scalar (si puntos_optimizacion <= 0): Método más rápido de SciPy.

    --- VARIABLES DE ENTRADA ---
    :param df_sintetic: DataFrame con datos horarios. Requiere columnas ['Hydro_Level_int', 'Hydro_Level_ebro', 'gap'].
    :param energia_turbinada_mensual_internes: Serie mensual con la energía a generar (Cuencas Internas).
    :param energia_turbinada_mensual_ebre: Serie mensual con la energía a generar (Cuenca Ebro).
    :param potencia_max_int: Potencia máxima instalada (MWh/h) para Internas.
    :param potencia_max_ebro: Potencia máxima instalada (MWh/h) para Ebro.
    :param sensibility_int: Factor de conversión MWh -> hm³ para Internas.
    :param sensibility_ebro: Factor de conversión MWh -> hm³ para Ebro.
    :param max_capacity_int: Capacidad máxima del embalse (hm³) para Internas.
    :param max_capacity_ebro: Capacidad máxima del embalse (hm³) para Ebro.
    :param hydro_min_for_level: Función que devuelve el % de generación mínima según el nivel del embalse.
    :param level_overflow_pct_int: Umbral de nivel (%) para activar el reajuste por overload (Internas).
    :param level_overflow_pct_ebro: Umbral de nivel (%) para activar el reajuste por overload (Ebro).
    :param reajustar_por_overload_numpy: Función NumPy para añadir generación y quemar excedente.
    :param reducir_generacion_numpy: Función NumPy para reducir generación y cubrir déficit.
    :param max_salto_pct_mensual: Máximo cambio porcentual permitido en el nivel del embalse por mes.
    :param puntos_optimizacion: Controla el método de optimización. Si > 0, usa búsqueda manual con N puntos.
                                Si <= 0, usa scipy.optimize.minimize_scalar para más velocidad.

    --- VARIABLES DE SALIDA ---
    :return: Un DataFrame con el mismo índice que el de entrada, pero con nuevas columnas que incluyen
             la generación hidráulica simulada ('hydro_int', 'hydro_ebro', 'Hidràulica') y los
             niveles de embalse actualizados ('Hydro_Level_int', 'Hydro_Level_ebro').
    """
    
    df_result = df_sintetic.copy()
    lookup_int = _crear_lookup_energia(energia_turbinada_mensual_internes)
    lookup_ebro = _crear_lookup_energia(energia_turbinada_mensual_ebre)
    
    # --- 1. PREPARACIÓN Y EXTRACCIÓN A NUMPY ---
    level_int_np = df_result['Hydro_Level_int'].to_numpy(dtype=np.float64)
    level_ebro_np = df_result['Hydro_Level_ebro'].to_numpy(dtype=np.float64)
    gap_np = df_result['gap'].to_numpy(dtype=np.float64)
    hydro_int_np, hydro_ebro_np = np.zeros_like(gap_np), np.zeros_like(gap_np)
    hydro_storage_int, hydro_storage_ebro = 0.0, 0.0

    max_delta_int_mwh = (max_salto_pct_mensual / 100) * max_capacity_int * sensibility_int
    max_delta_ebro_mwh = (max_salto_pct_mensual / 100) * max_capacity_ebro * sensibility_ebro

    meses_unicos = df_result.index.to_period('M').unique()
    boundaries = []
    for mes in meses_unicos:
        start_idx = df_result.index.searchsorted(mes.start_time, side='left')
        end_idx = df_result.index.searchsorted(mes.end_time, side='right')        
        # start_idx, end_idx = df_result.index.searchsorted([mes.start_time, mes.end_time], side='right')
        boundaries.append((mes.start_time, start_idx, end_idx))

    # --- 2. BUCLE PRINCIPAL SOBRE LOS MESES ---
    for month_start, start_idx, end_idx in boundaries:
        # ... (Lógica común para el mes sin cambios) ...
        key = (month_start.year, month_start.month)
        energia_mes_int = lookup_int.get(key, 0.0)
        energia_mes_ebro = lookup_ebro.get(key, 0.0)
        # mask_int = (energia_turbinada_mensual_internes.index.year == month_start.year) & \
        #            (energia_turbinada_mensual_internes.index.month == month_start.month)
        # energia_mes_int = energia_turbinada_mensual_internes[mask_int].values[0] if mask_int.any() else 0.0
        
        # mask_ebro = (energia_turbinada_mensual_ebre.index.year == month_start.year) & \
        #             (energia_turbinada_mensual_ebre.index.month == month_start.month)
        # energia_mes_ebro = energia_turbinada_mensual_ebre[mask_ebro].values[0] if mask_ebro.any() else 0.0
        
        total_energia_mes = energia_mes_int + energia_mes_ebro
        f_int = energia_mes_int / total_energia_mes if total_energia_mes > 0 else 0.0
        f_ebro = 1 - f_int
        gap_slice = gap_np[start_idx:end_idx]

        # --- PROCESAMIENTO CUENCA EBRO ---
        hid_ebro_mes, hydro_storage_ebro, level_ebro_np = procesar_cuenca(
            gap_slice=gap_slice,
            level_np=level_ebro_np,
            start_idx=start_idx,
            end_idx=end_idx,
            energia_mes=energia_mes_ebro,
            hydro_storage=hydro_storage_ebro,
            f_cuenca=f_ebro,
            potencia_max=potencia_max_ebro,
            sensibility=sensibility_ebro,
            max_capacity=max_capacity_ebro,
            level_overflow_pct=level_overflow_pct_ebro,
            max_delta_mwh=max_delta_ebro_mwh,
            hydro_min_for_level=hydro_min_for_level,
            puntos_optimizacion=puntos_optimizacion,
            reajustar_por_overload_numpy=reajustar_por_overload_numpy,
            suavizar_excedente_numpy=suavizar_excedente_numpy,
            reducir_generacion_numpy=reducir_generacion_numpy,
            _objetivo_optimizador=_objetivo_optimizador,
            minimize_scalar=minimize_scalar
        )
        hydro_ebro_np[start_idx:end_idx] = hid_ebro_mes

        # --- PROCESAMIENTO CUENCAS INTERNAS ---
        hid_int_mes, hydro_storage_int, level_int_np = procesar_cuenca(
            gap_slice=gap_slice,
            level_np=level_int_np,
            start_idx=start_idx,
            end_idx=end_idx,
            energia_mes=energia_mes_int,
            hydro_storage=hydro_storage_int,
            f_cuenca=f_int,
            potencia_max=potencia_max_int,
            sensibility=sensibility_int,
            max_capacity=max_capacity_int,
            level_overflow_pct=level_overflow_pct_int,
            max_delta_mwh=max_delta_int_mwh,
            hydro_min_for_level=hydro_min_for_level,
            puntos_optimizacion=puntos_optimizacion,
            reajustar_por_overload_numpy=reajustar_por_overload_numpy,
            suavizar_excedente_numpy=suavizar_excedente_numpy,
            reducir_generacion_numpy=reducir_generacion_numpy,
            _objetivo_optimizador=_objetivo_optimizador,
            minimize_scalar=minimize_scalar
        )
        hydro_int_np[start_idx:end_idx] = hid_int_mes

    # --- RECONSTRUCCIÓN FINAL ---
    df_result['hydro_int'] = hydro_int_np
    df_result['hydro_ebro'] = hydro_ebro_np
    df_result['Hidràulica'] = hydro_int_np + hydro_ebro_np
    df_result['Hydro_Level_int'] = level_int_np
    df_result['Hydro_Level_ebro'] = level_ebro_np
    
    return df_result

def _crear_lookup_energia(serie_mensual: pd.Series) -> dict:
    """
    Converteix una sèrie mensual en un diccionari {(year, month): valor}.
    Accés O(1) en lloc de O(n).
    """
    return {
        (idx.year, idx.month): valor 
        for idx, valor in serie_mensual.items()
    }

#%%

# =============================================================================
# EXEMPLE D'ÚS I BENCHMARK
# =============================================================================

if __name__ == "__main__":
    """
    Exemple d'ús del mòdul amb benchmark de rendiment.
    
    INSTRUCCIONS:
    1. Substitueix els stubs de les funcions auxiliars per les teves implementacions
    2. Carrega les teves dades reals
    3. Executa aquest script per mesurar el guany de rendiment
    """
    
    print("\n" + "=" * 70)
    print("BENCHMARK DE RENDIMENT: VERSIÓ ORIGINAL vs VERSIÓ OPTIMITZADA")
    print("=" * 70)
    
    # # -------------------------------------------------------------------------
    # # CARREGAR DADES (SUBSTITUIR PER LES TEVES DADES REALS)
    # # -------------------------------------------------------------------------
    # print("\n[!] NOTA: Aquest exemple usa dades simulades.")
    # print("    Substitueix aquesta secció per carregar les teves dades reals.\n")
    
    # # Crear dades de prova (SUBSTITUIR)
    # dates = pd.date_range('2016-01-01', '2024-12-31', freq='h')
    # n = len(dates)
    
    # # Dades simulades (SUBSTITUIR per les teves dades reals)
    # demanda = pd.Series(np.random.uniform(3000, 6000, n), index=dates, name='Demanda')
    # solar_h = pd.Series(np.random.uniform(0, 1000, n), index=dates)
    # eolica_h = pd.Series(np.random.uniform(0, 500, n), index=dates)
    # cogeneracion_h = pd.Series(np.random.uniform(200, 400, n), index=dates)
    # autoconsum_hourly = pd.Series(np.random.uniform(0, 100, n), index=dates)
    
    # # DataFrame de potència històrica simulat
    # dates_monthly = pd.date_range('2016-01-01', '2024-12-31', freq='MS')
    # potencia = pd.DataFrame({
    #     'Fotovoltaica': np.linspace(100, 500, len(dates_monthly)),
    #     'Termosolar': np.linspace(50, 50, len(dates_monthly)),
    #     'Eòlica': np.linspace(1000, 1500, len(dates_monthly)),
    #     'Cogeneració': np.linspace(800, 750, len(dates_monthly)),
    #     'Autoconsum': np.linspace(50, 300, len(dates_monthly))
    # }, index=dates_monthly)
    
    # # Nuclears
    # nuclears_base = pd.DataFrame({
    #     'Asco1': np.random.uniform(900, 1000, n),
    #     'Asco2': np.random.uniform(900, 1000, n),
    #     'Vandellos2': np.random.uniform(900, 1000, n)
    # }, index=dates)
    
    # # Embassaments
    # df_pct_int_h = pd.Series(np.random.uniform(0.3, 0.7, n), index=dates)
    # df_pct_ebre_h = pd.Series(np.random.uniform(0.4, 0.8, n), index=dates)
    
    # # Energia turbinada mensual
    # energia_turbinada_mensual_internes = pd.Series(
    #     np.random.uniform(20, 50, len(dates_monthly)), index=dates_monthly
    # )
    # energia_turbinada_mensual_ebre = pd.Series(
    #     np.random.uniform(100, 200, len(dates_monthly)), index=dates_monthly
    # )
    
    # Paràmetres físics
    potencia_max_hidraulica_ebro = 1374
    potencia_max_hidraulica_int = 163
    sensibility_ebro = 434
    sensibility_int = 323
    max_capacity_ebro = 2284
    max_capacity_int = 693
    
    # -------------------------------------------------------------------------
    # DEFINIR ESCENARIS DE PROVA
    # -------------------------------------------------------------------------
    escenarios_test = [
        {'solar': 1.0, 'wind': 1.0, 'dem': 1.0, 'cog': 1.0, 'auto': 1.0},
        {'solar': 1.5, 'wind': 1.2, 'dem': 1.0, 'cog': 0.9, 'auto': 1.5},
        {'solar': 2.0, 'wind': 1.5, 'dem': 1.0, 'cog': 0.8, 'auto': 2.0},
        {'solar': 2.5, 'wind': 2.0, 'dem': 0.95, 'cog': 0.7, 'auto': 2.5},
        {'solar': 3.0, 'wind': 2.5, 'dem': 0.9, 'cog': 0.6, 'auto': 3.0},
    ]
    
    n_escenarios = len(escenarios_test)
    
    # Configuració comuna
    base_config = {
        'df_demanda': demanda,
        'df_nucleares_base': nuclears_base,
        'df_cogeneracion': cogeneracion_h,
        'df_solar': solar_h,
        'df_eolica': eolica_h,
        'df_autoconsum': autoconsum_hourly,
        'df_potencia_historica': potencia,
        'df_capacidad_internes': df_pct_int_h,
        'df_capacidad_ebre': df_pct_ebre_h,
        'energia_turbinada_mensual_internes': energia_turbinada_mensual_internes,
        'energia_turbinada_mensual_ebre': energia_turbinada_mensual_ebre,
        'potencia_max_hidro': {'ebro': potencia_max_hidraulica_ebro, 'int': potencia_max_hidraulica_int},
        'sensibilidad_hidro': {'ebro': sensibility_ebro, 'int': sensibility_int},
        'capacidad_max_hidro': {'ebro': max_capacity_ebro, 'int': max_capacity_int},
        'nucleares_activas': [True, True, True],
        'baterias_config': [534, 5340, 0.8, 0],
        'max_salto_hidro_pct': 10.0,
    }
    
    # -------------------------------------------------------------------------
    # BENCHMARK 1: VERSIÓ ORIGINAL (SENSE PRECOMPUTACIÓ)
    # -------------------------------------------------------------------------
    print(f"\n[1/2] Executant {n_escenarios} escenaris SENSE precomputació...")
    
    t_start_original = time.time()
    
    for i, pesos in enumerate(escenarios_test):
        results, metrics = generar_escenario_sintetico(
            **base_config,
            pesos=pesos,
            precomputed=None  # Sense precomputació
        )
    
    t_original = time.time() - t_start_original
    t_per_escenari_original = t_original / n_escenarios
    
    print(f"      Temps total: {t_original:.2f} s")
    print(f"      Temps per escenari: {t_per_escenari_original:.3f} s")
    
    # -------------------------------------------------------------------------
    # BENCHMARK 2: VERSIÓ OPTIMITZADA (AMB PRECOMPUTACIÓ)
    # -------------------------------------------------------------------------
    print(f"\n[2/2] Executant {n_escenarios} escenaris AMB precomputació...")
    
    # Pas 1: Precomputar (una sola vegada)
    t_start_precompute = time.time()
    
    precomputed = precomputar_dades_base(
        df_demanda=demanda,
        df_solar=solar_h,
        df_eolica=eolica_h,
        df_cogeneracion=cogeneracion_h,
        df_autoconsum=autoconsum_hourly,
        df_potencia_historica=potencia,
        df_capacidad_internes=df_pct_int_h,
        df_capacidad_ebre=df_pct_ebre_h,
        potencia_cogeneracio_max=potencia.Cogeneració.iloc[-1],
        verbose=True
    )
    
    t_precompute = time.time() - t_start_precompute
    
    # Pas 2: Executar escenaris
    t_start_optimized = time.time()
    
    for i, pesos in enumerate(escenarios_test):
        results, metrics = generar_escenario_sintetico(
            **base_config,
            pesos=pesos,
            precomputed=precomputed  # Amb precomputació
        )
    
    t_optimized = time.time() - t_start_optimized
    t_per_escenari_optimized = t_optimized / n_escenarios
    t_total_optimized = t_precompute + t_optimized
    
    print(f"\n      Temps precomputació: {t_precompute:.2f} s (una sola vegada)")
    print(f"      Temps escenaris: {t_optimized:.2f} s")
    print(f"      Temps total: {t_total_optimized:.2f} s")
    print(f"      Temps per escenari: {t_per_escenari_optimized:.3f} s")
    
    # -------------------------------------------------------------------------
    # RESUM DE RESULTATS
    # -------------------------------------------------------------------------
    speedup = t_original / t_total_optimized if t_total_optimized > 0 else 0
    speedup_per_escenari = t_per_escenari_original / t_per_escenari_optimized if t_per_escenari_optimized > 0 else 0
    
    print("\n" + "=" * 70)
    print("RESUM DE RESULTATS")
    print("=" * 70)
    print(f"  Escenaris executats:        {n_escenarios}")
    print(f"  Temps original (total):     {t_original:.2f} s")
    print(f"  Temps optimitzat (total):   {t_total_optimized:.2f} s")
    print(f"  Speedup total:              {speedup:.2f}x")
    print("-" * 70)
    print(f"  Temps/escenari (original):  {t_per_escenari_original:.3f} s")
    print(f"  Temps/escenari (optimitzat):{t_per_escenari_optimized:.3f} s")
    print(f"  Speedup per escenari:       {speedup_per_escenari:.2f}x")
    print("=" * 70)
    
    # Projecció per a molts escenaris
    n_projectat = 100
    t_projectat_original = n_projectat * t_per_escenari_original
    t_projectat_optimized = t_precompute + (n_projectat * t_per_escenari_optimized)
    speedup_projectat = t_projectat_original / t_projectat_optimized if t_projectat_optimized > 0 else 0
    
    print(f"\n  PROJECCIÓ per a {n_projectat} escenaris:")
    print(f"    Original:   {t_projectat_original:.1f} s ({t_projectat_original/60:.1f} min)")
    print(f"    Optimitzat: {t_projectat_optimized:.1f} s ({t_projectat_optimized/60:.1f} min)")
    print(f"    Speedup:    {speedup_projectat:.2f}x")
    print("=" * 70)
    
    
#%%

# =============================================================================
# EXECUCIÓ D'UN SOL ESCENARI - VALIDACIÓ
# =============================================================================

# Assumeixo que ja tens carregades les dades:
# - demanda, nuclears_base, cogeneracion_h, solar_h, eolica_h
# - autoconsum_hourly, potencia (df_potencia_historica)
# - df_pct_int_h, df_pct_ebre_h
# - energia_turbinada_mensual_internes, energia_turbinada_mensual_ebre

# -----------------------------------------------------------------------------
# 1. DEFINIR L'ESCENARI A SIMULAR
# -----------------------------------------------------------------------------

# Paràmetres de configuració de l'escenari
mi_escenario = {
    'nucleares_activas': [True, True, True],  # Ascó I, Ascó II, Vandellòs II
    'pesos': {
        'solar': 0.9369,    # Factor d'escala solar (1.0 = potència actual)
        'wind': 0.9893,     # Factor d'escala eòlica
        'dem': 1.0,      # Factor d'escala demanda
        'cog': 1.0,      # Factor d'escala cogeneració
        'auto': 0.8607      # Factor d'escala autoconsum
    },
    'baterias_config': [534, 5340, 0.8, 0],  # [capacitat MWh, potència MW, eficiència, SoC inicial]
    'max_salto_hidro_pct': 10.0,
    'optimizador_hidro': 'rapido'  # 'rapido' o 'robusto'
}

# Paràmetres físics del sistema
params_fisics = {
    'potencia_max_hidro': {'ebro': 1374, 'int': 163},
    'sensibilidad_hidro': {'ebro': 434, 'int': 323},
    'capacidad_max_hidro': {'ebro': 2284, 'int': 693},
    'umbral_overflow_pct': {'ebro': 75.0, 'int': 75.0}
}

# Rang de dates
start_date = '2016-01-01'
end_date = '2024-12-31'

# -----------------------------------------------------------------------------
# 2. EXECUTAR L'ESCENARI
# -----------------------------------------------------------------------------

print("=" * 60)
print("EXECUCIÓ D'ESCENARI INDIVIDUAL")
print("=" * 60)

results, metrics = generar_escenario_sintetico(
    # Dades base
    df_demanda=demanda,
    df_nucleares_base=nuclears_base,
    df_cogeneracion=cogeneracion_h,
    df_solar=solar_h,
    df_eolica=eolica_h,
    df_autoconsum=autoconsum_hourly,
    df_potencia_historica=potencia,
    df_capacidad_internes=df_pct_int_h.squeeze(),
    df_capacidad_ebre=df_pct_ebre_h.squeeze(),
    energia_turbinada_mensual_internes=energia_turbinada_mensual_internes,
    energia_turbinada_mensual_ebre=energia_turbinada_mensual_ebre,
    
    # Paràmetres físics
    **params_fisics,
    
    # Configuració de l'escenari
    **mi_escenario,
    
    # Rang de dates
    start_date=start_date,
    end_date=end_date,
    
    # Sense precomputació (per validar amb el codi antic)
    precomputed=None
)

# -----------------------------------------------------------------------------
# 3. MOSTRAR MÈTRIQUES
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("MÈTRIQUES DE L'ESCENARI")
print("=" * 60)

# Agrupar mètriques per categoria
metriques_composicio = ['Nuclear %', 'Wind %', 'Solar %', 'Hydro %', 'Cogeneració', 'Cicles + Import.', 'Batteries %']
metriques_cobertura = ['Low-carbon %', 'Renewables %', 'Clean-coverage %', 'Ren.-coverage', 'Ren.cov-B']
metriques_gestio = ['Autoconsum', 'Surpluses']

print("\n📊 COMPOSICIÓ DEL MIX ENERGÈTIC:")
print("-" * 40)
for key in metriques_composicio:
    if key in metrics:
        print(f"  {key:<20} {metrics[key]:>8.2f} %")

print("\n🌱 INDICADORS DE COBERTURA:")
print("-" * 40)
for key in metriques_cobertura:
    if key in metrics:
        print(f"  {key:<20} {metrics[key]:>8.2f} %")

print("\n⚡ GESTIÓ ENERGÈTICA:")
print("-" * 40)
for key in metriques_gestio:
    if key in metrics:
        print(f"  {key:<20} {metrics[key]:>8.2f} %")

# -----------------------------------------------------------------------------
# 4. RESUM RÀPID (per comparar amb codi antic)
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("RESUM PER VALIDACIÓ (copiar i comparar)")
print("=" * 60)
for key, value in sorted(metrics.items()):
    print(f"  '{key}': {value},")

# -----------------------------------------------------------------------------
# 5. VERIFICAR DATAFRAME DE RESULTATS
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("DATAFRAME DE RESULTATS")
print("=" * 60)
print(f"  Forma: {results.shape}")
print(f"  Període: {results.index.min()} → {results.index.max()}")
print(f"  Columnes: {list(results.columns)}")

# Mostrar estadístiques bàsiques
print("\n📈 Estadístiques per columna (mitjana):")
print("-" * 40)
cols_interessants = ['Demanda', 'Nuclear', 'Solar_w', 'Eòlica_w', 'Cogen_w', 'Hidràulica', 'Bateries', 'Gas+Imports']
for col in cols_interessants:
    if col in results.columns:
        print(f"  {col:<15} {results[col].mean():>10.2f} MW")
        
#%%
from numba import jit

@jit(nopython=True)
def battery_numba(power_max, capacity_max, efficiency, soc_inicial, gap):
    n = len(gap)
    soc = np.empty(n)
    power = np.empty(n)
    soc_actual = soc_inicial
    
    for i in range(n):
        p_ideal = gap[i]
        if p_ideal < 0:  # Càrrega
            p_carrega = min(-p_ideal, power_max, (capacity_max - soc_actual) / efficiency)
            power[i] = -p_carrega
            soc_actual += p_carrega * efficiency
        else:  # Descàrrega
            p_descarrega = min(p_ideal, power_max, soc_actual)
            power[i] = p_descarrega
            soc_actual -= p_descarrega
        soc[i] = soc_actual
    
    return soc, power
       
#%%

%%timeit
"""
Mòdul de Generació d'Escenaris Sintètics
========================================

Funció principal per simular escenaris energètics del sistema català.

Autor: Víctor García Carrasco
Data: 2024
"""

import pandas as pd
import numpy as np


def generar_escenario_sintetico(
    # --- DADES D'ENTRADA (DataFrames i Series) ---
    df_demanda: pd.Series,
    df_nucleares_base: pd.DataFrame,
    df_cogeneracion: pd.Series,
    df_solar: pd.Series,
    df_eolica: pd.Series,
    df_autoconsum: pd.Series,
    df_potencia_historica: pd.DataFrame,
    df_capacidad_internes: pd.DataFrame,
    df_capacidad_ebre: pd.DataFrame,
    energia_turbinada_mensual_internes: pd.Series,
    energia_turbinada_mensual_ebre: pd.Series,
        
    # --- PARÀMETRES DE CONFIGURACIÓ DE L'ESCENARI ---
    nucleares_activas: list = [True, True, True],
    pesos: dict = {'solar': 1, 'wind': 1, 'dem': 1, 'cog': 1, 'auto': 1},
    baterias_config: list = [500, 2000, 0.8, 0],
    max_salto_hidro_pct: float = 5.0,
    optimizador_hidro: str = 'rapido',
       
    # --- PARÀMETRES FÍSICS DEL MODEL ---
    potencia_max_hidro: dict = None,
    sensibilidad_hidro: dict = None,
    capacidad_max_hidro: dict = None,
    umbral_overflow_pct: dict = {'ebro': 75.0, 'int': 75.0},
    
    # --- RANG DE DATES ---
    start_date: str = '2016-01-01',
    end_date: str = '2024-12-31'
    
) -> tuple[pd.DataFrame, dict]:
    """
    Genera un escenari energètic sintètic complet a partir d'un conjunt de paràmetres.

    Aquesta funció encapsula tot el procés:
    1. Construcció del DataFrame base segons la configuració nuclear i de pesos.
    2. Simulació de la generació hidràulica amb lògica de suavitzat.
    3. Simulació de l'emmagatzematge amb bateries/bombeig.
    4. Càlcul de la generació tèrmica residual (Gas+Imports).
    5. Càlcul d'un conjunt de mètriques clau de l'escenari.

    :return: Tupla (df_escenari, metriques)
    """

    # =========================================================================
    # ETAPA 1: CONFIGURACIÓ NUCLEAR
    # =========================================================================
    nombres_nucleares = ['Asco1', 'Asco2', 'Vandellos2']
    nucleares_a_usar = [nombre for nombre, activo in zip(nombres_nucleares, nucleares_activas) if activo]
    
    if nucleares_a_usar:
        df_nuclear_total = df_nucleares_base[nucleares_a_usar].sum(axis=1).resample('h').ffill()
    else:
        df_nuclear_total = pd.Series(0, index=df_demanda.index).resample('h').ffill()

    # =========================================================================
    # ETAPA 2: CONSTRUCCIÓ I REESCALAT DEL DATAFRAME INICIAL
    # =========================================================================
    df_sintetic = pd.concat([
        df_demanda[start_date:end_date],
        df_nuclear_total[start_date:end_date]
    ], axis=1)
    df_sintetic.columns = ['Demanda', 'Nuclear']

    # Reescalat de renovables i cogeneració
    # NOTA: Usem .loc[:end_date].iloc[-1] per obtenir el valor més recent <= end_date
    potencia_solar_total = df_potencia_historica[['Fotovoltaica', 'Termosolar']].sum(axis=1)
    potencia_solar_ref = potencia_solar_total.loc[:end_date].iloc[-1]
    solar_reescalada = df_solar * (
        potencia_solar_ref / potencia_solar_total
    ).resample('h').ffill()
    
    potencia_eolica_ref = df_potencia_historica['Eòlica'].loc[:end_date].iloc[-1]
    eolica_reescalada = df_eolica * (
        potencia_eolica_ref / df_potencia_historica['Eòlica']
    ).resample('h').ffill()
    
    potencia_cogen_ref = df_potencia_historica['Cogeneració'].loc[:end_date].iloc[-1]
    cogen_reescalada = df_cogeneracion * (
        potencia_cogen_ref / df_potencia_historica['Cogeneració']
    ).resample('h').ffill()
    
    # Correcció de tendència de cogeneració
    tendencia = cogen_reescalada.rolling(window=365 * 24, center=True, min_periods=1).mean()
    mask_ref = (cogen_reescalada.index.year >= 2024)
    nivel_ref = tendencia.loc[mask_ref].mean()
    factor = pd.Series(1.0, index=cogen_reescalada.index)
    factor.loc[~mask_ref] = nivel_ref / tendencia.loc[~mask_ref]
    cogen_reescalada = cogen_reescalada * factor
    cogen_reescalada = cogen_reescalada.clip(upper=potencia_cogen_ref)

    df_sintetic['Solar_w'] = pesos['solar'] * solar_reescalada
    df_sintetic['Eòlica_w'] = pesos['wind'] * eolica_reescalada
    df_sintetic['Cogen_w'] = pesos['cog'] * cogen_reescalada
    
    df_sintetic = df_sintetic.dropna()

    # Recàlcul de la demanda considerant autoconsum
    df_sintetic.Demanda, autoconsum_historic = extraer_autoconsumo(
        df_sintetic.Demanda,
        df_sintetic.Solar_w, 
        df_autoconsum[start_date:end_date], 
        pr=0.75
    )
    df_sintetic.Demanda = pesos['dem'] * df_sintetic.Demanda
    potencia_autoconsum_ref = df_potencia_historica['Autoconsum'].loc[:end_date].iloc[-1]
    df_sintetic.Demanda, autoconsum_estimat = insertar_autoconsumo(
        df_sintetic.Demanda,
        df_sintetic.Solar_w, 
        pesos['auto'] * potencia_autoconsum_ref, 
        pr=0.75
    )
    
    # Nivells hidràulics
    df_sintetic['Hydro_Level_int'] = 100 * df_capacidad_internes.squeeze().reindex(
        df_sintetic.index, method='ffill'
    )
    df_sintetic['Hydro_Level_ebro'] = 100 * df_capacidad_ebre.squeeze().reindex(
        df_sintetic.index, method='ffill'
    )

    # =========================================================================
    # ETAPA 3: CÀLCUL DEL GAP INICIAL
    # =========================================================================
    df_sintetic['gap'] = (
        pesos['dem'] * df_sintetic['Demanda'] - 
        df_sintetic['Nuclear'] - 
        df_sintetic['Solar_w'] - 
        df_sintetic['Eòlica_w'] - 
        df_sintetic['Cogen_w']
    )
    
    df_sintetic = df_sintetic.dropna()
    
    # =========================================================================
    # ETAPA 4: SIMULACIÓ DE LA GENERACIÓ HIDRÀULICA
    # =========================================================================
    puntos_opt = 300 if optimizador_hidro == 'robusto' else 0
    
    df_sintetic = calcular_generacion_hidraulica(
        df_sintetic=df_sintetic,
        energia_turbinada_mensual_internes=energia_turbinada_mensual_internes,
        energia_turbinada_mensual_ebre=energia_turbinada_mensual_ebre,
        potencia_max_int=potencia_max_hidro['int'],
        potencia_max_ebro=potencia_max_hidro['ebro'],
        sensibility_int=sensibilidad_hidro['int'],
        sensibility_ebro=sensibilidad_hidro['ebro'],
        max_capacity_int=capacidad_max_hidro['int'],
        max_capacity_ebro=capacidad_max_hidro['ebro'],
        level_overflow_pct_int=umbral_overflow_pct['int'],
        level_overflow_pct_ebro=umbral_overflow_pct['ebro'],
        max_salto_pct_mensual=max_salto_hidro_pct,
        puntos_optimizacion=puntos_opt
    )
    df_sintetic.rename(columns={'Hidràulica': 'Hidráulica'}, inplace=True)

    df_sintetic['hydro_int'] = np.clip(df_sintetic['hydro_int'], 0, potencia_max_hidro['int'])
    df_sintetic['hydro_ebro'] = np.clip(df_sintetic['hydro_ebro'], 0, potencia_max_hidro['ebro'])
    df_sintetic['Hidráulica'] = df_sintetic['hydro_int'] + df_sintetic['hydro_ebro']

    # =========================================================================
    # ETAPA 5: SIMULACIÓ DE BATERIES/BOMBEIG
    # =========================================================================
    # Gap residual sense fonts no renovables (per mètriques)
    df_sintetic['gap0'] = (
        pesos['dem'] * df_sintetic['Demanda'] - 
        df_sintetic['Solar_w'] - 
        df_sintetic['Eòlica_w'] - 
        df_sintetic['Hidráulica']
    )
    
    # Gap residual després de hidràulica
    df_sintetic['gap'] = (
        pesos['dem'] * df_sintetic['Demanda'] - 
        df_sintetic['Nuclear'] - 
        df_sintetic['Solar_w'] - 
        df_sintetic['Eòlica_w'] - 
        df_sintetic['Cogen_w'] - 
        df_sintetic['Hidráulica']
    )
      
    # capacity, power = battery(baterias_config, df_sintetic['gap'])
    capacity, power = battery_numba(mi_escenario['baterias_config'][0], mi_escenario['baterias_config'][1],mi_escenario['baterias_config'][2],mi_escenario['baterias_config'][3], df_sintetic['gap'].values)
    df_sintetic['gap'] = df_sintetic['gap'] - power
    df_sintetic['gap0'] = df_sintetic['gap0'] - power
    
    df_sintetic['Bateries'] = pd.Series(power, index=df_sintetic.index)
    df_sintetic.loc[df_sintetic['Bateries'] < 0, 'Bateries'] = 0
    df_sintetic['Càrrega'] = pd.Series(-power, index=df_sintetic.index)
    df_sintetic.loc[df_sintetic['Càrrega'] < 0, 'Càrrega'] = 0
    df_sintetic['Càrrega'] = df_sintetic['Càrrega'] + df_sintetic['Demanda'] * pesos['dem']
    df_sintetic['SOC'] = pd.Series(capacity, index=df_sintetic.index)
    
    # =========================================================================
    # ETAPA 6: CÀLCUL DE LA GENERACIÓ RESIDUAL (TÈRMICA + IMPORTS)
    # =========================================================================
    df_sintetic['Gas+Imports'] = (
        df_sintetic['Demanda'] * pesos['dem'] -
        (df_sintetic['Nuclear'] + df_sintetic['Solar_w'] + 
         df_sintetic['Eòlica_w'] + df_sintetic['Cogen_w'] + 
         df_sintetic['Hidráulica'] + df_sintetic['Bateries'])
    )
    df_sintetic['Gas+Imports'] = df_sintetic['Gas+Imports'].clip(lower=0)
    
    # =========================================================================
    # ETAPA 7: CÀLCUL DE MÈTRIQUES FINALS
    # =========================================================================
    sample = df_sintetic.rename(columns={
        'Cogen_w': 'Cogeneració', 'Eòlica_w': 'Eòlica', 'Solar_w': 'Solar'
    })[['Demanda', 'Nuclear', 'Cogeneració', 'Eòlica', 'Solar', 'Hidráulica', 
        'Bateries', 'Gas+Imports', 'Càrrega', 'gap0']]
    
    total_demand = sample['Demanda'].sum() * pesos['dem']
    
    if total_demand < 1e-6:
        return df_sintetic, {}

    sample.loc[:, 'Total'] = (
        sample['Gas+Imports'] + sample['Cogeneració'] + sample['Nuclear'] + 
        sample['Solar'] + sample['Eòlica'] + sample['Hidráulica']
    )

    w_dem = pesos['dem']
    
    metrics = {
        'Wind %': sample['Eòlica'].sum() * 100 / total_demand,
        'Solar %': sample['Solar'].sum() * 100 / total_demand,
        'Autoconsum': autoconsum_estimat.sum() * 100 / total_demand,        
        'Hydro %': sample['Hidráulica'].sum() * 100 / total_demand,
        'Nuclear %': sample['Nuclear'].sum() * 100 / total_demand,
        'Cogeneració': sample['Cogeneració'].sum() * 100 / total_demand,
        'Cicles + Import.': sample['Gas+Imports'].sum() * 100 / total_demand,        
        'Batteries %': sample['Bateries'].sum() * 100 / total_demand,
        'Fossil+Imports %': (sample['Gas+Imports'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Low-carbon %': sample[['Eòlica', 'Solar', 'Hidráulica', 'Nuclear']].sum().sum() * 100 / total_demand,
        'Renewables %': sample[['Eòlica', 'Solar', 'Hidráulica']].sum().sum() * 100 / total_demand,
        'Ren.-coverage': 100 - (sample['Gas+Imports'] + sample['Nuclear'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Ren.cov-B': round((1 - sum(sample.gap0[sample.gap0 > 0]) / sum(w_dem * sample.Demanda)) * 100, 1),
        'Clean-coverage %': 100 - (sample['Gas+Imports'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Surpluses': (sample['Total'] + sample['Bateries'] - sample['Càrrega']).sum() * 100 / total_demand
    }

    df_sintetic.rename(columns={'Hidráulica': 'Hidràulica'}, inplace=True)

    return df_sintetic, {k: round(v, 2) for k, v in metrics.items()}


# =============================================================================
# EXEMPLE D'ÚS
# =============================================================================

# Paràmetres de configuració de l'escenari
mi_escenario = {
    'nucleares_activas': [True, True, True],  # Ascó I, Ascó II, Vandellòs II
    'pesos': {
        'solar': 0.9369,    # Factor d'escala solar (1.0 = potència actual)
        'wind': 0.9893,     # Factor d'escala eòlica
        'dem': 1.0,      # Factor d'escala demanda
        'cog': 1.0,      # Factor d'escala cogeneració
        'auto': 0.8607      # Factor d'escala autoconsum
    },
    'baterias_config': [534, 5340, 0.8, 0],  # [capacitat MWh, potència MW, eficiència, SoC inicial]
    'max_salto_hidro_pct': 10.0,
    'optimizador_hidro': 'rapido'  # 'rapido' o 'robusto'
}

# Paràmetres físics del sistema
params_fisics = {
    'potencia_max_hidro': {'ebro': 1374, 'int': 163},
    'sensibilidad_hidro': {'ebro': 434, 'int': 323},
    'capacidad_max_hidro': {'ebro': 2284, 'int': 693},
    'umbral_overflow_pct': {'ebro': 75.0, 'int': 75.0}
}

# Rang de dates
start_date = '2016-01-01'
end_date = '2024-12-31'


# Executar
results, metrics = generar_escenario_sintetico(
    df_demanda=demanda,
    df_nucleares_base=nuclears_base,
    df_cogeneracion=cogeneracion_h,
    df_solar=solar_h,
    df_eolica=eolica_h,
    df_autoconsum=autoconsum_hourly,
    df_potencia_historica=potencia,
    df_capacidad_internes=df_pct_int_h,
    df_capacidad_ebre=df_pct_ebre_h,
    energia_turbinada_mensual_internes=energia_turbinada_mensual_internes,
    energia_turbinada_mensual_ebre=energia_turbinada_mensual_ebre,
    **params_fisics,
    **mi_escenario,
    start_date='2016-01-01',
    end_date='2024-12-31'
)



#%%
%%time
"""
Mòdul de Generació d'Escenaris Sintètics - Versió Optimitzada
=============================================================

Optimitzacions implementades:
- Factor de correcció de cogeneració precalculat (evita rolling cada execució)
- Compatible amb Numba per a la funció battery

Autor: Víctor García Carrasco
Data: 2024
"""

import pandas as pd
import numpy as np


def precalcular_factor_cogeneracio(
    df_cogeneracion: pd.Series,
    df_potencia_historica: pd.DataFrame,
    start_date: str = '2016-01-01',
    end_date: str = '2024-12-31',
    any_referencia: int = 2024
) -> pd.Series:
    """
    Precalcula el factor de correcció de tendència per a la cogeneració.
    
    Aquesta funció s'executa UNA SOLA VEGADA i el resultat es passa a
    generar_escenario_sintetico() mitjançant el paràmetre 'factor_cogen_precalculat'.
    
    El càlcul inclou:
    1. Reescalat a potència vigent
    2. Càlcul de tendència (rolling 365*24 hores)
    3. Normalització respecte a l'any de referència
    
    :param df_cogeneracion: Sèrie horària de cogeneració original
    :param df_potencia_historica: DataFrame amb evolució de potència instal·lada
    :param start_date: Data d'inici del període
    :param end_date: Data de fi del període
    :param any_referencia: Any de referència per la normalització (default: 2024)
    
    :return: Serie amb el factor multiplicatiu per cada hora
    """
    # print("Precalculant factor de correcció de cogeneració...")
    
    # Reescalat a potència vigent
    potencia_cogen_ref = df_potencia_historica['Cogeneració'].loc[:end_date].iloc[-1]
    ratio_cogen = (potencia_cogen_ref / df_potencia_historica['Cogeneració']).resample('h').ffill()
    cogen_reescalada = df_cogeneracion * ratio_cogen
    
    # Tendència suau (mitjana mòbil centrada) - OPERACIÓ COSTOSA
    tendencia = cogen_reescalada.rolling(
        window=365 * 24, 
        center=True, 
        min_periods=1
    ).mean()
    
    # Màscara de referència
    mask_ref = (cogen_reescalada.index.year >= any_referencia)
    
    # Nivell de referència
    nivel_ref = tendencia.loc[mask_ref].mean()
    
    # Factor variable
    factor = pd.Series(1.0, index=cogen_reescalada.index)
    factor.loc[~mask_ref] = nivel_ref / tendencia.loc[~mask_ref]
    
    # # Retornar només el rang d'interès
    # print(f"  Factor calculat per al període {start_date} a {end_date}")
    # print(f"  Any de referència: {any_referencia}")
    
    return factor[start_date:end_date]

def generar_escenario_sintetico(
    # --- DADES D'ENTRADA (DataFrames i Series) ---
    df_demanda: pd.Series,
    df_nucleares_base: pd.DataFrame,
    df_cogeneracion: pd.Series,
    df_solar: pd.Series,
    df_eolica: pd.Series,
    df_autoconsum: pd.Series,
    df_potencia_historica: pd.DataFrame,
    df_capacidad_internes: pd.DataFrame,
    df_capacidad_ebre: pd.DataFrame,
    energia_turbinada_mensual_internes: pd.Series,
    energia_turbinada_mensual_ebre: pd.Series,
        
    # --- PARÀMETRES DE CONFIGURACIÓ DE L'ESCENARI ---
    nucleares_activas: list = [True, True, True],
    pesos: dict = {'solar': 1, 'wind': 1, 'dem': 1, 'cog': 1, 'auto': 1},
    baterias_config: list = [500, 2000, 0.8, 0],
    max_salto_hidro_pct: float = 5.0,
    optimizador_hidro: str = 'rapido',
       
    # --- PARÀMETRES FÍSICS DEL MODEL ---
    potencia_max_hidro: dict = None,
    sensibilidad_hidro: dict = None,
    capacidad_max_hidro: dict = None,
    umbral_overflow_pct: dict = {'ebro': 75.0, 'int': 75.0},
    
    # --- RANG DE DATES ---
    start_date: str = '2016-01-01',
    end_date: str = '2024-12-31',
    
    # --- FACTOR PRECALCULAT (OPCIONAL) ---
    factor_cogen_precalculat: pd.Series = None
    
) -> tuple[pd.DataFrame, dict]:
    """
    Genera un escenari energètic sintètic complet.
    
    OPTIMITZACIÓ: Si es passa 'factor_cogen_precalculat' (resultat de
    precalcular_factor_cogeneracio()), s'evita el càlcul del rolling
    de 365*24 finestres a cada execució.

    :param factor_cogen_precalculat: Factor de correcció precalculat (opcional).
                                     Si és None, es calcula dins la funció.
    :return: Tupla (df_escenari, metriques)
    """

    # =========================================================================
    # ETAPA 1: CONFIGURACIÓ NUCLEAR
    # =========================================================================
    nombres_nucleares = ['Asco1', 'Asco2', 'Vandellos2']
    nucleares_a_usar = [nombre for nombre, activo in zip(nombres_nucleares, nucleares_activas) if activo]
    
    if nucleares_a_usar:
        df_nuclear_total = df_nucleares_base[nucleares_a_usar].sum(axis=1).resample('h').ffill()
    else:
        df_nuclear_total = pd.Series(0, index=df_demanda.index).resample('h').ffill()

    # =========================================================================
    # ETAPA 2: CONSTRUCCIÓ I REESCALAT DEL DATAFRAME INICIAL
    # =========================================================================
    df_sintetic = pd.concat([
        df_demanda[start_date:end_date],
        df_nuclear_total[start_date:end_date]
    ], axis=1)
    df_sintetic.columns = ['Demanda', 'Nuclear']

    # --- Solar ---
    potencia_solar_total = df_potencia_historica[['Fotovoltaica', 'Termosolar']].sum(axis=1)
    potencia_solar_ref = potencia_solar_total.loc[:end_date].iloc[-1]
    solar_reescalada = df_solar * (
        potencia_solar_ref / potencia_solar_total
    ).resample('h').ffill()
    
    # --- Eòlica ---
    potencia_eolica_ref = df_potencia_historica['Eòlica'].loc[:end_date].iloc[-1]
    eolica_reescalada = df_eolica * (
        potencia_eolica_ref / df_potencia_historica['Eòlica']
    ).resample('h').ffill()
    
    # --- Cogeneració (amb o sense factor precalculat) ---
    potencia_cogen_ref = df_potencia_historica['Cogeneració'].loc[:end_date].iloc[-1]
    ratio_cogen = (potencia_cogen_ref / df_potencia_historica['Cogeneració']).resample('h').ffill()
    cogen_reescalada = df_cogeneracion * ratio_cogen
    
    if factor_cogen_precalculat is not None:
        # BRANCA OPTIMITZADA: usar factor precalculat
        cogen_reescalada = cogen_reescalada[start_date:end_date] * factor_cogen_precalculat
    else:
        # BRANCA ORIGINAL: calcular factor (costós)
        tendencia = cogen_reescalada.rolling(window=365 * 24, center=True, min_periods=1).mean()
        mask_ref = (cogen_reescalada.index.year >= 2024)
        nivel_ref = tendencia.loc[mask_ref].mean()
        factor = pd.Series(1.0, index=cogen_reescalada.index)
        factor.loc[~mask_ref] = nivel_ref / tendencia.loc[~mask_ref]
        cogen_reescalada = cogen_reescalada * factor
    
    cogen_reescalada = cogen_reescalada.clip(upper=potencia_cogen_ref)

    df_sintetic['Solar_w'] = pesos['solar'] * solar_reescalada
    df_sintetic['Eòlica_w'] = pesos['wind'] * eolica_reescalada
    df_sintetic['Cogen_w'] = pesos['cog'] * cogen_reescalada
    
    df_sintetic = df_sintetic.dropna()

    # --- Autoconsum ---
    df_sintetic.Demanda, autoconsum_historic = extraer_autoconsumo(
        df_sintetic.Demanda,
        df_sintetic.Solar_w, 
        df_autoconsum[start_date:end_date], 
        pr=0.75
    )
    df_sintetic.Demanda = pesos['dem'] * df_sintetic.Demanda
    potencia_autoconsum_ref = df_potencia_historica['Autoconsum'].loc[:end_date].iloc[-1]
    df_sintetic.Demanda, autoconsum_estimat = insertar_autoconsumo(
        df_sintetic.Demanda,
        df_sintetic.Solar_w, 
        pesos['auto'] * potencia_autoconsum_ref, 
        pr=0.75
    )
    
    # --- Nivells hidràulics ---
    df_sintetic['Hydro_Level_int'] = 100 * df_capacidad_internes.squeeze().reindex(
        df_sintetic.index, method='ffill'
    )
    df_sintetic['Hydro_Level_ebro'] = 100 * df_capacidad_ebre.squeeze().reindex(
        df_sintetic.index, method='ffill'
    )

    # =========================================================================
    # ETAPA 3: CÀLCUL DEL GAP INICIAL
    # =========================================================================
    df_sintetic['gap'] = (
        pesos['dem'] * df_sintetic['Demanda'] - 
        df_sintetic['Nuclear'] - 
        df_sintetic['Solar_w'] - 
        df_sintetic['Eòlica_w'] - 
        df_sintetic['Cogen_w']
    )
    
    df_sintetic = df_sintetic.dropna()
    
    # =========================================================================
    # ETAPA 4: SIMULACIÓ DE LA GENERACIÓ HIDRÀULICA
    # =========================================================================
    puntos_opt = 300 if optimizador_hidro == 'robusto' else 0
    
    df_sintetic = calcular_generacion_hidraulica(
        df_sintetic=df_sintetic,
        energia_turbinada_mensual_internes=energia_turbinada_mensual_internes,
        energia_turbinada_mensual_ebre=energia_turbinada_mensual_ebre,
        potencia_max_int=potencia_max_hidro['int'],
        potencia_max_ebro=potencia_max_hidro['ebro'],
        sensibility_int=sensibilidad_hidro['int'],
        sensibility_ebro=sensibilidad_hidro['ebro'],
        max_capacity_int=capacidad_max_hidro['int'],
        max_capacity_ebro=capacidad_max_hidro['ebro'],
        level_overflow_pct_int=umbral_overflow_pct['int'],
        level_overflow_pct_ebro=umbral_overflow_pct['ebro'],
        max_salto_pct_mensual=max_salto_hidro_pct,
        puntos_optimizacion=puntos_opt
    )
    df_sintetic.rename(columns={'Hidràulica': 'Hidráulica'}, inplace=True)

    df_sintetic['hydro_int'] = np.clip(df_sintetic['hydro_int'], 0, potencia_max_hidro['int'])
    df_sintetic['hydro_ebro'] = np.clip(df_sintetic['hydro_ebro'], 0, potencia_max_hidro['ebro'])
    df_sintetic['Hidráulica'] = df_sintetic['hydro_int'] + df_sintetic['hydro_ebro']

    # =========================================================================
    # ETAPA 5: SIMULACIÓ DE BATERIES/BOMBEIG
    # =========================================================================
    df_sintetic['gap0'] = (
        pesos['dem'] * df_sintetic['Demanda'] - 
        df_sintetic['Solar_w'] - 
        df_sintetic['Eòlica_w'] - 
        df_sintetic['Hidráulica']
    )
    
    df_sintetic['gap'] = (
        pesos['dem'] * df_sintetic['Demanda'] - 
        df_sintetic['Nuclear'] - 
        df_sintetic['Solar_w'] - 
        df_sintetic['Eòlica_w'] - 
        df_sintetic['Cogen_w'] - 
        df_sintetic['Hidráulica']
    )
    
    # NOTA: Si uses battery_numba, passa .values per convertir a numpy array
    capacity, power = battery_numba(mi_escenario['baterias_config'][1], mi_escenario['baterias_config'][0],mi_escenario['baterias_config'][2],mi_escenario['baterias_config'][3], df_sintetic['gap'].values)
    # capacity, power = battery(baterias_config, df_sintetic['gap'])
    
    df_sintetic['gap'] = df_sintetic['gap'] - power
    df_sintetic['gap0'] = df_sintetic['gap0'] - power
    
    df_sintetic['Bateries'] = pd.Series(power, index=df_sintetic.index)
    df_sintetic.loc[df_sintetic['Bateries'] < 0, 'Bateries'] = 0
    df_sintetic['Càrrega'] = pd.Series(-power, index=df_sintetic.index)
    df_sintetic.loc[df_sintetic['Càrrega'] < 0, 'Càrrega'] = 0
    df_sintetic['Càrrega'] = df_sintetic['Càrrega'] + df_sintetic['Demanda'] * pesos['dem']
    df_sintetic['SOC'] = pd.Series(capacity, index=df_sintetic.index)
    
    # =========================================================================
    # ETAPA 6: CÀLCUL DE LA GENERACIÓ RESIDUAL
    # =========================================================================
    df_sintetic['Gas+Imports'] = (
        df_sintetic['Demanda'] * pesos['dem'] -
        (df_sintetic['Nuclear'] + df_sintetic['Solar_w'] + 
         df_sintetic['Eòlica_w'] + df_sintetic['Cogen_w'] + 
         df_sintetic['Hidráulica'] + df_sintetic['Bateries'])
    )
    df_sintetic['Gas+Imports'] = df_sintetic['Gas+Imports'].clip(lower=0)
    
    # =========================================================================
    # ETAPA 7: CÀLCUL DE MÈTRIQUES FINALS
    # =========================================================================
    sample = df_sintetic.rename(columns={
        'Cogen_w': 'Cogeneració', 'Eòlica_w': 'Eòlica', 'Solar_w': 'Solar'
    })[['Demanda', 'Nuclear', 'Cogeneració', 'Eòlica', 'Solar', 'Hidráulica', 
        'Bateries', 'Gas+Imports', 'Càrrega', 'gap0']]
    
    total_demand = sample['Demanda'].sum() * pesos['dem']
    
    if total_demand < 1e-6:
        return df_sintetic, {}

    sample.loc[:, 'Total'] = (
        sample['Gas+Imports'] + sample['Cogeneració'] + sample['Nuclear'] + 
        sample['Solar'] + sample['Eòlica'] + sample['Hidráulica']
    )

    w_dem = pesos['dem']
    
    metrics = {
        'Wind %': sample['Eòlica'].sum() * 100 / total_demand,
        'Solar %': sample['Solar'].sum() * 100 / total_demand,
        'Autoconsum': autoconsum_estimat.sum() * 100 / total_demand,        
        'Hydro %': sample['Hidráulica'].sum() * 100 / total_demand,
        'Nuclear %': sample['Nuclear'].sum() * 100 / total_demand,
        'Cogeneració': sample['Cogeneració'].sum() * 100 / total_demand,
        'Cicles + Import.': sample['Gas+Imports'].sum() * 100 / total_demand,        
        'Batteries %': sample['Bateries'].sum() * 100 / total_demand,
        'Fossil+Imports %': (sample['Gas+Imports'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Low-carbon %': sample[['Eòlica', 'Solar', 'Hidráulica', 'Nuclear']].sum().sum() * 100 / total_demand,
        'Renewables %': sample[['Eòlica', 'Solar', 'Hidráulica']].sum().sum() * 100 / total_demand,
        'Ren.-coverage': 100 - (sample['Gas+Imports'] + sample['Nuclear'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Ren.cov-B': round((1 - sum(sample.gap0[sample.gap0 > 0]) / sum(w_dem * sample.Demanda)) * 100, 1),
        'Clean-coverage %': 100 - (sample['Gas+Imports'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Surpluses': (sample['Total'] + sample['Bateries'] - sample['Càrrega']).sum() * 100 / total_demand
    }

    df_sintetic.rename(columns={'Hidráulica': 'Hidràulica'}, inplace=True)

    return df_sintetic, {k: round(v, 2) for k, v in metrics.items()}

#%%
%%timeit
# =============================================================================
# EXEMPLE D'ÚS AMB FACTOR PRECALCULAT
# =============================================================================
# Paràmetres de configuració de l'escenari
mi_escenario = {
    'nucleares_activas': [True, True, True],  # Ascó I, Ascó II, Vandellòs II
    'pesos': {
        'solar': 0.9369,    # Factor d'escala solar (1.0 = potència actual)
        'wind': 0.9893,     # Factor d'escala eòlica
        'dem': 1.0,      # Factor d'escala demanda
        'cog': 1.0,      # Factor d'escala cogeneració
        'auto': 0.8607      # Factor d'escala autoconsum
    },
    'baterias_config': [534, 5340, 0.8, 0],  # [capacitat MWh, potència MW, eficiència, SoC inicial]
    'max_salto_hidro_pct': 10.0,
    'optimizador_hidro': 'rapido'  # 'rapido' o 'robusto'
}

# Paràmetres físics del sistema
params_fisics = {
    'potencia_max_hidro': {'ebro': 1374, 'int': 163},
    'sensibilidad_hidro': {'ebro': 434, 'int': 323},
    'capacidad_max_hidro': {'ebro': 2284, 'int': 693},
    'umbral_overflow_pct': {'ebro': 75.0, 'int': 75.0}
}

# Rang de dates
start_date = '2016-01-01'
end_date = '2024-12-31'


# ============================================
# PAS 1: Precalcular factor (UNA SOLA VEGADA)
# ============================================
factor_cogen = precalcular_factor_cogeneracio(
    df_cogeneracion=cogeneracion_h,
    df_potencia_historica=potencia,
    start_date='2016-01-01',
    end_date='2024-12-31',
    any_referencia=2024
)

# ============================================
# PAS 2: Executar escenaris (moltes vegades)
# ============================================

results, metrics = generar_escenario_sintetico(
    df_demanda=demanda,
    df_nucleares_base=nuclears_base,
    df_cogeneracion=cogeneracion_h,
    df_solar=solar_h,
    df_eolica=eolica_h,
    df_autoconsum=autoconsum_hourly,
    df_potencia_historica=potencia,
    df_capacidad_internes=df_pct_int_h,
    df_capacidad_ebre=df_pct_ebre_h,
    energia_turbinada_mensual_internes=energia_turbinada_mensual_internes,
    energia_turbinada_mensual_ebre=energia_turbinada_mensual_ebre,
    **params_fisics,
    **mi_escenario,
    start_date='2016-01-01',
    end_date='2024-12-31',
    factor_cogen_precalculat=factor_cogen  # <- CLAU: passar el factor precalculat
)


# # Mostrar mètriques
# for k, v in metrics.items():
#     print(f"{k}: {v}")
    

# results.Hydro_Level_int.plot()
# results.Hydro_Level_ebro.plot()
