# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:42:10 2026

@author: tirki
"""

#%%
import pandas as pd
import numpy as np
from typing import Optional
import time
from load_data import cargar_datos_simulador, hydro_min_for_level

from EnerSimFunc import (extraer_autoconsumo,
                         insertar_autoconsumo,
                         calcular_generacion_hidraulica,
                         battery_numba, remove_restrictions_seasonal,
                         simulate_full_water_management,
                         # increments_a_llindars
                         )

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

import sys
import os

# Afegir el directori actual al path per als workers de joblib
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

import pickle
#%% - Carrega de dades [26.5 s]
%%time
datos = cargar_datos_simulador(verbose=True)

#%% - Funció del simulador
def precalcular_boundaries(df: pd.DataFrame) -> list:
    """Pre-calcula límits mensuals."""
    mesos_unics = df.index.to_period('M').unique()
    boundaries = []
    for mes in mesos_unics:
        start_idx = df.index.searchsorted(mes.start_time, side='left')
        end_idx = df.index.searchsorted(mes.end_time, side='right')
        boundaries.append((mes.start_time, start_idx, end_idx))
    return boundaries


def factor_capacitat_eolica(
    multiple: float, 
    cf_actual: float = 0.24, 
    cf_modern: float = 0.34, # 0.30
    p: float = 2.0, 
    c: float = 1.0 #1/2  # Perquè T(2) = 0.75
) -> float:
    """
    Calcula el factor de capacitat mitjà del parc eòlic amb funció sigmoide.
    
    multiple=1: Capacitat actual (100%)
    multiple=2: Doble capacitat → factor ≈ 28.5%
    """
    if multiple <= 1:
        return cf_actual
    else:
        T = (multiple - 1)**p / ((multiple - 1)**p + c)
        return cf_actual + (cf_modern - cf_actual) * T

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
    df_dessalacio_historica: pd.Series,
    df_regeneracio_estimada: pd.Series,
    potencia_cogeneracio_max: float,
    # energia_turbinada_mensual_internes: pd.Series,
    # energia_turbinada_mensual_ebre: pd.Series,    
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
    # 5. ESTALVI DE LES INTERVENCIONS A LES CONQUES INTERNES
    # -------------------------------------------------------------------------    
    if verbose:
        print("  [5/5] Descomposició de la sèrie històrica...")
        
    # 1 Estima el ahorro debido a las restricciones históricas        
    ahorro, nivel_sr = remove_restrictions_seasonal(100 * df_capacidad_internes.squeeze()[start_date:end_date])
    # ahorro, nivel_sr = remove_restrictions_seasonal(hydro_level_int.resample('D').last())
    # ahorro, nivel_sr = remove_restrictions_seasonal(100*df_capacidad_internes.squeeze()[start_date:end_])

    ahorro = ahorro.resample('D').last()
    ahorro = ahorro.resample('h').interpolate(method='linear')

    # 2 Añade la desalación histórica
    dessalacio_acum_diaria = df_dessalacio_historica[start_date:end_date].cumsum()
    # reindexado horario + interpolación lineal
    dessalacio_acum_horaria = (
        dessalacio_acum_diaria
        .resample('h')
        .interpolate(method='linear')
    )
    # sumo al ahorro de las restricciones
    ahorro += dessalacio_acum_horaria
    
    # 3 Añade la regeneración histórica
    regeneracio_acum_diaria = df_regeneracio_estimada[start_date:end_date].cumsum()
    # reindexado horario + interpolación lineal
    regeneracio_acum_horaria = (
        regeneracio_acum_diaria
        .resample('h')
        .interpolate(method='linear')
    )
    ahorro += regeneracio_acum_horaria
    
    # =========================================================================
    # 6. PRE-CÀLCUL PER AJUST DE CF EÒLIC (escalat gamma)
    # =========================================================================
    if verbose:
        print("  [6/6] Precalculant l'ajust del CF eòlic...")
    
    # Potència eòlica de referència (última disponible)
    potencia_eolica_ref = df_potencia_historica['Eòlica'].iloc[-1]
    
    # Sèrie normalitzada (corregida per evolució històrica de potència)
    eolica_normalitzada = eolica_base * (
        potencia_eolica_ref / df_potencia_historica['Eòlica']
    ).resample('h').ffill().reindex(eolica_base.index, method='ffill')
    
    # Ratio normalitzat [0, 1]
    eolica_ratio = (eolica_normalitzada / potencia_eolica_ref).clip(0, 1)
    
    # Lookup table: gamma → CF
    gammas = np.linspace(0.5, 1.5, 21)
    CFs = np.array([(eolica_ratio ** g).mean() for g in gammas])
    
    
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
        'estalvi': ahorro,
        
        # Metadades
        'start_date': start_date,
        'end_date': end_date,
        'pr_autoconsum': pr_autoconsum,
        
        # Per ajust CF eòlic
        'potencia_eolica_ref': potencia_eolica_ref,
        'eolica_ratio': eolica_ratio,
        'gamma_lookup': (gammas, CFs),        
        
        # # Boundaries
        # 'boundaries': precalcular_boundaries(pd.DataFrame(index=demanda_neta.index)),
        # 'lookup_int': {(idx.year, idx.month): val for idx, val in energia_turbinada_mensual_internes.items()},
        # 'lookup_ebro': {(idx.year, idx.month): val for idx, val in energia_turbinada_mensual_ebre.items()},
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
    # hydro_min_for_level,
        
    # --- PARÀMETRES DE CONFIGURACIÓ DE L'ESCENARI ---
    nucleares_activas: list = [True, True, True],
    pesos: dict = {'solar': 1, 'wind': 1, 'dem': 1, 'cog': 1, 'auto': 1},
    baterias_config: list = [500, 2000, 0.8, 0],
    max_salto_hidro_pct: float = 5.0,
    optimizador_hidro: str = 'rapido',
    CF_eolica_obj: float = None,
       
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
    # BRANCA OPTIMITZADA: Usar dades precomputades
    # -------------------------------------------------------------------------
    # Extreure dades precomputades
    solar_base = precomputed['solar_base']
    eolica_base = precomputed['eolica_base']
    cogen_base = precomputed['cogen_base']
    demanda_neta = precomputed['demanda_neta']
    autoconsum_historic = precomputed['autoconsum_historic']
    hydro_level_int = precomputed['hydro_level_int']
    hydro_level_ebro = precomputed['hydro_level_ebro']
    pr_autoconsum = precomputed.get('pr_autoconsum', 0.75)
    
    potencia_eolica_ref = precomputed.get('potencia_eolica_ref', None)
    eolica_ratio = precomputed.get('eolica_ratio', None)
    gamma_lookup = precomputed.get('gamma_lookup', None)    
    
    # Construir DataFrame aplicant només els pesos
    df_sintetic = pd.DataFrame({
        'Demanda': demanda_neta.copy(),
        'Nuclear': df_nuclear_total[start_date:end_date]
    })
    
    # Ajust CF eòlic si s'especifica objectiu
    if CF_eolica_obj is not None and gamma_lookup is not None and eolica_ratio is not None:
        gammas, CFs = gamma_lookup
        # Interpolar gamma per obtenir CF objectiu (CFs decreixent amb gamma)
        gamma = np.interp(CF_eolica_obj, CFs[::-1], gammas[::-1])
        eolica_ajustada = potencia_eolica_ref * (eolica_ratio ** gamma)
        df_sintetic['Eòlica_w'] = pesos['wind'] * eolica_ajustada.reindex(df_sintetic.index)
    else:
        df_sintetic['Eòlica_w'] = pesos['wind'] * eolica_base
    
    df_sintetic['Solar_w'] = pesos['solar'] * solar_base
    # df_sintetic['Eòlica_w'] = pesos['wind'] * eolica_base
    df_sintetic['Cogen_w'] = pesos['cog'] * cogen_base
    
    df_sintetic = df_sintetic.dropna()
    
    # Aplicar pes a demanda i reinsertar autoconsum amb nova capacitat
    df_sintetic['Demanda'] = pesos['dem'] * df_sintetic['Demanda']
    potencia_autoconsum_ref = df_potencia_historica['Autoconsum'].iloc[-1]
    df_sintetic['Demanda'], autoconsum_estimat = insertar_autoconsumo(
        df_sintetic['Demanda'],
        df_sintetic['Solar_w'], 
        pesos['auto'] * potencia_autoconsum_ref, 
        pr=pr_autoconsum
    )
    
    # Afegir nivells hidràulics (ja precomputats)
    df_sintetic['Hydro_Level_int'] = hydro_level_int.reindex(df_sintetic.index)
    df_sintetic['Hydro_Level_ebro'] = hydro_level_ebro.reindex(df_sintetic.index)
    

    # =========================================================================
    # ETAPA 3: CÀLCUL DEL GAP INICIAL
    # =========================================================================
    df_sintetic['gap'] = (
        df_sintetic['Demanda'] - 
        df_sintetic['Nuclear'] - 
        df_sintetic['Solar_w'] - 
        df_sintetic['Eòlica_w'] - 
        df_sintetic['Cogen_w']
    )
    
    df_sintetic = df_sintetic.dropna()
    
    # =========================================================================
    # ETAPA 4: SIMULACIÓ DE LA GENERACIÓ HIDRÀULICA
    # =========================================================================
    puntos_opt = 20 if optimizador_hidro == 'robusto' else 0
    
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
        # hydro_min_for_level=hydro_min_for_level,
        hydro_min_lookup=hydro_min_lookup,
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
        df_sintetic['Demanda'] - 
        df_sintetic['Solar_w'] - 
        df_sintetic['Eòlica_w'] - 
        df_sintetic['Hidráulica']
    )
    
    # Gap residual després de hidràulica
    df_sintetic['gap'] = (
        df_sintetic['Demanda'] - 
        df_sintetic['Nuclear'] - 
        df_sintetic['Solar_w'] - 
        df_sintetic['Eòlica_w'] - 
        df_sintetic['Cogen_w'] - 
        df_sintetic['Hidráulica']
    )
      
    # capacity, power = battery(baterias_config, df_sintetic['gap'])
    capacity, power = battery_numba(baterias_config[0], baterias_config[1],baterias_config[2],baterias_config[3], df_sintetic['gap'].values)    
    df_sintetic['gap'] = df_sintetic['gap'] - power
    df_sintetic['gap0'] = df_sintetic['gap0'] - power
    
    df_sintetic['Bateries'] = pd.Series(power, index=df_sintetic.index)
    df_sintetic.loc[df_sintetic['Bateries'] < 0, 'Bateries'] = 0
    df_sintetic['Càrrega'] = pd.Series(power * (-1), index=df_sintetic.index)
    df_sintetic.loc[df_sintetic['Càrrega'] < 0, 'Càrrega'] = 0
    df_sintetic['Càrrega'] = df_sintetic['Càrrega'] + df_sintetic['Demanda']
    df_sintetic['SOC'] = pd.Series(capacity, index=df_sintetic.index)
    
    # =========================================================================
    # ETAPA 6: CÀLCUL DE LA GENERACIÓ RESIDUAL (TÈRMICA + IMPORTS)
    # =========================================================================
    df_sintetic['Gas+Imports'] = (
        df_sintetic['Demanda'] -
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
    
    total_demand = sample['Demanda'].sum()
    
    if total_demand < 1e-6:
        return df_sintetic, {}

    sample.loc[:, 'Total'] = (
        sample['Gas+Imports'] + sample['Cogeneració'] + sample['Nuclear'] + 
        sample['Solar'] + sample['Eòlica'] + sample['Hidráulica']
    )

    
    metrics = {
        'Wind %': sample['Eòlica'].sum() * 100 / total_demand,
        'Solar %': sample['Solar'].sum() * 100 / total_demand,
        'Autoconsum %': autoconsum_estimat.sum() * 100 / total_demand,        
        'Hydro %': sample['Hidráulica'].sum() * 100 / total_demand,
        'Nuclear %': sample['Nuclear'].sum() * 100 / total_demand,
        'Cogeneració %': sample['Cogeneració'].sum() * 100 / total_demand,
        'Cicles + Import. %': sample['Gas+Imports'].sum() * 100 / total_demand,        
        'Batteries %': sample['Bateries'].sum() * 100 / total_demand,
        'Fossil+Imports %': (sample['Gas+Imports'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Low-carbon %': (sample[['Eòlica', 'Solar', 'Hidráulica', 'Nuclear']].sum().sum()) * 100 / total_demand,
        'Renewables %': sample[['Eòlica', 'Solar', 'Hidráulica']].sum().sum() * 100 / total_demand,
        'Ren.-coverage': 100 - (sample['Gas+Imports'] + sample['Nuclear'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Ren.cov-B': round((1 - sum(sample.gap0[sample.gap0 > 0]) / sum(sample.Demanda)) * 100, 1),
        'Clean-coverage %': 100 - (sample['Gas+Imports'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Surpluses': ((sample['Total'] + sample['Bateries'] - sample['Càrrega']).sum()) * 100 / total_demand
    }

    years = len(sample) / 8760
    
    metrics_mwh = {
        'Wind (GWh/y)': sample['Eòlica'].sum() / years / 1000,
        'Solar (GWh/y)': sample['Solar'].sum() / years / 1000,
        'Autoconsum (GWh/y)': autoconsum_estimat.sum() / years / 1000,
        'Hydro (GWh/y)': sample['Hidráulica'].sum() / years / 1000,
        'Nuclear (GWh/y)': sample['Nuclear'].sum() / years / 1000,
        'Cogeneració (GWh/y)': sample['Cogeneració'].sum() / years / 1000,
        'Cicles + Import. (GWh/y)': sample['Gas+Imports'].sum() / years / 1000,
        'Batteries (GWh/y)': sample['Bateries'].sum() / years / 1000,
        'Fossil+Imports (GWh/y)': (sample['Gas+Imports'] + sample['Cogeneració']).sum() / years / 1000,
        'Low-carbon (GWh/y)': sample[['Eòlica', 'Solar', 'Hidráulica', 'Nuclear']].sum().sum() / years / 1000,
        'Renewables (GWh/y)': sample[['Eòlica', 'Solar', 'Hidráulica']].sum().sum() / years / 1000,
        'Clean-coverage (GWh/y)': (total_demand - (sample['Gas+Imports'] + sample['Cogeneració']).sum()) / years / 1000,
        'Surpluses (GWh/y)': (sample['Total'] + sample['Bateries'] - sample['Càrrega']).sum() / years / 1000
    }
    
    
    total_generation = (sample['Total'] + autoconsum_estimat).sum()
    metrics0 = {
        'Wind %': sample['Eòlica'].sum() * 100 / total_generation,
        'Solar %': sample['Solar'].sum() * 100 / total_generation,
        'Autoconsum %': autoconsum_estimat.sum() * 100 / total_generation,        
        'Hydro %': sample['Hidráulica'].sum() * 100 / total_generation,
        'Nuclear %': sample['Nuclear'].sum() * 100 / total_generation,
        'Cogeneració %': sample['Cogeneració'].sum() * 100 / total_generation,
        'Cicles + Import. %': sample['Gas+Imports'].sum() * 100 / total_generation,
        'Renewables %': sample[['Eòlica', 'Solar', 'Hidráulica']].sum().sum() * 100 / total_generation,
        'Net-Zero %': sample[['Eòlica', 'Solar', 'Hidráulica', 'Nuclear']].sum().sum() * 100 / total_generation,
    }


    df_sintetic.rename(columns={'Hidráulica': 'Hidràulica'}, inplace=True)

    # return df_sintetic, {k: round(v, 2) for k, v in metrics.items()}

    # return (
    #     df_sintetic,
    #     {k: round(v, 2) for k, v in metrics.items()},
    #     {k: round(v, 2) for k, v in metrics_mwh.items()},
    #     {k: round(v, 2) for k, v in metrics0.items()}
    # )

    return (
        df_sintetic,
        {k: float(round(v, 2)) for k, v in metrics.items()},
        {k: float(round(v, 2)) for k, v in metrics_mwh.items()},
        {k: float(round(v, 2)) for k, v in metrics0.items()}
    )

#%%
%%time

# Paràmetres físics
potencia_max_hidraulica_ebro = 1374
potencia_max_hidraulica_int = 163
sensibility_ebro = 434
sensibility_int = 323
max_capacity_ebro = 2284
max_capacity_int = 693

precomputed = precomputar_dades_base(
    df_demanda=datos.demanda,
    df_solar=datos.solar_h,
    df_eolica=datos.eolica_h,
    df_cogeneracion=datos.cogeneracion_h,
    df_autoconsum=datos.autoconsum_hourly,
    df_potencia_historica=datos.potencia,
    df_capacidad_internes=datos.df_pct_int_h,
    df_capacidad_ebre=datos.df_pct_ebre_h,
    df_dessalacio_historica=datos.dessalacio_diaria,
    df_regeneracio_estimada=datos.regeneracio_diaria,
    potencia_cogeneracio_max=datos.potencia.Cogeneració.iloc[-1],
    # energia_turbinada_mensual_internes=datos.energia_turbinada_mensual_internes,
    # energia_turbinada_mensual_ebre=datos.energia_turbinada_mensual_ebre,     
    verbose=True
)

# CREAR LOOKUP TABLE (1 sola vegada, abans de Parallel)
nivells = np.arange(0, 100.1, 0.1)  # Resolució 0.1%
hydro_min_lookup = np.array([hydro_min_for_level(n) for n in nivells])

# Afegir a precomputed
precomputed['hydro_min_lookup'] = hydro_min_lookup

consumo_base_diario_estacional = np.array([
    1.37, 1.37, 1.42, 1.74, 2.32, 3.03, 
    3.51, 3.24, 2.36, 1.48, 1.37, 1.37
])

#%%

hydro_base_level = ((datos.df_pct_int_h.squeeze() - precomputed['estalvi']/max_capacity_int)*100).dropna()
min_run_hours = 6
max_desalation = 32
midpoint_estimation = 75 # parámetro del sigmoide de desalación
overflow_threshold_pct = 90
seasonal_phase_months = 0.0 # Maximo en enero
seasonal_desal_amplitude = 0.0 # Sin variación estacional
desal_sensibility = 3500


seasonal_capacity = seasonal_decompose(
    100*datos.df_pct_int_h['2016-01-01':'2024-12-31'].squeeze().resample('ME').last(), 
    model='additive', 
    period=12
).seasonal
monthly_seasonal = seasonal_capacity.groupby(seasonal_capacity.index.month).mean()
seasonal_amplitude = monthly_seasonal.max() - monthly_seasonal.min()

historic_hydro_metrics = {
    'Llenado mínimo (%)': round(100*datos.df_pct_int_h.squeeze().min(), 1),
    'Llenado promedio (%)': round(100*datos.df_pct_int_h.squeeze().mean(), 1),
    'Variación estacional (%)': round(seasonal_amplitude, 1) if not np.isnan(seasonal_amplitude) else None,
    'Desalación FC (%)': int(100 * datos.dessalacio_diaria.sum()*desal_sensibility / (max_desalation*24*len(datos.dessalacio_diaria))),
    'Regeneración total (hm³)': round(datos.regeneracio_diaria.sum(), 1),
    'Desalación total (hm³)': round(datos.dessalacio_diaria.sum(), 1),
    'Restricciones históricas (días)': int(np.ceil((datos.df_pct_int_h.squeeze()[datos.df_pct_int_h.squeeze() * 100 < 40].count()) / 24)),    
    'Restriccions históricas (hm³)': round(precomputed['estalvi'].iloc[-1], 1)
}




def procesar_escenario(
    # =========================================================================
    # SÈRIES DE DADES D'ENTRADA
    # =========================================================================
    df_demanda: pd.Series,
    df_nuclear: pd.DataFrame,
    df_cogeneracion: pd.Series,
    df_solar: pd.Series,
    df_eolica: pd.Series,
    df_autoconsum: pd.Series,
    df_potencia: pd.DataFrame,
    df_niveles_int: pd.DataFrame,
    df_niveles_ebro: pd.DataFrame,
    df_energia_turbinada_mensual_internes: pd.Series,
    df_energia_turbinada_mensual_ebre: pd.Series,
    df_nivel_si: pd.Series,  # Nivells sense intervencions (contrafàctic)
    
    # =========================================================================
    # PARÀMETRES FÍSICS DEL SISTEMA (fixes per escenari)
    # =========================================================================
    max_capacity_int: float,
    max_capacity_ebro: float,
    potencia_max_int: float,
    potencia_max_ebro: float,
    sensibility_int: float,
    sensibility_ebro: float,
    # hydro_min_for_level: pd.DataFrame,  # AFEGIT: Requerit per generar_escenario
    
    # =========================================================================
    # PARÀMETRES DE CONSUM HÍDRIC
    # =========================================================================
    consumo_base_diario_estacional_hm3: np.ndarray = None,  # Array de 12 valors
    umbrales_sequia: dict = None,
    restricciones_sectoriales: dict=None,
    
    # =========================================================================
    # DADES PRECOMPUTADES (OPTIMITZACIÓ)
    # =========================================================================
    precomputed: dict = None,  # AFEGIT: Resultat de precomputar_dades_base()
    
    # =========================================================================
    # VARIABLES DE DECISIÓ - SISTEMA ELÈCTRIC
    # =========================================================================
    nucleares_activas: list = [True, True, True],
    potencia_solar: float = None,
    potencia_eolica: float = None,
    potencia_cogeneracion: float = None,
    potencia_autoconsumo: float = None,
    potencia_baterias: float = 534,
    duracion_horas: float = 4,
    demanda_electrica: float = 1,
    CF_eolica_obj: float = None,        # Opció A: Valor fix
    usar_CF_automatic: bool = True,      # Opció B: Calcular amb funció sigmoide
    
    # =========================================================================
    # VARIABLES DE DECISIÓ - GESTIÓ HÍDRICA
    # =========================================================================
    overflow_threshold_pct: float = 90,

    min_run_hours: int = 6,
    max_desalation: float = 32,
    midpoint_estimation: float = 75,
    llindar_activacio_desal_max: int = 2,
    seasonal_phase_months: float = 0.0,
    seasonal_desal_amplitude: float = 0.0,
    trend_time_window: float = 24*30,
    # desal_derivada_sensibilitat: float = 2.0, #típic (1-5)
    # desal_derivada_amplitud_up: float = 0.0, #1.5, #típic (0.5 1.5)
    # desal_derivada_amplitud_down: float = 0.8, #1.5, #típic (0.5 1.5)
    k_deriv: float = 0,
    max_regen: float = 0.173,
    regen_base_pct: float = 0.5,
    llindar_activacio_regen_max: int = 2,
    
    # =========================================================================
    # PARÀMETRES DE CONVERSIÓ
    # =========================================================================
    save_hm3_per_mwh: float = 1/3500,  # AFEGIT: Per conversió MW → hm³
    
) -> dict:
    """
    Executa la simulació completa d'un escenari energètic-hídric integrat.
    
    Procés:
        1. Calcula pesos relatius de les potències
        2. Genera escenari elèctric sintètic (generar_escenario_sintetico)
        3. Simula gestió hídrica (simulate_full_water_management)
        4. Acobla resultats: modifica hidràulica i tèrmica segons turbinació extra
        5. Calcula mètriques energètiques i hídriques
    
    Args:
        [veure paràmetres amb comentaris]
    
    Returns:
        dict amb claus:
            - 'energy_data': DataFrame amb totes les sèries horàries
            - 'energy_metrics': Mètriques del sistema elèctric
            - 'hydro_metrics': Mètriques del sistema hídric
            - 'level_final': Evolució del nivell dels embassaments
            - 'desal_final': Dessalinització [MW]
            - 'desal_final_hm3': Dessalinització [hm³]
            - 'regen_final': Regeneració [hm³]
            - 'savings_final': Estalvi per restriccions [hm³]
            - 'costes': Desglossament de costos d'inversió
    """
    if umbrales_sequia is None:
        umbrales_sequia = {
        'Emergencia_3': 5.5, 
        'Emergencia_2': 11.0, 
        'Emergencia': 16.0,
        'Excepcionalitat': 25.0, 
        'Alerta': 40.0, 
        'Prealerta': 60.0
    }
    
    
    # =========================================================================
    # 1. CONFIGURACIÓ DE POTÈNCIES I PESOS
    # =========================================================================
    max_desalation_ref = 32
    
    # Potències actuals de referència
    potencia_solar_ref = df_potencia.Fotovoltaica.iloc[-1] + df_potencia.Termosolar.iloc[-1]
    potencia_eolica_ref = df_potencia.Eòlica.iloc[-1]
    potencia_cogen_ref = df_potencia.Cogeneració.iloc[-1]
    potencia_auto_ref = df_potencia.Autoconsum.iloc[-1]
    
    # Usar valors actuals si no s'especifiquen
    if potencia_solar is None:
        potencia_solar = potencia_solar_ref
    if potencia_eolica is None:
        potencia_eolica = potencia_eolica_ref
    if potencia_cogeneracion is None:
        potencia_cogeneracion = potencia_cogen_ref
    if potencia_autoconsumo is None:
        potencia_autoconsumo = potencia_auto_ref        
    
    # Calcular pesos relatius
    peso_solar = potencia_solar / potencia_solar_ref
    peso_wind = potencia_eolica / potencia_eolica_ref
    peso_cogen = potencia_cogeneracion / potencia_cogen_ref
    peso_auto = potencia_autoconsumo / potencia_auto_ref
    peso_dem = demanda_electrica
       
    # NOU: Determinar CF eòlic objectiu
    if CF_eolica_obj is not None:
        # Opció A: Valor fix especificat
        cf_obj_final = CF_eolica_obj
    elif usar_CF_automatic:
        # Opció B: Calcular amb sigmoide segons múltiple de capacitat
        cf_obj_final = factor_capacitat_eolica(multiple=peso_wind)
    else:
        # Sense ajust
        cf_obj_final = None    
    
    # Configuració de bateries [capacitat, energia, eficiència, SOC_inicial]
    bat_config = [potencia_baterias, potencia_baterias * duracion_horas, 0.8, 0]
    
    # =========================================================================
    # 2. GENERACIÓ DE L'ESCENARI ELÈCTRIC
    # =========================================================================
    
    results, energy_metrics_pct, energy_metrics_MWh, energy_metrics_pct2 = generar_escenario_sintetico(
        # Dades d'entrada
        df_demanda=df_demanda,
        df_nucleares_base=df_nuclear,
        df_cogeneracion=df_cogeneracion,
        df_solar=df_solar,
        df_eolica=df_eolica,
        CF_eolica_obj=cf_obj_final,
        df_autoconsum=df_autoconsum,
        df_potencia_historica=df_potencia,
        df_capacidad_internes=df_niveles_int,
        df_capacidad_ebre=df_niveles_ebro,
        energia_turbinada_mensual_internes=df_energia_turbinada_mensual_internes,
        energia_turbinada_mensual_ebre=df_energia_turbinada_mensual_ebre,
        # hydro_min_for_level=hydro_min_for_level,
        
        # Configuració de l'escenari
        nucleares_activas=nucleares_activas,
        pesos={'solar': peso_solar, 'wind': peso_wind, 'dem': peso_dem, 'cog': peso_cogen, 'auto': peso_auto},
        baterias_config=bat_config,
        max_salto_hidro_pct=5.0,
        optimizador_hidro='robusto',
        
        # Paràmetres físics
        potencia_max_hidro={'ebro': potencia_max_ebro, 'int': potencia_max_int},
        sensibilidad_hidro={'ebro': sensibility_ebro, 'int': sensibility_int},
        capacidad_max_hidro={'ebro': max_capacity_ebro, 'int': max_capacity_int},
        # umbral_overflow_pct={'ebro': 75.0, 'int': 75.0},
        umbral_overflow_pct={'ebro': 75.0, 'int': 75},
        
        # Optimització
        precomputed=precomputed
    )
    
    # =========================================================================
    # 3. PREPARACIÓ D'INPUTS PER LA GESTIÓ HÍDRICA
    # =========================================================================
    
    # Calcular excedents (gap negatiu → excedent positiu)
    surpluses = results.gap.copy()
    surpluses = surpluses.clip(upper=0) * (-1)
    
    # =========================================================================
    # 4. SIMULACIÓ DE GESTIÓ HÍDRICA
    # =========================================================================
    
    (level_final, regen_final, desal_final, savings_final, savings_agro_final, savings_urba_final, 
     extra_hydro_final, surpluses_net, spillage_final) = simulate_full_water_management(
        # Inputs del model elèctric
        surpluses=surpluses,
        level_base=df_nivel_si,
        thermal_generation=results['Gas+Imports'],
        base_hydro_generation=results['hydro_int'],
        
        # Paràmetres físics
        max_capacity_int=max_capacity_int,
        consumo_base_diario_estacional_hm3=consumo_base_diario_estacional_hm3,
        max_hydro_capacity_mw=potencia_max_int,
        overflow_threshold_pct=overflow_threshold_pct,
        sensitivity_mwh_per_percent=max_capacity_int * 0.01 * sensibility_int,

        # Paràmetres de regeneració
        regen_base_pct=regen_base_pct,
        llindar_activacio_regen_max=llindar_activacio_regen_max,
        max_regen_hm3_dia=max_regen,
        
        # Paràmetres de dessalinització
        llindar_activacio_desal_max=llindar_activacio_desal_max,
        min_run_hours=min_run_hours,
        max_desal_mw=max_desalation,
        midpoint=midpoint_estimation,
        save_hm3_per_mwh=save_hm3_per_mwh,
        seasonal_phase_months=seasonal_phase_months,
        seasonal_amplitude=seasonal_desal_amplitude,
        finestra_hores=trend_time_window,
        k_deriv=k_deriv,
        
        # Llindars de sequers
        umbrales_sequia=umbrales_sequia
        
    )
    
    # Conversió dessalinització a hm³
    desal_final_hm3 = desal_final * save_hm3_per_mwh
    
    # Càlcul de les restriccions industrials
    savings_ind_final = savings_final - savings_agro_final - savings_urba_final
    
    # =========================================================================
    # 5. ACOBLAMENT: MODIFICAR RESULTATS ELÈCTRICS
    # =========================================================================
    
    # Turbinació extra substitueix tèrmica
    results['hydro_int'] = results['hydro_int'] + extra_hydro_final
    results['Hidràulica'] = results['hydro_ebro'] + results['hydro_int']
    results['Gas+Imports'] = (results['Gas+Imports'] - extra_hydro_final).clip(lower=0)
    results['gap'] = results['gap'] - extra_hydro_final
    
    # Afegir sèries hídriques al DataFrame
    results['Hydro_Level_int'] = level_final
    results['surpluses_net'] = surpluses_net
    results['savings_hm3'] = savings_final
    results['regen_hm3'] = regen_final
    results['desal_mw'] = desal_final
    results['spillage_hm3'] = spillage_final
    
    
    # =========================================================================
    # CÀLCUL DE CONSUM EN HORES DE DÈFICIT
    # =========================================================================
    
    # Màscara d'hores amb dèficit (surpluses_net < 0)
    mask_deficit = surpluses_net <= 0
    mask_surplus = surpluses_net > 0
    mask_zero = surpluses_net == 0
    
    # paràmetre de sensibilitat energètica regeneració
    regen_cost_mwh_per_hm3 = 1200.0
    
    # Dessalinització en hores de dèficit [MWh]
    desal_en_deficit_mwh = desal_final[mask_deficit].sum()
    desal_en_surplus_mwh = desal_final[mask_surplus].sum()
    
    # Regeneració: convertir hm³ a MW (regen_hm3 * cost_mwh_per_hm3 = MWh per hora = MW)
    
    regen_mw = regen_final * regen_cost_mwh_per_hm3  # Sèrie en MW
    regen_en_deficit_mwh = regen_mw[mask_deficit].sum()
    regen_en_surplus_mwh = regen_mw[mask_surplus].sum()
    
    # Total consum hídric en dèficit
    consum_hidric_en_deficit_mwh = desal_en_deficit_mwh + regen_en_deficit_mwh
    consum_hidric_en_surplus_mwh = desal_en_surplus_mwh + regen_en_surplus_mwh
        
    regen_en_deficit_hm3 = regen_final[mask_deficit].sum()
    regen_en_surplus_hm3 = regen_final[mask_surplus].sum()
    desal_en_deficit_hm3 = (desal_final * save_hm3_per_mwh)[mask_deficit].sum()
    desal_en_surplus_hm3 = (desal_final * save_hm3_per_mwh)[mask_surplus].sum()
    
    # % dessalació en excedents (factor d'oportunisme)
    desal_FO = 100 * desal_en_surplus_hm3 / (desal_en_surplus_hm3 + desal_en_deficit_hm3)
    
    # Nombre d'arrancades i parades
    # 1. Definim un llindar de tolerància (opcional però recomanat)
    # A vegades els zeros flotants poden ser 0.0000001. Així evitem falsos positius.
    llindar = desal_final.min() + 0.001 
    # 2. Creem una sèrie binària: 1 si està funcionant, 0 si està parada
    estat = (desal_final > llindar).astype(int)
    # 3. Calculem la diferència amb l'hora anterior usant .diff()
    # Si passa de 0 a 1 -> 1 - 0 = 1  (Arrancada)
    # Si passa d'1 a 0 -> 0 - 1 = -1 (Parada)
    # Si no canvia d'estat -> 0
    canvis = estat.diff()
    # 4. Comptem els esdeveniments
    arrancades = (canvis == 1).sum()
    # parades = (canvis == -1).sum()
    
    # =========================================================================
    # 6. CÀLCUL DE MÈTRIQUES HÍDRIQUES
    # =========================================================================
    
    # Variació estacional (si hi ha prou dades)
    try:
        seasonal_capacity = seasonal_decompose(
            level_final.resample('ME').last(), 
            model='additive', 
            period=12
        ).seasonal
        monthly_seasonal = seasonal_capacity.groupby(seasonal_capacity.index.month).mean()
        seasonal_amplitude = monthly_seasonal.max() - monthly_seasonal.min()
    except:
        seasonal_amplitude = np.nan
    
    # Dies en restricció
    # dias_restriccion_historico = int(np.ceil((df_niveles_int[df_niveles_int * 100 < 40].count()) / 24))
    dias_restriccion_escenario = int(np.ceil(level_final[level_final < umbrales_sequia['Alerta']].count() / 24))
    
    hydro_metrics = {
        # 'Restricciones históricas (días)': dias_restriccion_historico,
        'Restricciones escenario (días)': dias_restriccion_escenario,
        'Llenado mínimo (%)': round(level_final.min(), 1),
        'Llenado promedio (%)': round(level_final.mean(), 1),
        'Variación estacional (%)': round(seasonal_amplitude, 1) if not np.isnan(seasonal_amplitude) else None,
        'Desalación FC (%)': int(100 * desal_final.sum() / (max_desalation*len(desal_final))),
        'Regeneración total (hm³)': round(regen_final.sum(), 1),
        'Desalación total (hm³)': round(desal_final_hm3.sum(), 1),
        'Estalvi restriccions (hm³)': round(savings_final.sum(), 1),
        'Regen. i Dessal. en dèficit (MWh)': round(consum_hidric_en_deficit_mwh, 1),
        'Regen. i Dessal. en excedents (MWh)': round(consum_hidric_en_surplus_mwh, 1),
        'Vessaments totals (hm³)': round(spillage_final.sum(), 2),
        'Factor oportunista (%)': desal_FO,
        'Arrancades': arrancades,
    }
    
    # =========================================================================
    # 7. CÀLCUL DE COSTOS D'INVERSIÓ
    # =========================================================================
    
    # costes = {
    #     'solar': (potencia_solar - potencia_solar_ref) * 775_000,
    #     'wind': (potencia_eolica - potencia_eolica_ref) * 1_400_000,
    #     'baterias': (bat_config[1] - 2000) * 325_000,
    #     'desalacion': max_desalation * 16_000_000
    # }
    # EURUSD en diciembre de 2024 = 1.04
    n_year = round(len(level_final) / (24 * 365))
    
    capex_solar = 600_000 * (1-0.04)**(10*0.65)
    capex_eolica = 1_100_000 * (1-0.02)**(10*0.65)
    capex_bate = 1_000_000 * (1-0.08)**(10*0.65)
    
    costes = {
        # 'solar': (potencia_solar - potencia_solar_ref) * 640_000 * n_year / 35, 
        # 'wind': (potencia_eolica - potencia_eolica_ref) * 963_000 * n_year / 25, 
        # 'baterias': (bat_config[0] - 532) * 712_000 * n_year / 15,
        'solar': (potencia_solar - potencia_solar_ref) * capex_solar * n_year / 35, 
        'wind': (potencia_eolica - potencia_eolica_ref) * capex_eolica * n_year / 25, 
        'baterias': (bat_config[0] - 532) * capex_bate * n_year / 20,
        
        'desalacion': (max_desalation - max_desalation_ref) * 15_500_000 * n_year / 30, # from 80hm3/y * 6200000€/(hm3/y) / 32MW [€/MW]
        'regeneracion': (max_regen - 0.173)*365 * 4_130_000 * n_year / 30,
        'restriccions': 2_000_000 * savings_agro_final.sum() + 35_000_000 * savings_ind_final.sum()
        + 1_600_000 * savings_urba_final.sum(),
        'desal_opex': 556_000 * desal_en_deficit_hm3 + 206_000 * desal_en_surplus_hm3,
        'regen_opex': 190_000 * regen_en_deficit_hm3 + 70_000 * regen_en_surplus_hm3,
        # 'estalvi_gas': (results['Gas+Imports'].sum() - 87380181.0257) * 100,
        'solar_opex': (potencia_solar - potencia_solar_ref) * 19000 * n_year,
        'wind_opex': (potencia_eolica - potencia_eolica_ref) * 27000 * n_year,
        'bate_opex': (bat_config[0] - 532) * 18000 * n_year
    }
    
    costes['total'] = sum(costes.values())
    
    costes['opex'] = costes['restriccions'] + costes['desal_opex'] + costes['regen_opex'] + costes['solar_opex'] + costes['wind_opex'] + costes['bate_opex']
    costes['elec'] = costes['solar'] + costes['solar_opex'] + costes['wind'] + costes['wind_opex'] + costes['baterias'] + costes['bate_opex']
    costes['hidr'] = costes['desalacion'] + costes['regeneracion'] + costes['desal_opex'] + costes['regen_opex']
    costes['rest'] = costes['restriccions']
    # =========================================================================
    # 8. RETORN DE RESULTATS
    # =========================================================================
    
    return {
        'energy_data': results,
        'energy_metrics_pct': energy_metrics_pct,
        'energy_metrics_MWh': energy_metrics_MWh,
        'energy_metrics_pct2': energy_metrics_pct2,
        'hydro_metrics': hydro_metrics,
        'level_final': level_final,
        'regen_final': regen_final,
        'desal_final': desal_final,
        'desal_final_hm3': desal_final_hm3,
        'savings_final': savings_final,
        'savings_agro_final': savings_agro_final,
        'savings_ind_final': savings_ind_final,
        'savings_urba_final': savings_urba_final,
        'spillage_hm3': spillage_final,
        'extra_hydro': extra_hydro_final,
        'excedents': surpluses_net[surpluses_net>0],
        'deficits': surpluses_net[surpluses_net<0],
        'capacity_factor': int(100 * desal_final.mean() / max_desalation),
        'costes': costes
    }

#%%
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# =============================================================================
# CONFIGURACIÓ DE VARIABLES DE DECISIÓ
# =============================================================================
config = {
    # --- Variables elèctriques ---
    # PROENCAT 2030
    # 'potencia_solar':      {'actiu': True,  'rang': np.arange(400, 10100, 100),   'defecte': 7180},
    # 'potencia_eolica':     {'actiu': True,  'rang': np.arange(1400, 10100, 100),  'defecte': 6234},
    # 'potencia_baterias':   {'actiu': True,  'rang': np.arange(500, 3100, 100),    'defecte': 2000},

    # PROENCAT 2050
    'potencia_solar':      {'actiu': False,  'rang': np.arange(1000, 30100, 100),   'defecte': 17431}, #CAMBIO 1
    'potencia_eolica':     {'actiu': False,  'rang': np.arange(1000, 30100, 100),  'defecte': 18439},
    'potencia_baterias':   {'actiu': False,  'rang': np.arange(0, 11000, 2000),    'defecte': 4034},
    
    # --- Variables hídriques generals ---
    'min_run_hours':       {'actiu': True,  'rang': np.arange(6, 13, 1),          'defecte': 8},
    'max_desalation':      {'actiu': True,  'rang': np.arange(32, 97, 1),        'defecte': 64}, # CAMBIO 2
    'max_regen':           {'actiu': True,  'rang': np.arange(0.173, 0.351, 0.001), 'defecte': 0.350}, #TRUE !!!
    'llindar_desal_max':   {'actiu': True,  'rang': np.arange(1, 5, 1),           'defecte': 2}, #TRUE !!!
    'midpoint_estimation': {'actiu': True,  'rang': np.arange(50, 96, 1),         'defecte': 95},
    'overflow_threshold':  {'actiu': True,  'rang': np.arange(50, 96, 1),         'defecte': 90},
    
    'seasonal_phase':      {'actiu': False,  'rang': np.arange(0, 11.9, 0.1),      'defecte': 0.0},
    'seasonal_amplitude':  {'actiu': False,  'rang': np.arange(0, 1.1, 0.1),       'defecte': 0.0},
    'regen_base_pct':      {'actiu': False,  'rang': np.arange(0.35, 0.65, 0.05),  'defecte': 0.5},
    'llindar_regen_max':   {'actiu': False,  'rang': np.arange(1, 5, 1),           'defecte': 1}, #TRUE !!!
    'derivada_nivell':     {'actiu': False,  'rang': np.arange(0, 11, 0.5),       'defecte': 0.0},
    
    # --- Llindars de sequera (per increments) ---
    'x1_base_eme':         {'actiu': True, 'rang': np.arange(10, 21, 1),         'defecte': 16}, #TRUE A TOTS!!!
    'x2_gap_exc':          {'actiu': True, 'rang': np.arange(5, 21, 1),         'defecte': 9},
    'x3_gap_ale':          {'actiu': True, 'rang': np.arange(5, 21, 1),         'defecte': 15}, #se puede dejar que alerta y prealerta fluctuen
    'x4_gap_pre':          {'actiu': True, 'rang': np.arange(5, 21, 1),         'defecte': 20},
}

# nucleares_activas=[True, True, True]
# potencia_cogeneracion=943.503
# potencia_autoconsum=2185
# demanda_electrica=1  #1.25

nucleares_activas=[False, False, False]
potencia_cogeneracion=122.4
potencia_autoconsum=  5000 #2185 #7250
demanda_electrica=1.5


# Restricció global per llindars
MAX_PREALERTA = 75

# =============================================================================
# FUNCIÓ DE TRANSFORMACIÓ DE LLINDARS
# =============================================================================
def increments_a_llindars(x1, x2, x3, x4):
    """Transforma increments en llindars absoluts."""
    L_eme = x1
    L_exc = x1 + x2
    L_ale = x1 + x2 + x3
    L_pre = x1 + x2 + x3 + x4
    
    return {
        'Emergencia_3': L_eme * 0.35,
        'Emergencia_2': L_eme * 0.70,
        'Emergencia': L_eme,
        'Excepcionalitat': L_exc,
        'Alerta': L_ale,
        'Prealerta': L_pre
    }

# =============================================================================
# GENERADOR D'ESCENARIS
# =============================================================================
def generar_escenaris(config, n_samples, seed=42, max_prealerta=90):
    """Genera escenaris segons configuració activa."""
    rng = np.random.default_rng(seed)
    escenaris = []
    attempts = 0
    max_attempts = n_samples * 20
    
    while len(escenaris) < n_samples and attempts < max_attempts:
        attempts += 1
        params = {}
        
        # Generar valors per cada variable
        for var, cfg in config.items():
            if cfg['actiu']:
                valor = rng.choice(cfg['rang'])
                # Convertir a int si el rang és enter
                if cfg['rang'].dtype in [np.int32, np.int64]:
                    valor = int(valor)
                params[var] = valor
            else:
                params[var] = cfg['defecte']
        
        # Restricció global llindars (només si algun està actiu)
        llindars_actius = any(config[f'x{i}_{"base_eme" if i==1 else "gap_exc" if i==2 else "gap_ale" if i==3 else "gap_pre"}']['actiu'] 
                             for i in range(1, 5))
        
        if llindars_actius:
            suma = params['x1_base_eme'] + params['x2_gap_exc'] + params['x3_gap_ale'] + params['x4_gap_pre']
            if suma > max_prealerta:
                continue
        
        escenaris.append(params)
    
    if len(escenaris) < n_samples:
        print(f"AVÍS: Només s'han generat {len(escenaris)} escenaris vàlids de {n_samples} sol·licitats")
    
    return escenaris

# =============================================================================
# FUNCIÓ D'AVALUACIÓ
# =============================================================================
def run_case(params):
    """Executa un escenari amb paràmetres en format diccionari."""
    
    # Transformar llindars
    umbrales = increments_a_llindars(
        params['x1_base_eme'],
        params['x2_gap_exc'],
        params['x3_gap_ale'],
        params['x4_gap_pre']
    )
    
    results = procesar_escenario(
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
        # ---- Sèries a modificar segons escenari d'estrés -------
        df_energia_turbinada_mensual_internes=datos.energia_turbinada_mensual_internes,
        df_energia_turbinada_mensual_ebre=datos.energia_turbinada_mensual_ebre,
        df_nivel_si=hydro_base_level,
        
        # --- Paràmetres físics ---
        max_capacity_int=max_capacity_int,
        max_capacity_ebro=max_capacity_ebro,
        potencia_max_int=potencia_max_hidraulica_int,
        potencia_max_ebro=potencia_max_hidraulica_ebro,
        sensibility_int=sensibility_int,
        sensibility_ebro=sensibility_ebro,
        
        # --- Paràmetres hídrics ---
        consumo_base_diario_estacional_hm3=consumo_base_diario_estacional,
        save_hm3_per_mwh=1/desal_sensibility,
        
        # --- Dades precomputades ---
        precomputed=precomputed,
        
        # --- Variables fixes ---
        nucleares_activas=nucleares_activas,
        # nucleares_activas=[False, False, False],
        potencia_cogeneracion=potencia_cogeneracion,
        # potencia_cogeneracion=122.4,
        duracion_horas=4,
        potencia_autoconsumo= potencia_autoconsum, #2185,#1381,
        demanda_electrica=demanda_electrica,
        CF_eolica_obj=None,
        usar_CF_automatic=True,
        trend_time_window=24*30,
        
        # --- Variables de decisió (del diccionari) ---
        potencia_solar=params['potencia_solar'],
        potencia_eolica=params['potencia_eolica'],
        potencia_baterias=params['potencia_baterias'],
        min_run_hours=params['min_run_hours'],
        max_desalation=params['max_desalation'],
        midpoint_estimation=params['midpoint_estimation'],
        overflow_threshold_pct=params['overflow_threshold'],
        llindar_activacio_desal_max=params['llindar_desal_max'],        
        seasonal_phase_months=params['seasonal_phase'],
        seasonal_desal_amplitude=params['seasonal_amplitude'],
        regen_base_pct=params['regen_base_pct'],
        llindar_activacio_regen_max=params['llindar_regen_max'],
        max_regen=params['max_regen'],
        k_deriv=params['derivada_nivell'],
        umbrales_sequia=umbrales,
    )
    
    # Calcular llindars absoluts per output
    x1, x2, x3, x4 = params['x1_base_eme'], params['x2_gap_exc'], params['x3_gap_ale'], params['x4_gap_pre']
    
    return {
        # === Variables d'entrada ===
        **params,  # Inclou totes les variables de decisió
        
        # Llindars en format llegible
        "L_emergencia": x1,
        "L_excepcionalitat": x1 + x2,
        "L_alerta": x1 + x2 + x3,
        "L_prealerta": x1 + x2 + x3 + x4,
        "llindars_fases": [x1, x1+x2, x1+x2+x3, x1+x2+x3+x4],
        
        # === Mètriques de sortida ===
        "energy_metrics_pct": results['energy_metrics_pct'],
        "energy_metrics_MWh": results['energy_metrics_MWh'],
        "energy_metrics_pct2": results['energy_metrics_pct2'],
        "min_level": results["level_final"].min(),
        "max_level": results["level_final"].max(),
        "mean_level": results['level_final'].mean(),
        "squared_dev": ((100 - results['level_final'])**2).sum() / len(results['level_final']),
        "desal_hm3": results['desal_final_hm3'].sum(),
        "regen_hm3": results['regen_final'].sum(),
        "spillage_hm3": results['spillage_hm3'].sum(),
        "regen_desal_deficit_MWh": results['hydro_metrics']['Regen. i Dessal. en dèficit (MWh)'],
        "restriction_days": results['hydro_metrics']['Restricciones escenario (días)'],
        "restriction_savings": results['savings_final'].sum(),
        "desal_cf": results['capacity_factor'],
        "surpluses_total": results['excedents'].sum(),
        "deficits_total": results['deficits'].sum(),
        "gas_imports": results['energy_data']['Gas+Imports'].sum(),
        "total_costs": results['costes']['total'],
        "opex_costs": results['costes']['opex'],
        "costs": results['costes'],
        "seasonal_amplitude_out": results['hydro_metrics']['Variación estacional (%)'],
    }

#%%

# =============================================================================
# EXECUCIÓ
# =============================================================================
n_samples = 30000 # 50000 #50000

# # Exemple: Només optimitzar solar, eòlica i dessalinització
# for var in config:
#     config[var]['actiu'] = False  # Desactivar tot

# config['potencia_solar']['actiu'] = True
# config['potencia_eolica']['actiu'] = True
# config['max_desalation']['actiu'] = True

# # Exemple: Activar optimització de llindars
# config['x1_base_eme']['actiu'] = True
# config['x2_gap_exc']['actiu'] = True
# config['x3_gap_ale']['actiu'] = True
# config['x4_gap_pre']['actiu'] = True


# Generar escenaris
escenaris = generar_escenaris(config, n_samples, seed=987654321, max_prealerta=MAX_PREALERTA)

# Executar en paral·lel
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# results = Parallel(n_jobs=-1, verbose=10)(
#     delayed(run_case)(params) for params in escenaris
# )

import time
t0 = time.time()

results = Parallel(n_jobs=-1, verbose=0, max_nbytes=None)(
    delayed(run_case)(params) for params in tqdm(escenaris, desc="Escenaris")
)
print(f"Temps total: {time.time() - t0:.1f}s")

# Convertir a DataFrame
df_rgs_2040 = pd.DataFrame(results)

df_rgs_2040.to_parquet('df_rgs_30k_2040_hydro.parquet')

# df_rgs_mix = pd.read_parquet('df_rgs_50k_2040_mix.parquet')
#%%

import time
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("DEAP no disponible. Instal·la amb: pip install deap")



# Afegeix això ABANS de la classe OptimizadorEnergetico
def increments_a_llindars(x1, x2, x3, x4):
    """Transforma increments en llindars absoluts."""
    L_eme = x1
    L_exc = x1 + x2
    L_ale = x1 + x2 + x3
    L_pre = x1 + x2 + x3 + x4
    
    return {
        'Emergencia_3': L_eme * 0.35,
        'Emergencia_2': L_eme * 0.70,
        'Emergencia': L_eme,
        'Excepcionalitat': L_exc,
        'Alerta': L_ale,
        'Prealerta': L_pre
    }

class OptimizadorEnergetico:
    """
    Classe per optimitzar escenaris energètics usant diferents metaheurístiques.
    """
    
    def __init__(self, datos_base, funcion_procesar_escenario, llindars_fixos=None):
        self.datos_base = datos_base
        self.procesar_escenario = funcion_procesar_escenario
        self.mejor_resultado = None
        self.historial_evaluaciones = []
        self.variables = None
        self.cache_resultados = {}
        self.llindars_fixos = llindars_fixos or {}

        
    def definir_variables_decision(self):
        """Defineix els rangs de les variables de decisió."""
        self.variables = {
            # 'potencia_solar': {'min': 400, 'max': 10000},   #ALERTA ------------
            # 'potencia_eolica': {'min': 1400, 'max': 10000},
            # 'potencia_baterias': {'min': 500, 'max': 5000},
            
            'potencia_solar': {'min': 2000, 'max': 30000},
            'potencia_eolica': {'min': 6000, 'max': 30000},
            'potencia_baterias': {'min': 1000, 'max': 15000},
            
            # 'max_desalation': {'min': 32, 'max': 240},
            'max_desalation': {'min': 64, 'max': 240},  # ALERTA
            # 'overflow_threshold_pct': {'min': 40, 'max': 95},
            'overflow_threshold_pct': {'min': 50, 'max': 95},
            'min_run_hours': {'min': 6, 'max': 12, 'tipo': 'integer'},
            # 'midpoint_estimation': {'min': 10, 'max': 95},
            'midpoint_estimation': {'min': 50, 'max': 95},
            # 'seasonal_phase_months': {'min': 0, 'max': 11.9},
            # 'seasonal_desal_amplitude': {'min': 0, 'max': 1.0},
            # 'max_regen': {'min': 0.173, 'max': 0.350},
            'llindar_activacio_desal_max': {'min': 1, 'max': 5, 'tipo': 'integer'}, #ALERTA -------------
            'llindar_activacio_regen_max': {'min': 1, 'max': 4, 'tipo': 'integer'},
            # 'x1_base_eme': {'min': 10, 'max': 20, 'tipo': 'integer'},
            # 'x2_gap_exc': {'min': 5, 'max': 20, 'tipo': 'integer'},
            'x3_gap_ale': {'min': 5, 'max': 20, 'tipo': 'integer'},  # ALERTA  -----------------
            'x4_gap_pre': {'min': 5, 'max': 20, 'tipo': 'integer'},
        }
        
        # Bounds per scipy (ordre important!)
        self.bounds = [(v['min'], v['max']) for v in self.variables.values()]
        self.var_names = list(self.variables.keys())
        
    def decodificar_individuo(self, x):
        """Converteix el vector d'optimització en paràmetres."""
        params = {}
        for i, name in enumerate(self.var_names):
            val = x[i]
            if self.variables[name].get('tipo') == 'integer':
                val = int(round(val))
            params[name] = val
        return params
    
    # def evaluar_escenario(self, params):
    #     """Executa la simulació i retorna les mètriques."""
    #     try:
    #         resultado = self.procesar_escenario(
    #             **self.datos_base,
    #             **params
    #         )
    #         return resultado
    #     except Exception as e:
    #         print(f"Error en simulació: {e}")
    #         return None
        
    def evaluar_escenario(self, params):
        """Executa la simulació i retorna les mètriques."""
        try:
            # Clau per al cache
            cache_key = tuple(sorted(params.items())) 
            
            # Usar valors fixos de datos_base si no estan a params
            # Obtenir llindars (de params si optimitzats, de llindars_fixos si no)
            x1 = params.pop('x1_base_eme', self.llindars_fixos.get('x1_base_eme'))
            x2 = params.pop('x2_gap_exc', self.llindars_fixos.get('x2_gap_exc'))
            x3 = params.pop('x3_gap_ale', self.llindars_fixos.get('x3_gap_ale'))
            x4 = params.pop('x4_gap_pre', self.llindars_fixos.get('x4_gap_pre'))           
            
            umbrales = increments_a_llindars(x1, x2, x3, x4)
            
            # umbrales = increments_a_llindars(
            #     params.pop('x1_base_eme'),
            #     params.pop('x2_gap_exc'),
            #     params.pop('x3_gap_ale'),
            #     params.pop('x4_gap_pre')
            # )
            
            resultado = self.procesar_escenario(
                **self.datos_base,
                **params,
                umbrales_sequia=umbrales
            )
            # Guardar al cache
            self.cache_resultados[cache_key] = resultado            
            
            return resultado
        except Exception as e:
            print(f"Error en simulació: {e}")
            return None
    
    # =========================================================================
    # DIFFERENTIAL EVOLUTION (Scipy)
    # =========================================================================
    
    def funcion_objetivo_escalar(self, x, pesos=None):
        """
        Funció objectiu escalar per Differential Evolution.
        Combina múltiples objectius amb pesos.
        """
        if pesos is None:
            pesos = {
                'mean_sqd_dev': 1/3, #0.4,
                'gas_imports': 1/3, #0.2,
                # 'mean_level:' -0.5,
                #'min_level': -0.5,  # Negatiu perquè volem maximitzar
                'total_costs': 1/3, #0.4,
                # 'restriction_days': 0.1
            }
        
        params = self.decodificar_individuo(x)
        resultado = self.evaluar_escenario(params)
        
        if resultado is None:
            return 1e10
        
        # Verificar desbordament
        if resultado['level_final'].max() > 100.5:
            return 1e10
        
        # Normalitzar i combinar objectius
        msqd = ((100 - resultado['level_final'])**2).sum() / len(resultado['level_final'])
        objetivo = (
            pesos['gas_imports'] * normalitzar(resultado['energy_data']['Gas+Imports'].sum(), *gasimports_range) +
            pesos['mean_sqd_dev'] * normalitzar(msqd, *msqdev_range) +
            # pesos['min_level'] * normalitzar(resultado['level_final'].min(), *min_level_range) +
            pesos['total_costs'] * normalitzar(resultado['costes']['total'], *costs_range) 
            # pesos['restriction_days'] * (resultado['hydro_metrics']['Restricciones escenario (días)'] / 1000)
        )
        
        # Guardar historial
        self.historial_evaluaciones.append({
            'params': params,
            'objetivo': objetivo,
            'min_level': resultado['level_final'].min(),
            'gas_imports': resultado['energy_data']['Gas+Imports'].sum()
        })
        
        return objetivo
    
    def optimizar_differential_evolution(self, maxiter=50, popsize=15, seed=42):
        """Optimització amb Differential Evolution de scipy."""
        print("Iniciant optimització amb Differential Evolution...")
        print(f"  maxiter={maxiter}, popsize={popsize}")
        
        self.historial_evaluaciones = []
        
        def callback(xk, convergence):
            n = len(self.historial_evaluaciones)
            if n % 10 == 0 and n > 0:
                valids = [h['objetivo'] for h in self.historial_evaluaciones if h['objetivo'] < 1e5]
                if valids:
                    print(f"  Eval {n}: Millor obj = {min(valids):.4f}")
        
        resultado = differential_evolution(
            self.funcion_objetivo_escalar,
            self.bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            callback=callback,
            disp=True,
            workers=1,  # No paral·lelitzar internament (ja ho fem a nivell de simulació)
            # updating='deferred',
            polish=False
        )
        
        mejor_params = self.decodificar_individuo(resultado.x)
        
        print(f"\nOptimització completada!")
        print(f"Millor objectiu: {resultado.fun:.4f}")
        print("Millors paràmetres:")
        for k, v in mejor_params.items():
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
        
        return resultado, mejor_params
    
    # =========================================================================
    # NSGA-II (DEAP)
    # =========================================================================
    
    def optimizar_nsga2(self, ngen=30, pop_size=50, seed=42, parallel=True, pop_inicial=None, hof_inicial=None):  # AFEGIT parallel
        """Optimització multiobjectiu amb NSGA-II de DEAP."""
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP no disponible. Instal·la amb: pip install deap")
        
        print(f"Iniciant NSGA-II: ngen={ngen}, pop_size={pop_size}, parallel={parallel}")
        
        # Netejar creators anteriors si existeixen
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # Definir fitness: minimitzar emissions, maximitzar min_level, minimitzar costos
        # weights: -1 = minimitzar, +1 = maximitzar
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        toolbox = base.Toolbox()
        
        # Registrar generadors per cada variable
        for i, (name, cfg) in enumerate(self.variables.items()):
            if cfg.get('tipo') == 'integer':
                toolbox.register(f"attr_{i}", np.random.randint, cfg['min'], cfg['max'] + 1)
            else:
                toolbox.register(f"attr_{i}", np.random.uniform, cfg['min'], cfg['max'])
        
        # Crear funció per generar individu
        def crear_individuo():
            ind = []
            for i, (name, cfg) in enumerate(self.variables.items()):
                if cfg.get('tipo') == 'integer':
                    ind.append(np.random.randint(cfg['min'], cfg['max'] + 1))
                else:
                    ind.append(np.random.uniform(cfg['min'], cfg['max']))
            return creator.Individual(ind)
        
        toolbox.register("individual", crear_individuo)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Funció d'avaluació multiobjectiu
        def evaluar_multiobjetivo(individual):
            params = self.decodificar_individuo(individual)
            resultado = self.evaluar_escenario(params)
            
            # # Guardar resultat a l'individu
            # individual.resultado = resultado  # 👈 CLAVE
            
            if resultado is None:
                return (1e10, 1e10, 1e10)
            
            # Penalitzar desbordament
            if resultado['level_final'].max() > 100.5:
                return (1e10, 1e10, 1e10)
            
            # Definim el gap i el factor de penalització (ex: 2 vegades més greu el dèficit)
            gap = resultado['energy_data']['gap'] / 1e6
            alpha = 5.0
            # Creem el vector de pesos: alpha si falta energia (>0), 1 si en sobra (<=0)
            weights = np.where(gap > 0, alpha, 1.0)
            
            # Objectius: emissions (min), mean_sq_dev (min), costos (min)
            # residu = ((resultado['energy_data']['gap']/1e6)**2).sum()
            # residu = resultado['energy_data']['Gas+Imports'].sum() / 1e6  # TWh
            # Multipliquem els pesos pel quadrat abans de sumar
            residu = (weights * (gap**2)).sum()
            mean_sq_dev = ((100 - resultado['level_final'])**2).sum() / len(resultado['level_final'])
            # costos = (resultado['costes']['total'] - resultado['costes']['invest']) / 1e6
            costos = resultado['costes']['total'] / 1e6  # M€
            # costos = resultado['costes']['hidr'] / 1e6  # M€
            
            return (residu, mean_sq_dev, costos), resultado
        
        toolbox.register("evaluate", evaluar_multiobjetivo)
        
        # Operadors genètics
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                        low=[b[0] for b in self.bounds],
                        up=[b[1] for b in self.bounds],
                        eta=20.0)
        
        toolbox.register("mutate", tools.mutPolynomialBounded,
                        low=[b[0] for b in self.bounds],
                        up=[b[1] for b in self.bounds],
                        eta=20.0,
                        indpb=1.0/len(self.bounds))
        
        toolbox.register("select", tools.selNSGA2)
        
        # =========================================================================
        # CONFIGURACIÓ AVALUACIÓ: SEQÜENCIAL O PARAL·LEL
        # =========================================================================
        if parallel:
            def evaluar_batch(individuals):
                return Parallel(n_jobs=-1, backend='loky')(
                    delayed(evaluar_multiobjetivo)(ind) for ind in individuals
                )
        else:
            def evaluar_batch(individuals):
                return [evaluar_multiobjetivo(ind) for ind in individuals]
        
        # Seed per reproductibilitat
        np.random.seed(seed)
        
        
        # =========================================================================
        # AVALUAR POBLACIÓ INICIAL o Continuació
        # =========================================================================
        if pop_inicial is not None:
            print(f"Continuant des de població anterior ({len(pop_inicial)} individus)...")
            # Reconstruir individus amb el nou creator
            pop = []
            for ind_old in pop_inicial:
                ind_new = creator.Individual(list(ind_old))
                # Copiar fitness si existeix
                if hasattr(ind_old, 'fitness') and ind_old.fitness.valid:
                    ind_new.fitness.values = ind_old.fitness.values
                if hasattr(ind_old, 'resultado'):
                    ind_new.resultado = ind_old.resultado
                pop.append(ind_new)
            
            # Avaluar només els que no tenen fitness vàlid
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            if invalid_ind:
                print(f"  Avaluant {len(invalid_ind)} individus sense fitness...")
                fitnesses = evaluar_batch(invalid_ind)
                for ind, (fit, res) in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                    ind.resultado = res
        else:
            # Població nova
            pop = toolbox.population(n=pop_size)
            print("Avaluant població inicial...")
            fitnesses = evaluar_batch(pop)
            for ind, (fit, res) in zip(pop, fitnesses):
                ind.fitness.values = fit
                ind.resultado = res        
        
        # Hall of Fame (Pareto front)
        if hof_inicial is not None:
            print(f"Continuant amb HoF anterior ({len(hof_inicial)} individus)...")
            hof = tools.ParetoFront()
            # Reconstruir individus per al nou hof
            for ind_old in hof_inicial:
                ind_new = creator.Individual(list(ind_old))
                if hasattr(ind_old, 'fitness') and ind_old.fitness.valid:
                    ind_new.fitness.values = ind_old.fitness.values
                hof.update([ind_new])
        else:
            hof = tools.ParetoFront()

        
        pop = toolbox.select(pop, len(pop))
        hof.update(pop)
        
        # =========================================================================
        # EVOLUCIÓ
        # =========================================================================
        for gen in range(ngen):
            # Seleccionar i clonar
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]
            
            # Creuar
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < 0.9:
                    toolbox.mate(ind1, ind2)
                    del ind1.fitness.values
                    del ind2.fitness.values
            
            # Mutar
            for ind in offspring:
                if np.random.random() < 0.2:
                    toolbox.mutate(ind)
                    del ind.fitness.values
            
            # Avaluar offspring amb fitness invàlid
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = evaluar_batch(invalid_ind)  # CANVIAT: usa evaluar_batch
            for ind, (fit,res) in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                ind.resultado = res 
            
            # Seleccionar nova generació
            pop = toolbox.select(pop + offspring, pop_size)
            hof.update(pop)
            
            # Mostrar progrés
            if (gen + 1) % 5 == 0 or gen == 0:
                fits = [ind.fitness.values for ind in pop]
                print(f"  Gen {gen+1}: Pareto size={len(hof)}, "
                      f"GapSq[TWh]=[{min(f[0] for f in fits):.1f}, {max(f[0] for f in fits):.1f}], "
                      f"MSD=[{min(f[1] for f in fits):.1f}, {max(f[1] for f in fits):.1f}], "
                      f"Cost[M€]=[{min(f[2] for f in fits):.1f}, {max(f[2] for f in fits):.1f}]")
        
        print(f"\nNSGA-II completat! Solucions Pareto: {len(hof)}")
        
        return pop, hof    

    def hof_to_dataframe_old(self, hof):
        """Converteix el Hall of Fame a DataFrame."""
        records = []
        for ind in hof:
            params = self.decodificar_individuo(ind)
            params['obj_gapsq'] = ind.fitness.values[0]
            params['obj_msqdev'] = ind.fitness.values[1]
            params['obj_costs'] = ind.fitness.values[2]                     
            records.append(params)
        return pd.DataFrame(records)


    def hof_to_dataframe(self, hof):
        """Converteix el Hall of Fame a DataFrame amb mètriques addicionals."""
        records = []
        for ind in hof:
            params = self.decodificar_individuo(ind)
            cache_key = tuple(sorted(params.items()))
            
            
            # # # PRIMER: Re-avaluar amb còpia neta (sense objectius)
            # # resultado = self.evaluar_escenario(params.copy())
            # # Recuperar del cache (o re-avaluar si no existeix)
            # resultado = self.cache_resultados.get(cache_key)
            resultado = getattr(ind, "resultado", None)
            if resultado is None:
                resultado = self.evaluar_escenario(params.copy())            

            
            # DESPRÉS: Afegir objectius optimitzats
            # params['obj_gasimports'] = ind.fitness.values[0]
            params['obj_gapsq'] = ind.fitness.values[0]
            params['obj_msqdev'] = ind.fitness.values[1]
            params['obj_costs'] = ind.fitness.values[2]
            
            
            
            if resultado is not None:
                # Mètriques de nivell
                params['min_level'] = resultado['level_final'].min()
                params['max_level'] = resultado['level_final'].max()
                params['mean_level'] = resultado['level_final'].mean()
                params['squared_dev'] = ((100 - resultado['level_final'])**2).mean()
                
                # Mètriques hídriques
                params['desal_hm3'] = resultado['desal_final_hm3'].sum()
                params['regen_hm3'] = resultado['regen_final'].sum()
                params['spillage_hm3'] = resultado['spillage_hm3'].sum()
                params['restriction_days'] = resultado['hydro_metrics']['Restricciones escenario (días)']
                params['restriction_savings'] = resultado['savings_final'].sum()
                params['desal_cf'] = resultado['capacity_factor']
                
                # Mètriques elèctriques
                params['surpluses_total'] = resultado['excedents'].sum()
                params['deficits_total'] = resultado['deficits'].sum()
                params['gas_imports'] = resultado['energy_data']['Gas+Imports'].sum() / 1e6

                def get_llindar_value(var_name):
                    if var_name in self.var_names:
                        return int(round(ind[self.var_names.index(var_name)]))
                    else:
                        return self.llindars_fixos.get(var_name)
                
                # Llindars
                x1 = get_llindar_value('x1_base_eme')
                x2 = get_llindar_value('x2_gap_exc')
                x3 = get_llindar_value('x3_gap_ale')
                x4 = get_llindar_value('x4_gap_pre')               
                umbrales = increments_a_llindars(x1, x2, x3, x4)
                # umbrales = increments_a_llindars(
                #     int(round(ind[self.var_names.index('x1_base_eme')])),
                #     int(round(ind[self.var_names.index('x2_gap_exc')])),
                #     int(round(ind[self.var_names.index('x3_gap_ale')])),
                #     int(round(ind[self.var_names.index('x4_gap_pre')]))
                # )
                params['L_emergencia'] = umbrales['Emergencia']
                params['L_excepcionalitat'] = umbrales['Excepcionalitat']
                params['L_alerta'] = umbrales['Alerta']
                params['L_prealerta'] = umbrales['Prealerta']
                
                params['costes'] = resultado['costes']
                
                params['energy_metrics_pct'] = resultado['energy_metrics_pct']
                params['energy_metrics_MWh'] = resultado['energy_metrics_MWh']
                params['energy_metrics_pct2'] = resultado['energy_metrics_pct2']
            
            records.append(params)
        return pd.DataFrame(records)

# =============================================================================
# CONFIGURACIÓ I EXECUCIÓ
# =============================================================================

def configurar_optimizador(procesar_escenario_func, precomputed, datos):
    """Configura l'optimitzador amb les dades necessàries."""
    
    datos_base = {
        # Sèries de dades
        'df_demanda': datos.demanda,
        'df_nuclear': datos.nuclears_base,
        'df_cogeneracion': datos.cogeneracion_h,
        'df_solar': datos.solar_h,
        'df_eolica': datos.eolica_h,
        'df_autoconsum': datos.autoconsum_hourly,
        'df_potencia': datos.potencia,
        'df_niveles_int': datos.df_pct_int_h.squeeze(),
        'df_niveles_ebro': datos.df_pct_ebre_h.squeeze(),
        'df_energia_turbinada_mensual_internes': datos.energia_turbinada_mensual_internes,
        'df_energia_turbinada_mensual_ebre': datos.energia_turbinada_mensual_ebre,
        'df_nivel_si': hydro_base_level,
        
        # Paràmetres físics
        'max_capacity_int': max_capacity_int,
        'max_capacity_ebro': max_capacity_ebro,
        'potencia_max_int': potencia_max_hidraulica_int,
        'potencia_max_ebro': potencia_max_hidraulica_ebro,
        'sensibility_int': sensibility_int,
        'sensibility_ebro': sensibility_ebro,
        
        # Paràmetres hídrics
        'consumo_base_diario_estacional_hm3': consumo_base_diario_estacional,
        'save_hm3_per_mwh': 1/desal_sensibility,
        
        # Precomputats
        'precomputed': precomputed,
        
        # Variables fixes
        # 'nucleares_activas': [True, True, True], # ALERT -----------------
        # 'potencia_cogeneracion': 542,
        # 'potencia_cogeneracion': 943.503,
        'nucleares_activas': [False, False, False],
        'potencia_cogeneracion': 122.4,

        'duracion_horas': 4,
        # 'potencia_autoconsumo': 2185, # ALERT ------------------
        'potencia_autoconsumo': 7275,
        # 'demanda_electrica': 1.25, # ALERT-------------------
        'demanda_electrica': 2,
        'CF_eolica_obj': None,
        'usar_CF_automatic': True,
        'trend_time_window': 24*30,
        'k_deriv': 0,
        'regen_base_pct': 0.5,
        # 'llindar_activacio_desal_max': 4, # ALERT ------------
        'seasonal_phase_months': 0,
        'seasonal_desal_amplitude': 0,
        # 'max_desalation': 64,   # ALERT 2030
        # 'max_regen': 0.250,  # ALERT 2030
        'max_regen': 0.350,
        # 'overflow_threshold': 90,
        # 'umbrales_sequia': increments_a_llindars(16, 9, 15, 20),  # Valors per defecte
    }
    
    # Llindars fixos (separats de datos_base)
    llindars_fixos = {
        'x1_base_eme': 16,
        'x2_gap_exc': 9,
        # 'x3_gap_ale': 15, # ALERT ------------------
        # 'x4_gap_pre': 20, # ALERT ------------------
    }    
    
    optimizador = OptimizadorEnergetico(datos_base, procesar_escenario_func, llindars_fixos=llindars_fixos)
    optimizador.definir_variables_decision()
    
    return optimizador


#%%
# =============================================================================
# EXEMPLE D'ÚS
# =============================================================================

if __name__ == "__main__":
    
    # Configurar
    optimizador = configurar_optimizador(procesar_escenario, precomputed, datos)
    
    
    # # Opció 1: Differential Evolution (escalar, més ràpid)
    # print("\n" + "="*60)
    # print("DIFFERENTIAL EVOLUTION")
    # print("="*60)
    # t0 = time.time()
    # resultado_de, params_de = optimizador.optimizar_differential_evolution(
    #     maxiter=25, #20,
    #     popsize=20,
    #     seed=42
    # )
    # print(f"Temps: {time.time()-t0:.1f}s") #654s 2 iteraciones

# -----------------------------------------------------------------------------
    
    # Opció 2: NSGA-II (multiobjectiu)
    print("\n" + "="*60)
    print("NSGA-II")
    print("="*60)
    t0 = time.time()
    pop, hof = optimizador.optimizar_nsga2(
        ngen=300, #100 #800
        pop_size=64, #32,
        seed=42,
        parallel=True
    )
    print(f"Temps: {time.time()-t0:.1f}s") #99s 1 iteracion #903s 20 iter
    # 4.5h 64 pop, 1000 gen
    t0 = time.time()
    # Convertir resultats a DataFrame
    df_pareto_nsga = optimizador.hof_to_dataframe(hof)
    print(f"\nSolucions Pareto trobades: {len(df_pareto_nsga)}")
    print(df_pareto_nsga.head())
    print(f"Temps: {time.time()-t0:.1f}s")
    

      # Guardar ràpid (només el necessari)
    checkpoint = {
        'pop': [(list(ind), ind.fitness.values) for ind in pop],
        'hof': [(list(ind), ind.fitness.values) for ind in hof]
    }
    with open('checkpoint_nsga.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)
# -----------------------------------------------------------------------------    
    # df_pareto_nsga.to_parquet('df_nsga2_500g_10vE1_2030.parquet')
    # df_pareto_nsga.to_parquet('df_nsga_300g_11vE1b_2050.parquet')

    # df_nsga = pd.read_parquet('df_nsga_3k_9v.parquet')  # ✅ Funció "read_parquet"

# front_nsga[front_nsga.min_level > 50].midpoint_estimation.min()
# front_nsga[front_nsga.mean_level > 80].midpoint_estimation.min()
# front_nsga[front_nsga.min_level > 50].overflow_threshold.min()
# front_nsga[front_nsga.mean_level > 80].overflow_threshold.min()
# front_nsga[front_nsga.mean_level > 80].L_prealerta.min()
# front_nsga[front_nsga.mean_level > 80].L_alerta.min()

# front_nsga[front_nsga.mean_level > 80].L_prealerta.mean()

# test = pd.read_parquet('df_nsga2_500g_10vE1b_2030.parquet')


# front_nsga = pd.read_parquet('df_nsga_100g_7vE1_2030.parquet')
front_nsga = pd.read_parquet('df_nsga_500g_10vE1_2030.parquet')
# front_nsga = pd.read_parquet('df_nsga2_500g_10vE1b_2030.parquet')
# front_nsga = pd.read_parquet('df_nsga_500g_10vE1_2050.parquet')
#%%
#--------------------------------------
# # CARREGAR I CONTINUAR
# with open('checkpoint_nsga.pkl', 'rb') as f:
#     checkpoint = pickle.load(f)

# # Reconstruir població (els resultats es recalcularan si cal)
# pop_loaded = []
# for values, fitness in checkpoint['pop']:
#     ind = creator.Individual(values)
#     ind.fitness.values = fitness
#     pop_loaded.append(ind)
#--------------------------------------
# Continuar des d'on vas acabar
pop2, hof2 = optimizador.optimizar_nsga2(
    ngen=100,  # 100 generacions més
    pop_size=64,
    seed=43,  # Canviar seed per varietat
    pop_inicial=pop,  # ← Passar la població anterior
    hof_inicial=hof   # ← Passar el hof anterior
)
df_pareto_nsga = optimizador.hof_to_dataframe(hof2)
df_pareto_nsga.to_parquet('df_nsga_250g_11vE1_2050.parquet')


# Convertir resultats a DataFrame
extended_pareto = optimizador.hof_to_dataframe(hof2)
print(f"\nSolucions Pareto trobades: {len(df_pareto_nsga)}")
print(estended_paretoa.head())
print(f"Temps: {time.time()-t0:.1f}s")

def merge_pareto_dataframes(df1, df2):
    """Fusiona dos DataFrames de Pareto eliminant dominats."""
    # Concatenar
    df_all = pd.concat([df1, df2], ignore_index=True)
    
    # Columnes objectiu (tots a minimitzar)
    obj_cols = ['obj_gapsq', 'obj_msqdev', 'obj_costs']
    
    # Filtrar dominats
    is_pareto = []
    values = df_all[obj_cols].values
    
    for i in range(len(values)):
        dominated = False
        for j in range(len(values)):
            if i != j:
                if all(values[j] <= values[i]) and any(values[j] < values[i]):
                    dominated = True
                    break
        is_pareto.append(not dominated)
    
    df_pareto = df_all[is_pareto].drop_duplicates(subset=obj_cols).reset_index(drop=True)
    print(f"Merge: {len(df1)} + {len(df2)} → {len(df_pareto)} solucions Pareto")
    
    return df_pareto

# Ús
df_merged = merge_pareto_dataframes(df_pareto_nsga, extended_pareto)

df_merged.to_parquet('df_nsga2_500g_10vE1b_2030.parquet')
#%%
import matplotlib.pyplot as plt


# Filtrar per residu tèrmic (ajusta el llindar)
llindar_residu = 10  # TWh/y
df_filtrat = front_nsga[front_nsga['obj_gasimports'] < llindar_residu]

# Gràfic
fig, ax = plt.subplots(figsize=(10, 7))

scatter = ax.scatter(
    df_filtrat['potencia_solar'] / 1000,      # Convertir a GW
    df_filtrat['potencia_eolica'] / 1000,     # Convertir a GW
    c=df_filtrat['potencia_baterias'] / 1000, # Color = bateries en GWh
    cmap='viridis',
    s=50,
    alpha=0.7,
    edgecolors='k',
    linewidths=0.3
)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Bateries (GWh)', fontsize=11)

ax.set_xlabel('Potència Solar (GW)', fontsize=11)
ax.set_ylabel('Potència Eòlica (GW)', fontsize=11)
ax.set_title(f'Combinacions renovables amb residu tèrmic < {llindar_residu} TWh/y (n={len(df_filtrat)})')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()