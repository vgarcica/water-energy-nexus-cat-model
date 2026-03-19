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
                         increments_a_llindars
                         )

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

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


#%% - BENCHMARK
# 292 amb minimize -> 160 ms amb robust
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
        'df_demanda': datos.demanda,
        'df_nucleares_base': datos.nuclears_base,
        'df_cogeneracion': datos.cogeneracion_h,
        'df_solar': datos.solar_h,
        'df_eolica': datos.eolica_h,
        'df_autoconsum': datos.autoconsum_hourly,
        'df_potencia_historica': datos.potencia,
        'df_capacidad_internes': datos.df_pct_int_h,
        'df_capacidad_ebre': datos.df_pct_ebre_h,
        'energia_turbinada_mensual_internes': datos.energia_turbinada_mensual_internes,
        'energia_turbinada_mensual_ebre': datos.energia_turbinada_mensual_ebre,
        'potencia_max_hidro': {'ebro': potencia_max_hidraulica_ebro, 'int': potencia_max_hidraulica_int},
        'sensibilidad_hidro': {'ebro': sensibility_ebro, 'int': sensibility_int},
        'capacidad_max_hidro': {'ebro': max_capacity_ebro, 'int': max_capacity_int},
        'umbral_overflow_pct': {'ebro': 75, 'int': 75},
        'nucleares_activas': [True, True, True],
        'baterias_config': [534, 5340, 0.8, 0],
        'max_salto_hidro_pct': 10.0,
    }
    
    # -------------------------------------------------------------------------
    # BENCHMARK: VERSIÓ OPTIMITZADA (AMB PRECOMPUTACIÓ)
    # -------------------------------------------------------------------------
    print(f"\n[2/2] Executant {n_escenarios} escenaris AMB precomputació...")
    
    # Pas 1: Precomputar (una sola vegada)
    t_start_precompute = time.time()
    
    precomputed = precomputar_dades_base(
        df_demanda=datos.demanda,
        df_solar=datos.solar_h,
        df_eolica=datos.eolica_h,
        df_cogeneracion=datos.cogeneracion_h,
        df_autoconsum=datos.autoconsum_hourly,
        df_potencia_historica=datos.potencia,
        df_capacidad_internes=datos.df_pct_int_h,
        df_capacidad_ebre=datos.df_pct_ebre_h,
        potencia_cogeneracio_max=datos.potencia.Cogeneració.iloc[-1],
        verbose=True
    )
    
    t_precompute = time.time() - t_start_precompute
    
    # Pas 2: Executar escenaris
    t_start_optimized = time.time()
    
    for i, pesos in enumerate(escenarios_test):
        results, metrics = generar_escenario_sintetico(
            **base_config,
            pesos=pesos,
            precomputed=precomputed,  # Amb precomputació
            # hydro_min_for_level=hydro_min_for_level,
            optimizador_hidro = 'robusto'            
        )
    
    t_optimized = time.time() - t_start_optimized
    t_per_escenari_optimized = t_optimized / n_escenarios
    t_total_optimized = t_precompute + t_optimized
    
    print(f"\n      Temps precomputació: {t_precompute:.2f} s (una sola vegada)")
    print(f"      Temps escenaris: {t_optimized:.2f} s")
    print(f"      Temps total: {t_total_optimized:.2f} s")
    print(f"      Temps per escenari: {t_per_escenari_optimized:.3f} s")    



#%%
#340 +- 8 ms
# 235 +- 3 ms
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
#%%

# Configuració comuna
base_config = {
    'df_demanda': datos.demanda,
    'df_nucleares_base': datos.nuclears_base,
    'df_cogeneracion': datos.cogeneracion_h,
    'df_solar': datos.solar_h,
    'df_eolica': datos.eolica_h,
    'df_autoconsum': datos.autoconsum_hourly,
    'df_potencia_historica': datos.potencia,
    'df_capacidad_internes': datos.df_pct_int_h,
    'df_capacidad_ebre': datos.df_pct_ebre_h,
    'energia_turbinada_mensual_internes': datos.energia_turbinada_mensual_internes,
    'energia_turbinada_mensual_ebre': datos.energia_turbinada_mensual_ebre,
    'potencia_max_hidro': {'ebro': potencia_max_hidraulica_ebro, 'int': potencia_max_hidraulica_int},
    'sensibilidad_hidro': {'ebro': sensibility_ebro, 'int': sensibility_int},
    'capacidad_max_hidro': {'ebro': max_capacity_ebro, 'int': max_capacity_int},
    'umbral_overflow_pct': {'ebro': 75, 'int': 75},
    'nucleares_activas': [True, True, True], #2024 a 2030
    # 'nucleares_activas': [False, False, False], #2040
    'baterias_config': [534, 5340, 0.8, 0], #2024
    # 'baterias_config': [534, 2136, 0.8, 0],
    # 'baterias_config': [4000, 16000, 0.8, 0], #2040
    # 'baterias_config': [2234, 2234*4, 0.8, 0], #2030
    'max_salto_hidro_pct': 5 #10.0,
}

pesos = {'solar': 0.9369, 'wind': 0.9893, 'dem': 1, 'cog': 1.0, 'auto': 0.8607} #2024
# pesos = {'solar': 2, 'wind': 1.2, 'dem': 1, 'cog': 1.0, 'auto': 0.8607}
# pesos = {'solar': 8, 'wind': 1.2, 'dem': 1, 'cog': 1.0, 'auto': 0.8607}
# pesos = {'solar': 3, 'wind': 8, 'dem': 1.5, 'cog': 1.0, 'auto': 0.8607}
# pesos = {'solar': 39, 'wind': 13.1, 'dem': 1.86, 'cog': 0.12, 'auto': 5.3} #2040

# pesos = {'solar': 39, 'wind': 13.1, 'dem': 2, 'cog': 0.12, 'auto': 5.3} #2040

# pesos = {'solar': 12.87, 'wind': 4.43, 'dem': 1.33, 'cog': 0.63, 'auto': 1.58} #2030

cf_obj_final = factor_capacitat_eolica(multiple=pesos['wind'])



results, metrics_pct, metrics_GWh, metrics_pct2 = generar_escenario_sintetico(
    **base_config,
    pesos=pesos,
    CF_eolica_obj=cf_obj_final,
    precomputed=precomputed,  # Amb precomputació
    # hydro_min_for_level=hydro_min_for_level,
    optimizador_hidro = 'robusto',
    # start_date = '2024-01-01',
    # end_date = '2024-12-31'
    )


# ((datos.df_pct_int_h.squeeze() - precomputed['estalvi']/max_capacity_int)*100).dropna().plot()
# results.Hydro_Level_int.plot()
# results.Hydro_Level_ebro.plot()

surpluses = results.Nuclear + results.Solar_w + results.Eòlica_w + results.Cogen_w + results.Hidràulica + results.Bateries + results['Gas+Imports'] - results.Càrrega
surpluses = results['Gas+Imports'] - results.gap
surpluses[surpluses < 0] = 0
hydro_base_level = ((datos.df_pct_int_h.squeeze() - precomputed['estalvi']/max_capacity_int)*100).dropna()


min_run_hours = 6
max_desalation = 32
midpoint_estimation = 75 # parámetro del sigmoide de desalación
overflow_threshold_pct = 90
seasonal_phase_months = 0.0 # Maximo en enero
seasonal_desal_amplitude = 0.0 # Sin variación estacional
desal_sensibility = 3500

#%% 
# 1.26 s ± 31 ms
# 1.18 s ± 10.3 ms
# 865 ms ± 1.28 ms
# 1.5 s con regeneración y calculo de rampas para regen y desal
# 521 ms (optimizando la actualización de nivel)
# 584 ms Añado derivada
%%time

    level_final, regen_final, desal_final, savings_final, savings_agro_final, savings_urba_final, extra_hydro_final, new_surpluses, spillage_final = simulate_full_water_management(
        surpluses=surpluses,
        level_base=hydro_base_level,
        thermal_generation=results['Gas+Imports'],
        base_hydro_generation=results['hydro_int'],
        max_capacity_int=max_capacity_int,
        # consumo_base_diario_hm3= None, #2.05, #consumo_base_diario_hm3,
        max_hydro_capacity_mw=potencia_max_hidraulica_int,
        overflow_threshold_pct=overflow_threshold_pct,
        sensitivity_mwh_per_percent=max_capacity_int * 0.01 * sensibility_int,
        # Parámetros de desalación
        min_run_hours=min_run_hours,
        max_desal_mw=max_desalation,
        desal_min_pct = 0.18,
        finestra_hores = 168,
        llindar_activacio_regen_max=2,
        llindar_activacio_desal_estacional=6,
        save_hm3_per_mwh = 1/desal_sensibility,
        midpoint=midpoint_estimation,
        # desal_derivada_amplitud= 0, #0.5,
        seasonal_phase_months = seasonal_phase_months,
        seasonal_amplitude = seasonal_desal_amplitude
    )

# new_hydro = results['Hidràulica'] + extra_hydro_final
# new_gas = results['Gas+Imports'] - extra_hydro_final


#%%
def procesar_escenario(
    # series de datos
    df_demanda,
    df_nuclear,
    df_cogeneracion,      
    df_solar,
    df_eolica,
    df_autoconsum,
    df_potencia,
    df_niveles_int,
    df_niveles_ebro,
    df_energia_turbinada_mensual_internes,
    df_energia_turbinada_mensual_ebre,
    df_nivel_si, #niveles sin intervención, ni desalación ni restricciones
    
    # parámetros fijos
    max_capacity_int,
    max_capacity_ebro,
    potencia_max_int,
    potencia_max_ebro,
    consumo_base_diario_hm3,
    sensibility_int,
    sensibility_ebro,
    
    # variables de decisión (ajustables)
    nucleares_activas = [True,True,True],
    potencia_solar = None,  # Si es None, usa valores actuales
    potencia_eolica = None,
    potencia_cogeneracion = None,
    potencia_baterias = 500,
    min_run_hours = 6,
    max_desalation = 32,
    midpoint_estimation = 75, # parámetro del sigmoide de desalación
    overflow_threshold_pct = 90,
    seasonal_phase_months = 0.0, # Maximo en enero
    seasonal_desal_amplitude = 0.0 # Sin variación estacional
    ):
    
    # Calcular potencias actuales si no se especifican
    if potencia_solar is None:
        potencia_solar = df_potencia.Fotovoltaica.iloc[-1] + df_potencia.Termosolar.iloc[-1]
    if potencia_eolica is None:
        potencia_eolica = df_potencia.Eòlica.iloc[-1]
    if potencia_cogeneracion is None:
        potencia_cogeneracion = df_potencia.Cogeneració.iloc[-1]

    # Calcular pesos relativos
    peso_solar = potencia_solar / (df_potencia.Fotovoltaica.iloc[-1] + df_potencia.Termosolar.iloc[-1])
    peso_wind = potencia_eolica / df_potencia.Eòlica.iloc[-1]
    peso_cogen = potencia_cogeneracion / df_potencia.Cogeneració.iloc[-1]
    
    # Configuración de baterías
    bat = [potencia_baterias, potencia_baterias*4, 0.8, 0]

    escenario = {
        'nucleares_activas': nucleares_activas,
        'pesos': {'solar': peso_solar, 'wind': peso_wind, 'dem': 1, 'cog': peso_cogen, 'auto': 1},
        'baterias_config': bat,
        'max_salto_hidro_pct': 5.0,
        'optimizador_hidro': 'rapido'
    }

    # Calcular costes (corregido)
    coste_Solar = (potencia_solar - df_potencia.Fotovoltaica.iloc[-1] - df_potencia.Termosolar.iloc[-1]) * 775000
    coste_Wind = (potencia_eolica - df_potencia.Eòlica.iloc[-1]) * 1400000
    coste_Bat = (bat[1] - 2000) * 325000
    coste_Desal = max_desalation * 16000000

    # Configuración de desalación
    # w_desal = max_desalation / 30 #30 MW es el valor actual instalado
    # Si tienes la función find_midpoint_for_target_factor, úsala, sino usa el valor directo
    # midpoint_estimation = find_midpoint_for_target_factor(75, 0.5 / w_desal)
    # Si no existe la función, usa el valor por defecto
    # midpoint_estimation = 75

    # Generar escenario sintético
    results, energy_metrics = generar_escenario_sintetico(
        # Parámetros del escenario
        **escenario,
        
        # Datos base
        df_demanda=df_demanda,
        df_nucleares_base=df_nuclear,
        df_cogeneracion=df_cogeneracion,
        df_solar=df_solar,
        df_eolica=df_eolica,
        df_autoconsum=df_autoconsum,        
        df_potencia_historica=df_potencia,
        df_capacidad_internes=df_niveles_int,
        df_capacidad_ebre=df_niveles_ebro,
        energia_turbinada_mensual_internes=df_energia_turbinada_mensual_internes,
        energia_turbinada_mensual_ebre=df_energia_turbinada_mensual_ebre,
        
        # Parámetros físicos
        potencia_max_hidro={'ebro': potencia_max_ebro, 'int': potencia_max_int},
        sensibilidad_hidro={'ebro': sensibility_ebro, 'int': sensibility_int},  # Corregido
        capacidad_max_hidro={'ebro': max_capacity_ebro, 'int': max_capacity_int},
        umbral_overflow_pct={'ebro': 75.0, 'int': 75.0}
    )
    
    # Calcular excedentes
    surpluses = results.gap.copy()
    surpluses[surpluses > 0] = 0
    surpluses *= -1
    
    # Simulación de gestión hídrica
    level_final, desal_final, savings_final, extra_hydro_final, new_surpluses = simulate_full_water_management(
        surpluses=surpluses,
        level_base=df_nivel_si,
        thermal_generation=results['Gas+Imports'],
        base_hydro_generation=results['hydro_int'],
        max_capacity_int=max_capacity_int,
        consumo_base_diario_hm3=consumo_base_diario_hm3,
        max_hydro_capacity_mw=potencia_max_int,
        overflow_threshold_pct=overflow_threshold_pct,
        sensitivity_mwh_per_percent=max_capacity_int * 0.01 * sensibility_int,
        # Parámetros de desalación
        min_run_hours=min_run_hours,
        max_desal_mw=max_desalation,
        midpoint=midpoint_estimation,
        seasonal_phase_months = seasonal_phase_months,
        seasonal_amplitude = seasonal_desal_amplitude
    )
    
    # Convertir desalación a hm3
    desal_final_hm3 = desal_final / desal_sensibility
    
    # Calcular excedentes descontando desalación
    # surpluses_desal = surpluses - desal_final
    
    # Calcular amplitud estacional
    seasonal_capacity = seasonal_decompose(level_final.resample('ME').last(), model='additive', period=12).seasonal
    monthly_seasonal = seasonal_capacity.groupby(seasonal_capacity.index.month).mean()
    seasonal_amplitude = monthly_seasonal.max() - monthly_seasonal.min()

    # Recalcular los niveles de los embalses y de Gas+Imports
    results['hydro_int'] += extra_hydro_final
    results['Hidráulica'] = results.hydro_ebro + results.hydro_int
    results['Gas+Imports'] -= extra_hydro_final
    results['Hydro_Level_int'] = level_final
    results['surpluses'] = new_surpluses
    results['savings'] = savings_final
   
    
    # Métricas hídricas (corregido para usar df_niveles_int)
    hydro_metrics = {
        # 'Restricciones históricas (días)': df_niveles_int[(100*df_niveles_int.Nivel) < 40]['2015-06-01':'2024-12-31'].count().iloc[0]*7,
        'Restricciones históricas (días)': df_niveles_int[(100*df_niveles_int) < 40].count()*7,
        'Restricciones escenario (días)': int(level_final[level_final < 40].count()/24),
        'Llenado mínimo (%)': level_final.min(),
        'Llenado promedio (%)': level_final.mean(),
        'Variación estacional (%)': seasonal_amplitude,
        'Desalación. Factor de capacidad (%)': int(100 * desal_final.mean() / max_desalation)
    }
    
    # Retornar resultados completos
    return {
        'energy_data': results,
        'energy_metrics': energy_metrics,
        'level_final': level_final,
        'desal_final': desal_final,
        'desal_final_hm3': desal_final_hm3,
        'hydro_metrics': hydro_metrics,
        'surpluses_afterdesal': new_surpluses.sum(),
        'capacity_factor': int(100 * desal_final.mean() / max_desalation),
        'savings_final': savings_final,
        'costes': {
            'solar': coste_Solar,
            'wind': coste_Wind,
            'baterias': coste_Bat,
            'desalacion': coste_Desal,
            'total': coste_Solar+coste_Wind+coste_Bat+coste_Desal
        }
    }




#%%

sample = results[['Demanda','Nuclear','Cogen_w','Eòlica_w','Solar_w', 'Hidràulica', 'Bateries', 'Gas+Imports', 'Càrrega']]
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
dia = '2023-07-01'
dia = '2020-04-01'
# dia = '2020-05-01'
# dia = '2019-12-05'
inicio = pd.to_datetime(dia)
fin = inicio + pd.Timedelta(hours=23)
fin = inicio + pd.Timedelta(hours=24*7)
# fin = inicio + pd.Timedelta(hours=24*30)

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
sample[inicio:fin]['Càrrega'].plot(ax=ax, color='black', style='--', linewidth=2)
(sample[inicio:fin]['Demanda']).plot(ax=ax, color='grey', linewidth=2, label='Demanda')


ax.legend(loc='upper left')
plt.show()


#%%
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
    verbose=True
)

# CREAR LOOKUP TABLE (1 sola vegada, abans de Parallel)
nivells = np.arange(0, 100.1, 0.1)  # Resolució 0.1%
hydro_min_lookup = np.array([hydro_min_for_level(n) for n in nivells])

# Afegir a precomputed
precomputed['hydro_min_lookup'] = hydro_min_lookup

hydro_base_level = ((datos.df_pct_int_h.squeeze() - precomputed['estalvi']/max_capacity_int)*100).dropna()
min_run_hours = 6
max_desalation = 32
midpoint_estimation = 75 # parámetro del sigmoide de desalación
overflow_threshold_pct = 90
seasonal_phase_months = 0.0 # Maximo en enero
seasonal_desal_amplitude = 0.0 # Sin variación estacional
desal_sensibility = 3500


seasonal_capacity = seasonal_decompose(
    100*datos.df_pct_int_h.squeeze().resample('ME').last(), 
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
    n_year = round(len(level_final) / (24 * 365))
    costes = {
        'solar': (potencia_solar - potencia_solar_ref) * 640_000 * n_year / 35, 
        'wind': (potencia_eolica - potencia_eolica_ref) * 963_000 * n_year / 25, 
        'baterias': (bat_config[0] - 532) * 712_000 * n_year / 15,
        'desalacion': (max_desalation - max_desalation_ref) * 15_500_000 * n_year / 30, # from 80hm3/y * 6200000€/(hm3/y) / 32MW [€/MW]
        'regeneracion': (max_regen - 0.173)*365 * 4_130_000 * n_year / 30,
        'restriccions': 2_000_000 * savings_agro_final.sum() + 35_000_000 * savings_ind_final.sum()
        + 1_600_000 * savings_urba_final.sum(),
        'desal_opex': 556_000 * desal_en_deficit_hm3 + 206_000 * desal_en_surplus_hm3,
        'regen_opex': 190_000 * regen_en_deficit_hm3 + 70_000 * regen_en_surplus_hm3,
        # 'estalvi_gas': (results['Gas+Imports'].sum() - 87380181.0257) * 100,
        'solar_opex': (potencia_solar - potencia_solar_ref) * 19000,
        'wind_opex': (potencia_eolica - potencia_eolica_ref) * 27000,
        'bate_opex': (bat_config[0] - 532) * 18000
    }
    
    costes['total'] = sum(costes.values())
    
    costes['opex'] = costes['restriccions'] + costes['desal_opex'] + costes['regen_opex'] + costes['solar_opex'] + costes['wind_opex'] + costes['bate_opex']
    
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
%%time

# =============================================================================
# PARÀMETRES OPCIONALS DE POLÍTICA (comentar per usar valors per defecte)
# =============================================================================

# umbrales_sequia = {
#     'Emergencia_3': 5.5, 
#     'Emergencia_2': 11.0, 
#     'Emergencia': 16.0,
#     'Excepcionalitat': 25.0, 
#     'Alerta': 40.0, 
#     'Prealerta': 60.0
# }

# restricciones_sectoriales = {
#     'Urba':        {'Normalitat':0, 'Prealerta':0.025, 'Alerta':0.05, 'Excepcionalitat':0.075, 'Emergencia':0.10, 'Emergencia_2':0.12, 'Emergencia_3':0.14},
#     'Regadiu':     {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.25, 'Excepcionalitat':0.40, 'Emergencia':0.80, 'Emergencia_2':0.80, 'Emergencia_3':0.80},
#     'Ramaderia':   {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.10, 'Excepcionalitat':0.30, 'Emergencia':0.50, 'Emergencia_2':0.50, 'Emergencia_3':0.50},
#     'Ind_Bens':    {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.05, 'Excepcionalitat':0.15, 'Emergencia':0.25, 'Emergencia_2':0.25, 'Emergencia_3':0.25},
#     'Ind_Turisme': {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.05, 'Excepcionalitat':0.15, 'Emergencia':0.25, 'Emergencia_2':0.25, 'Emergencia_3':0.25}
# }

# Consum base mensual [hm³/dia] - de gener a desembre
consumo_base_diario_estacional = np.array([
    1.37, 1.37, 1.42, 1.74, 2.32, 3.03, 
    3.51, 3.24, 2.36, 1.48, 1.37, 1.37
])

# =============================================================================
# CRIDA A PROCESAR_ESCENARIO
# =============================================================================

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
    # hydro_min_for_level=hydro_min_for_level,  # NOU: Afegir aquesta variable
    
    # --- Paràmetres hídrics ---
    consumo_base_diario_estacional_hm3=consumo_base_diario_estacional,  # NOU
    save_hm3_per_mwh=1/desal_sensibility,  # NOU
    
    # --- Dades precomputades (opcional, per accelerar múltiples escenaris) ---
    precomputed=precomputed,  # O passar precomputed si l'has calculat abans
    
    # --- Variables de decisió: Sistema elèctric ---
    nucleares_activas=[True, True, True],
    potencia_solar=11225.932, #1158.432, #7180.8,
    potencia_eolica=16234, #11249.2, #6234.2,
    potencia_cogeneracion=943.503, #470.2,
    potencia_baterias=534, #2234,
    duracion_horas = 10,
    potencia_autoconsumo=1188.6267,
    demanda_electrica=1,
    CF_eolica_obj = None,
    usar_CF_automatic = True,
    
    # --- Variables de decisió: Gestió hídrica ---
    overflow_threshold_pct=75,
    max_regen = 0.173,
    regen_base_pct=0.5,
    llindar_activacio_regen_max = 2,
    llindar_activacio_desal_max = 5,
    min_run_hours=6,
    max_desalation=32,
    midpoint_estimation=75,
    seasonal_phase_months=0.0,
    seasonal_desal_amplitude=0.0,
    # llindar activacio estacionalitat
    k_deriv = 0,
    trend_time_window = 24*30


    
    # # --- Variables de decisió: Política de sequera (comentar per usar defecte) ---
    # umbrales_sequia=umbrales_sequia,
    # restricciones_sectoriales=restricciones_sectoriales
)

# =============================================================================
# VISUALITZACIÓ RÀPIDA DE RESULTATS
# =============================================================================

print("=== MÈTRIQUES ENERGÈTIQUES ===")
for k, v in results['energy_metrics_pct'].items():
    print(f"  {k}: {v}")

print("\n=== MÈTRIQUES HÍDRIQUES ===")
for k, v in results['hydro_metrics'].items():
    print(f"  {k}: {v}")
    
print("\n=== MÈTRIQUES HÍDRIQUES HISTÒRIQUES===")
for k, v in historic_hydro_metrics.items():
    print(f"  {k}: {v}")

print(f"\n=== COSTOS ===")
for k, v in results['costes'].items():
    print(f"  {k}: {v:,.0f} €")
    
    

#%%
# --- 1. PREPARACIÓ DE DADES (Adapta-ho a les teves variables) ---

# Suma de la històrica (Conques Internes + Ebre)
# Assegura't que l'índex és datetime
hist_series = (datos.energia_turbinada_mensual_internes + datos.energia_turbinada_mensual_ebre)/1000
# hist_series = datos.energia_turbinada_mensual_ebre
# hist_series = datos.energia_turbinada_mensual_internes

# Sèrie sintètica de l'escenari
sim_series = results['energy_data'].Hidràulica.resample('ME').sum()/1000
# sim_series = results['energy_data'].hydro_ebro.resample('ME').sum()
# sim_series = results['energy_data'].hydro_int.resample('ME').sum()

# Creem un DataFrame per facilitar l'anàlisi
df = pd.DataFrame({
    'Historic': hist_series,
    'Simulat': sim_series
})

# Netegem possibles NaNs i alineem índexs
df = df.dropna()

# Calculem la diferència (Positiu = El simulador genera MÉS que l'històric)
df['Diferencia'] = df['Simulat'] - df['Historic']

# Afegim columna de mes per a l'agrupació
df['Mes'] = df.index.month

# --- 2. GENERACIÓ DELS GRÀFICS ---

fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

# --- GRÀFIC A: Perfil Estacional Mitjà (El "Shift" Estructural) ---
# Agrupem per mes i fem la mitjana per veure el comportament típic
seasonality = df.groupby('Mes')[['Historic', 'Simulat']].mean()

# Pintem les línies
axes[0].plot(seasonality.index, seasonality['Historic'], 'o--', label='Històric (Real)', color='grey', alpha=0.7)
axes[0].plot(seasonality.index, seasonality['Simulat'], 'o-', label='Escenari (Simulat)', color='blue', linewidth=2)

# Omplim l'àrea entre corbes per destacar el canvi
axes[0].fill_between(seasonality.index, seasonality['Historic'], seasonality['Simulat'], 
                     where=(seasonality['Simulat'] > seasonality['Historic']), 
                     interpolate=True, color='green', alpha=0.2, label='Increment Generació')
axes[0].fill_between(seasonality.index, seasonality['Historic'], seasonality['Simulat'], 
                     where=(seasonality['Simulat'] <= seasonality['Historic']), 
                     interpolate=True, color='red', alpha=0.2, label='Reducció Generació')

axes[0].set_title('Perfil Estacional Mitjà: Canvi en l\'Estratègia de Turbinació', fontsize=14)
axes[0].set_ylabel('Energia Mensual [GWh]', fontsize=12)
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(['Gen', 'Feb', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Oct', 'Nov', 'Des'])
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

# --- GRÀFIC B: Diferències en la Sèrie Temporal (El "Shift" Dinàmic) ---
# Mostra mes a mes on s'ha guanyat o perdut energia
colors = ['green' if x > 0 else 'red' for x in df['Diferencia']]

axes[1].bar(df.index, df['Diferencia'], color=colors, width=20, alpha=0.7) # width en dies aprox

# Línia de referència 0
axes[1].axhline(0, color='black', linewidth=1)

axes[1].set_title('Diferència Neta Mensual (Simulat - Històric)', fontsize=14)
axes[1].set_ylabel('Delta Energia (Simulat - Històric)', fontsize=12)
axes[1].text(df.index[0], df['Diferencia'].max(), "Verd: Simulador genera MÉS\nVermell: Simulador estalvia aigua", 
             verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.8))

axes[1].grid(True, linestyle=':', alpha=0.6)

plt.show()
    
#%%
import seaborn as sn
sn.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

# level_old.plot()
# results['level_final'].plot()
# plt.axhline(y=100, color='red', linestyle='--', alpha=0.7)

# Crear figura con tamaño apropiado para tesis
fig, ax = plt.subplots(figsize=(10, 6))

# Trazar las dos series con estilos diferenciados
level_old.plot(ax=ax, linewidth=1.8, color='#2E5A87', 
               label='Sense abocaments', alpha=0.85)
results['level_final'].plot(ax=ax, linewidth=1.8, color='#D9534F', 
                            label='Amb abocaments', alpha=0.85)

# Línea del 100% de capacidad (añadir etiqueta)
ax.axhline(y=100, color='#5D4037', linestyle='--', linewidth=1.5, 
           alpha=0.7, label='Capacitat màxima (100%)')

# Configurar ejes y título
ax.set_xlabel('Temps', fontsize=13, fontweight='medium')
ax.set_ylabel('Nivell dels embassaments (%)', fontsize=13, fontweight='medium')
ax.set_title('Comparació de nivells amb i sense gestió d\'abocaments', 
             fontsize=15, fontweight='bold', pad=15)

# Configurar límites del eje Y (evitar espacio excesivo)
ax.set_ylim(0, 105)

# Cuadrícula sutil
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Leyenda con ubicación óptima
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
          ncol=3, frameon=True, fancybox=False, shadow=False, 
          framealpha=0.9, edgecolor='black')

# Ajustar márgenes
plt.tight_layout()

# Mostrar gráfico
plt.show()
    
#%%
#provar primer amb les variables típiques, després afegir llindars i després restriccions sectorials
# estimar costos:
    #CAPEX: Fotovoltaica, Eòlica, Dessalació, Regeneració
    #OPEX: Dessalació, Regeneració, Restriccions


# ABANS de Parallel, crear lookup table
# nivells = np.arange(0, 101, 1)  # 0% a 100%
# hydro_min_lookup = np.array([hydro_min_for_level(n) for n in nivells])
# Passar a procesar_escenario com a array en lloc de funció
# hydro_min_for_level_array = hydro_min_lookup

nivells = np.arange(0, 100.1, 0.1)  # Resolució 0.1%
hydro_min_lookup = np.array([hydro_min_for_level(n) for n in nivells])
precomputed['hydro_min_lookup'] = hydro_min_lookup

# Consum base mensual [hm³/dia] - de gener a desembre
consumo_base_diario_estacional = np.array([
    1.37, 1.37, 1.42, 1.74, 2.32, 3.03, 
    3.51, 3.24, 2.36, 1.48, 1.37, 1.37
])

# === 1. Define tu espacio de búsqueda ===
potencia_solar = np.arange(400,20100,100)
potencia_eolica = np.arange(1400,10100,100)
potencia_baterias = np.arange(500,5100,100)
min_run_hours = np.arange(6, 13, 1)
max_desalation = np.arange(32, 161, 1)
midpoint_estimation = np.arange(10,91,1)
overflow_threshold_pct = np.arange(40,91,1)
seasonal_phase_months = np.arange(0, 11.9, 0.1)
seasonal_amplitude = np.arange(0,1.1,0.1)
regen_base_pct = np.arange(0.35,0.65,0.05)
llindar_activacio_regen_max = np.arange(1,4,1)


# Rangs per cada increment
x1_range = np.arange(10, 21, 2)   # Base Emergència: 10-30%
x2_range = np.arange(10, 21, 2)    # Gap Excepcionalitat: 5-20%
x3_range = np.arange(10, 21, 2)    # Gap Alerta: 5-25%
x4_range = np.arange(10, 21, 2)    # Gap Prealerta: 5-20%

def generar_escenaris_llindars(n_samples, rng, max_prealerta=90):
    """Genera combinacions vàlides de llindars."""
    escenaris = []
    attempts = 0
    
    while len(escenaris) < n_samples and attempts < n_samples * 10:
        x1 = rng.choice(x1_range)
        x2 = rng.choice(x2_range)
        x3 = rng.choice(x3_range)
        x4 = rng.choice(x4_range)
        
        # Restricció global: suma ≤ max_prealerta
        if x1 + x2 + x3 + x4 <= max_prealerta:
            escenaris.append((x1, x2, x3, x4))
        
        attempts += 1
    
    return escenaris

# Generar
rng = np.random.default_rng(42)
llindars_samples = generar_escenaris_llindars(100, rng)

n_samples = 100  # número de escenarios deseado
rng = np.random.default_rng(42)  # semilla reproducible

for _ in range(n_samples * 10):  # Més intents per assegurar n_samples vàlids
    x1 = int(rng.choice(x1_range))
    x2 = int(rng.choice(x2_range))
    x3 = int(rng.choice(x3_range))
    x4 = int(rng.choice(x4_range))
    
    # Restricció global
    if x1 + x2 + x3 + x4 > 90:
        continue

    scenarios_params = [
        (
            int(rng.choice(potencia_solar)),
            int(rng.choice(potencia_eolica)),
            int(rng.choice(potencia_baterias)),
            int(rng.choice(min_run_hours)),
            int(rng.choice(max_desalation)),
            int(rng.choice(midpoint_estimation)),
            int(rng.choice(overflow_threshold_pct)),
            rng.choice(seasonal_phase_months),
            rng.choice(seasonal_amplitude),
            rng.choice(regen_base_pct),
            int(rng.choice(llindar_activacio_regen_max)),
            x1, x2, x3, x4
        )
        for _ in range(n_samples)
    ]

    if len(scenarios_params) >= n_samples:
        break

# === 3. Define la función de evaluación ===
def run_case(p_solar, p_eolica, p_baterias, run_h, desal, midpoint, overflow, phase, amplitude, regen_base, llindar_regen, x1, x2, x3, x4):
    umbrales = increments_a_llindars(x1, x2, x3, x4)
    
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
        # hydro_min_for_level=hydro_min_for_level,  # NOU
        
        # --- Paràmetres hídrics ---
        consumo_base_diario_estacional_hm3=consumo_base_diario_estacional,  # NOU
        save_hm3_per_mwh=1/desal_sensibility,  # NOU 

        # --- Dades precomputades (opcional, per accelerar múltiples escenaris) ---
        precomputed=precomputed,
        
        # --- Variables de decisió: Sistema elèctric ---
        nucleares_activas=[True, True, True],
        potencia_solar=p_solar,
        potencia_eolica=p_eolica,
        potencia_cogeneracion=943.503, #470.2,
        potencia_baterias=p_baterias, #2234,
        duracion_horas = 4,
        potencia_autoconsumo=1188.6267,
        demanda_electrica=1,
        CF_eolica_obj = None,
        usar_CF_automatic = True,
        
        # --- Variables de decisió: Gestió hídrica ---
        umbrales_sequia=umbrales,
        overflow_threshold_pct=overflow,
        regen_base_pct=regen_base,
        llindar_activacio_regen_max = llindar_regen,
        min_run_hours=run_h,
        max_desalation=desal,
        midpoint_estimation=midpoint,
        seasonal_phase_months=phase,
        seasonal_desal_amplitude=amplitude,
        k_deriv = 0,
        trend_time_window = 24*30
    )

    return {
        # === Variables de entrada ===
        "potencia_solar": p_solar,
        "potencia_eolica": p_eolica,
        "potencia_baterias": p_baterias,
        "min_run_hours": run_h,
        "max_desalation": desal,
        "midpoint_estimation": midpoint,
        "overflow_pct": overflow,
        "seasonal_phase_corrector": phase,
        "seasonal_amplitude_corrector": amplitude,
        "regen_base_pct": regen_base,
        "llindar_activacio_regen_max": llindar_regen,
        "llindars_fases": [x1, x1+x2, x1+x2+x3, x1+x2+x3+x4],

        # === Métricas de salida (ejemplo, añade más) ===
        "min_level": results["level_final"].min(),
        "max_level": results["level_final"].max(),
        "mean_level": results['level_final'].mean(),
        "level_pct": results['level_final'],
        "desal_hm3": results['desal_final_hm3'].sum(),
        'regen_hm3': results['regen_final'].sum(),        
        'spillage_hm3': results['spillage_hm3'].sum(),        
        "restriction_days": results['hydro_metrics']['Restricciones escenario (días)'],
        "restriction_savings": results['savings_final'].sum(),
        "desal_cf": results['capacity_factor'],
        "surpluses": results['surpluses_net'],
        "gas_imports": results['energy_data']['Gas+Imports'].sum(),
        "instal_costs": results['costes']['total'],
        "seasonal_amplitude": results['hydro_metrics']['Variación estacional (%)']
    }

#%%
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# =============================================================================
# PRECOMPUTATS
# =============================================================================
nivells = np.arange(0, 100.1, 0.1)
hydro_min_lookup = np.array([hydro_min_for_level(n) for n in nivells])
precomputed['hydro_min_lookup'] = hydro_min_lookup

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
    'potencia_solar':      {'actiu': True,  'rang': np.arange(400, 35100, 100),   'defecte': 7180}, #CAMBIO 1
    'potencia_eolica':     {'actiu': True,  'rang': np.arange(1400, 30100, 100),  'defecte': 6234},
    'potencia_baterias':   {'actiu': True,  'rang': np.arange(500, 10100, 100),    'defecte': 2000},
    
    # --- Variables hídriques generals ---
    'min_run_hours':       {'actiu': True,  'rang': np.arange(6, 13, 1),          'defecte': 8},
    # 'max_desalation':      {'actiu': True,  'rang': np.arange(32, 161, 1),        'defecte': 32},
    'max_desalation':      {'actiu': True,  'rang': np.arange(32, 241, 1),        'defecte': 32}, # CAMBIO 2
    'max_regen':           {'actiu': True,  'rang': np.arange(0.173, 0.351, 0.001), 'defecte': 0.173}, #TRUE !!!
    'llindar_desal_max':   {'actiu': True,  'rang': np.arange(1, 6, 1),           'defecte': 2}, #TRUE !!!
    'midpoint_estimation': {'actiu': True,  'rang': np.arange(10, 96, 1),         'defecte': 75},
    'overflow_threshold':  {'actiu': True,  'rang': np.arange(40, 96, 1),         'defecte': 90},
    'seasonal_phase':      {'actiu': False,  'rang': np.arange(0, 11.9, 0.1),      'defecte': 0.0},
    'seasonal_amplitude':  {'actiu': False,  'rang': np.arange(0, 1.1, 0.1),       'defecte': 0.0},
    'regen_base_pct':      {'actiu': False,  'rang': np.arange(0.35, 0.65, 0.05),  'defecte': 0.5},
    'llindar_regen_max':   {'actiu': True,  'rang': np.arange(1, 5, 1),           'defecte': 1}, #TRUE !!!
    'derivada_nivell':     {'actiu': False,  'rang': np.arange(0, 11, 0.5),       'defecte': 0.0},
    
    # --- Llindars de sequera (per increments) ---
    'x1_base_eme':         {'actiu': True, 'rang': np.arange(10, 21, 1),         'defecte': 16}, #TRUE A TOTS!!!
    'x2_gap_exc':          {'actiu': True, 'rang': np.arange(5, 21, 1),         'defecte': 9},
    'x3_gap_ale':          {'actiu': True, 'rang': np.arange(5, 21, 1),         'defecte': 15}, #se puede dejar que alerta y prealerta fluctuen
    'x4_gap_pre':          {'actiu': True, 'rang': np.arange(5, 21, 1),         'defecte': 20},
}

nucleares_activas=[True, True, True]
potencia_cogeneracion=943.503

# nucleares_activas=[False, False, False]
# potencia_cogeneracion=122.4

demanda_electrica=1

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
        potencia_autoconsumo=1188.6267,
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
n_samples = 10#50000

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
escenaris = generar_escenaris(config, n_samples, seed=42, max_prealerta=MAX_PREALERTA)

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
df_rgs = pd.DataFrame(results)
# print(f"Escenaris executats: {len(df_tests)}")
# df_tests.head()

#%%


# Normalització min-max [0, 1]
def normalitzar(valor, vmin, vmax):
    return (valor - vmin) / (vmax - vmin)

# df_rgs.to_parquet('df_rgs_50k_14v_E2030.parquet')
# df_tests.to_parquet('df_rgs_50k_7v_E1a.parquet')
# df_tests.to_feather('df_tests_50k_9v.feather')

df_tests = df_rgs
# Obtenir rangs del teu df_tests (RGS previ)
min_level_range = (df_tests['min_level'].min(), df_tests['min_level'].max())
mean_level_range = (df_tests['mean_level'].min(), df_tests['mean_level'].max())  # ex: (5, 45)
costs_range = (df_tests['total_costs'].min(), df_tests['total_costs'].max())  # ex: (7e9, 15e9)
msqdev_range = (df_tests['squared_dev'].min(), df_tests['squared_dev'].max())
gasimports_range = (df_tests['gas_imports'].min(), df_tests['gas_imports'].max())

min_level_range = (4.091426559989756, 73.48088072170106)
mean_level_range = (25.959177233036332, 95.38898047533267)
costs_range = (1712806780.0313098, 22016563347.90802)
msqdev_range = (44.643195403750916, 5651.79378099446)
gasimports_range = (0.0, 319864049.3653991)

#%% - CARREGAR DATASETS I RANGS
# df_tests = pd.read_parquet('df_rgs_50k_7v_old.parquet')
# df_rgs = pd.read_parquet('df_rgs_50k_7v_E1b.parquet')

# Obtenir rangs del teu df_tests (RGS previ)
min_level_range = (df_tests['min_level'].min(), df_tests['min_level'].max())
mean_level_range = (df_tests['mean_level'].min(), df_tests['mean_level'].max())  # ex: (5, 45)
costs_range = (df_tests['total_costs'].min(), df_tests['total_costs'].max())  # ex: (7e9, 15e9)
msqdev_range = (df_tests['squared_dev'].min(), df_tests['squared_dev'].max())
gasimports_range = (df_tests['gas_imports'].min(), df_tests['gas_imports'].max())

# # A la funció objectiu:
# obj_level = normalitzar(resultado['level_final'].min(), *min_level_range)
# obj_costs = normalitzar(resultado['costes']['total'], *costs_range)

# # Combinar (ara tots dos entre 0 i 1)
# objetivo = -0.5 * obj_level + 0.5 * obj_costs  # Negatiu perquè volem MAX min_level

#%%
%%time
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# === 4. Ejecuta en paralelo ===
# if __name__ == "__main__":
#     results = Parallel(n_jobs=-1, verbose=10)(
#         delayed(run_case)(*params) for params in scenarios_params
#     )


results = Parallel(n_jobs=-1)(
    delayed(run_case)(*params) for params in tqdm(scenarios_params, desc="Escenaris")
)
    
  
# === 5. Guardar en DataFrame y exportar ===
df_tests = pd.DataFrame(results)
    
df_tests[df_tests['max_level'] <= 100.1]
# # df_tests[df_tests['max_level'] > 100]
# df_tests.iloc[98]['level_pct'].plot()
# df_tests.iloc[77]['level_pct'].plot()

# df_tests.iloc[98]['spillage_hm3']

# test_list = []
# count = 1
# for i in range(len(df_tests)):
#     test_list.append(df_tests.iloc[i]['spillage_hm3'])
#     if df_tests.iloc[i]['spillage_hm3'] == 0.0:
#         print(count)
#         count += 1
        
        
#%%
%%time
# Suposem que tens df_tests amb els resultats dels 100 escenaris
# i scenarios_params amb els paràmetres d'entrada

# Escollir escenari per índex (ex: el millor segons una mètrica)
idx = df_tests['min_level'].idxmax()  # O qualsevol criteri

# Obtenir paràmetres d'aquest escenari
params = scenarios_params[idx]
# params = escenaris[28755]

# Executar individualment
result = run_case(*params)


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
    # hydro_min_for_level=hydro_min_for_level,  # NOU: Afegir aquesta variable
    
    # --- Paràmetres hídrics ---
    consumo_base_diario_estacional_hm3=consumo_base_diario_estacional,  # NOU
    save_hm3_per_mwh=1/desal_sensibility, 

    precomputed=precomputed,
    potencia_solar=params[0],
    potencia_eolica=params[1],
    potencia_baterias=params[2],
    min_run_hours=params[3],
    max_desalation=params[4],
    midpoint_estimation=params[5],
    overflow_threshold_pct=params[6],
    seasonal_phase_months=params[7],
    seasonal_desal_amplitude=params[8],
)

df_sintetic = result_complet['energy_data']

#%%
%%time
# Suposem que tens df_tests amb els resultats dels 100 escenaris
# i scenarios_params amb els paràmetres d'entrada

# Escollir escenari per índex (ex: el millor segons una mètrica)
idx = df_tests['min_level'].idxmax()  # O qualsevol criteri

# Obtenir paràmetres d'aquest escenari
# params = escenaris[idx]
params = escenaris[28755]

umbrales = increments_a_llindars(
    params['x1_base_eme'],
    params['x2_gap_exc'],
    params['x3_gap_ale'],
    params['x4_gap_pre']
)

# Executar individualment
result = run_case(params)


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
    regen_base_pct=params['regen_base_pct'],
    llindar_activacio_regen_max=params['llindar_regen_max'],
    k_deriv=params['derivada_nivell'],
    umbrales_sequia=umbrales,    
)

df_sintetic = result_complet['energy_data']
#%%
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
sample[inicio:fin]['Càrrega'].plot(ax=ax, color='black', style='--', linewidth=2)
(sample[inicio:fin]['Demanda']).plot(ax=ax, color='grey', linewidth=2, label='Demanda')


ax.legend(loc='upper left')
plt.show()


#%%

def extract_pareto_front(df, objectives, directions):
    """
    Extrae las soluciones no dominadas (Pareto frontal) de un DataFrame.
    
    Args:
        df: DataFrame con las soluciones
        objectives: Lista de nombres de columnas que representan los objetivos
        directions: Lista con 'min' o 'max' para cada objetivo (misma longitud que objectives)
        
    Returns:
        DataFrame con solo las soluciones no dominadas
    """
    
    # Validaciones
    if len(objectives) != len(directions):
        raise ValueError("objectives y directions deben tener la misma longitud")
    
    for obj in objectives:
        if obj not in df.columns:
            raise ValueError(f"Columna '{obj}' no encontrada en el DataFrame")
    
    # Crear copia del DataFrame
    df_work = df.copy()
    
    # Normalizar direcciones: convertir 'max' a 'min' multiplicando por -1
    for i, (obj, direction) in enumerate(zip(objectives, directions)):
        if direction == 'max':
            df_work[f"{obj}_norm"] = -df_work[obj]
        else:
            df_work[f"{obj}_norm"] = df_work[obj]
    
    # Objetivos normalizados (todos para minimizar)
    obj_norm_cols = [f"{obj}_norm" for obj in objectives]
    
    # Matriz de objetivos
    obj_matrix = df_work[obj_norm_cols].values
    n_solutions = len(df_work)
    
    # Vector para marcar soluciones dominadas
    is_dominated = np.zeros(n_solutions, dtype=bool)
    
    # Algoritmo de dominancia O(n²)
    for i in range(n_solutions):
        if is_dominated[i]:
            continue
            
        for j in range(n_solutions):
            if i == j or is_dominated[j]:
                continue
                
            # Verificar si la solución i domina a la j
            # Una solución domina a otra si es mejor o igual en todos los objetivos
            # y estrictamente mejor en al menos uno
            dominates = True
            at_least_one_better = False
            
            for k in range(len(objectives)):
                if obj_matrix[i, k] > obj_matrix[j, k]:  # i es peor que j en objetivo k
                    dominates = False
                    break
                elif obj_matrix[i, k] < obj_matrix[j, k]:  # i es mejor que j en objetivo k
                    at_least_one_better = True
            
            # Si i domina a j, marcar j como dominada
            if dominates and at_least_one_better:
                is_dominated[j] = True
    
    # Retornar solo las soluciones no dominadas
    pareto_solutions = df[~is_dominated].copy()
    
    # Limpiar columnas temporales
    cols_to_drop = [col for col in df_work.columns if col.endswith('_norm') and col not in df.columns]
    
    return pareto_solutions

def plot_pareto_front_2d(df, pareto_df, obj1, obj2, dir1='min', dir2='min'):
    """
    Visualiza el frente de Pareto para 2 objetivos.
    
    Args:
        df: DataFrame completo con todas las soluciones
        pareto_df: DataFrame solo con soluciones no dominadas
        obj1, obj2: Nombres de las columnas de los objetivos
        dir1, dir2: 'min' o 'max' para cada objetivo
    """
    
    plt.figure(figsize=(10, 6))
    
    # Todas las soluciones
    plt.scatter(df[obj1], df[obj2], alpha=0.5, color='lightblue', 
                label=f'Totes les soluciones factibles ({len(df)})', s=50)
    
    # Soluciones de Pareto
    plt.scatter(pareto_df[obj1], pareto_df[obj2], color='red', 
                label=f'Front de Pareto ({len(pareto_df)})', s=100, edgecolor='black')
    
    plt.xlabel(f'{obj1} {"(minimitzar)" if dir1=="min" else "(maximitzar)"}')
    plt.ylabel(f'{obj2} {"(minimitzar)" if dir2=="min" else "(maximitzar)"}')
    plt.title('Front de Pareto - Solucions Factibles No Dominades')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ===================================================================
# EJEMPLOS DE USO ESPECÍFICOS PARA TU CASO
# ===================================================================

def extract_pareto_energy_scenarios(df_results):
    """
    Extrae soluciones no dominadas específicamente para el problema energético.
    Múltiples configuraciones típicas.
    """
    
    scenarios = {
        # Configuración 1: Minimizar costes e importaciones, maximizar nivel mínimo
        "costo_vs_seguridad": {
            "objectives": ["total_costs", "min_level"],
            "directions": ["min", "max"],
            "description": "Minimitzar costos i maximitzar seguretat hídrica"
        },


        "costo_vs_seguridad2": {
            "objectives": ["total_costs", "mean_level"],
            "directions": ["min", "max"],
            "description": "Minimitzar costos i maximitzar seguretat hídrica"
        },                
        
        "costo_vs_seguridad3": {
            "objectives": ["total_costs", "squared_dev"],
            "directions": ["min", "min"],
            "description": "Minimitzar costos i maximitzar seguretat hídrica"
        },        
        
        "costo_seguridad_descarbonzacion": {
            "objectives": ["total_costs", "squared_dev","gas_imports"],
            "directions": ["min", "min","min"],
            "description": "Minimitzar costos, maximitzar seguretat hídrica y minimizar dependencia"
        },        
        
        # Pregunta: Podem descarbonitzar sense comprometre l'aigua?
        "descarbonizacion_seguridad": {
            "objectives": ["gas_imports", "min_level"],
            "directions": ["min", "max"],
            "description": "Trade-off entre emissions i seguretat hídrica"
        },   
        
        # Pregunta: Podem descarbonitzar sense comprometre l'aigua?
        "descarbonizacion_seguridad": {
            "objectives": ["gas_imports", "min_level"],
            "directions": ["min", "max"],
            "description": "Trade-off entre emissions i seguretat hídrica"
        },         
        
        # Pregunta: Podem descarbonitzar sense comprometre l'aigua?
        "descarbonizacion_seguridad3": {
            "objectives": ["gas_imports", "squared_dev"],
            "directions": ["min", "min"],
            "description": "Trade-off entre emissions i seguretat hídrica"
        },
        
        # Configuración 3: Costes vs dias de restricción
        "costo_vs_seguridad4": {
            "objectives": ["total_costs", "restriction_days"],
            "directions": ["min", "min"],
            "description": "Minimizar costes, restricciones e importaciones"
        },        

        # Pregunta: Podem descarbonitzar sense comprometre l'aigua?
        "descarbonizacion_costes": {
            "objectives": ["gas_imports", "total_costs"],
            "directions": ["min", "min"],
            "description": "Trade-off entre emissions i costos"
        },          
        
        "seguridad_vs_desalacion": {
            "objectives": ["min_level", "max_desalation"],
            "directions": ["max", "min"],
            "description": "Minimizar potencia de desalación y maximizar seguridad hídrica"
        },
        
      
        # Configuración 4: Maximizar desalación vs minimizar costes
        "costo_vs_desalacion2": {
            "objectives": ["desal_cf", "total_costs"],
            "directions": ["max", "min"],
            "description": "Maximizar factor de capacidad desalación vs minimizar costes"
        }
    }
    
    results = {}
    
    for name, config in scenarios.items():
        print(f"\nExtrayendo Pareto para: {config['description']}")
        
        pareto_solutions = extract_pareto_front(
            df_results, 
            config["objectives"], 
            config["directions"]
        )
        
        print(f"Soluciones totales: {len(df_results)}")
        print(f"Soluciones no dominadas: {len(pareto_solutions)}")
        print(f"Reducción: {(1 - len(pareto_solutions)/len(df_results))*100:.1f}%")
        
        results[name] = {
            "pareto_df": pareto_solutions,
            "config": config,
            "reduction_pct": (1 - len(pareto_solutions)/len(df_results))*100
        }
    
    return results

from mpl_toolkits.mplot3d import Axes3D

def plot_pareto_front_3d(df, pareto_df, obj1, obj2, obj3, dir1='min', dir2='min', dir3='min'):
    """
    Visualiza el frente de Pareto para 3 objetivos en 3D.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Todas las soluciones
    ax.scatter(df[obj1], df[obj2], df[obj3], 
              alpha=0.3, color='lightblue', s=30,
              label=f'Todas ({len(df)})')
    
    # Soluciones de Pareto
    ax.scatter(pareto_df[obj1], pareto_df[obj2], pareto_df[obj3], 
              color='red', s=100, edgecolor='black',
              label=f'Pareto ({len(pareto_df)})')
    
    ax.set_xlabel(f'{obj1} {"(min)" if dir1=="min" else "(max)"}')
    ax.set_ylabel(f'{obj2} {"(min)" if dir2=="min" else "(max)"}')
    ax.set_zlabel(f'{obj3} {"(min)" if dir3=="min" else "(max)"}')
    ax.set_title('Frente de Pareto 3D')
    ax.legend()
    
    ax.zaxis.labelpad = -25
    plt.tight_layout()
    plt.show()

import plotly.express as px

def plot_pareto_3d_plotly(df, pareto_df, obj1, obj2, obj3, dir1='min', dir2='min', dir3='min'):
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Sampling para velocidad
    sample_df = df.sample(n=min(3000, len(df)))
    fig.add_trace(go.Scatter3d(
        x=sample_df[obj1], y=sample_df[obj2], z=sample_df[obj3],
        mode='markers', opacity=0.3, marker=dict(size=2, color='lightblue'),
        name=f'Todas ({len(df)})'
    ))
    
    # Pareto completo
    fig.add_trace(go.Scatter3d(
        x=pareto_df[obj1], y=pareto_df[obj2], z=pareto_df[obj3],
        mode='markers', marker=dict(size=6, color='red'),
        name=f'Pareto ({len(pareto_df)})'
    ))
    
    # Etiquetas personalizadas con direcciones
    dir_text = {'min': 'minimizar', 'max': 'maximizar'}
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=f'{obj1} ({dir_text[dir1]})'),
            yaxis=dict(title=f'{obj2} ({dir_text[dir2]})'),
            zaxis=dict(title=f'{obj3} ({dir_text[dir3]})')
        ),
        title='Frente de Pareto 3D'
    )
    
    fig.show(renderer="browser")

#%%

# ===================================================================
# EJEMPLO DE USO
# ===================================================================

# Extraer múltiples frontales de Pareto
pareto_results = extract_pareto_energy_scenarios(df_rgs)

# # Acceder a una configuración específica
# pareto_config1 = pareto_results['seguridad_vs_desalacion']['pareto_df']


plot_pareto_front_3d(
    df_tests, 
    pareto_results['costo_seguridad_dependencia']["pareto_df"],
    "total_costs", "squared_dev", "gas_imports",
    "min", "min", "min"
)

plot_pareto_front_3d(
    df_tests, 
    pareto_results['nexe_msd']["pareto_df"],
    "total_costs", "squared_dev", "gas_imports",
    "min", "min", "min"
)

plot_pareto_3d_plotly(
    df_tests, 
    pareto_results['costo_seguridad_dependencia']["pareto_df"],
    "total_costs", "squared_dev", "gas_imports",
    "min", "min", "min"
)


# Visualizar (para 2 objetivos)
plot_pareto_front_2d(
    df_tests, 
    pareto_results['seguridad_vs_desalacion']["pareto_df"],
    "max_desalation", "min_level", 
    "min", "max"
)

# pareto_config2 = pareto_results['costo_vs_seguridad']['pareto_df']


# Visualizar (para 2 objetivos)
plot_pareto_front_2d(
    df_tests, 
    pareto_results['costo_vs_seguridad']["pareto_df"],
    "total_costs", "min_level", 
    "min", "max"
)

plot_pareto_front_2d(
    df_tests, 
    pareto_results['costo_vs_seguridad2']["pareto_df"],
    "total_costs", "mean_level", 
    "min", "max"
)

plot_pareto_front_2d(
    df_tests, 
    pareto_results['costo_vs_seguridad3']["pareto_df"],
    "total_costs", "squared_dev", 
    "min", "min"
)

plot_pareto_front_2d(
    df_tests, 
    pareto_results['costo_vs_seguridad4']["pareto_df"],
    "total_costs", "restriction_days", 
    "min", "min"
)

plot_pareto_front_2d(
    df_tests, 
    pareto_results['seguridad_vs_desalacion2']["pareto_df"],
    "min_run_hours", "mean_level",
    "max", "max"
)

plot_pareto_front_2d(
    df_tests, 
    pareto_results['costo_vs_desalacion2']["pareto_df"],
    "total_costs", "desal_cf", 
    "min", "max"
)


# plot_pareto_front_2d(
#     df_tests, 
#     pareto_results['descarbonitzacio_vs_seguretat']["pareto_df"],
#     "gas_imports", "min_level", 
#     "min", "max"
# )

# plot_pareto_front_2d(
#     df_tests, 
#     pareto_results['descarbonitzacio_vs_seguretat2']["pareto_df"],
#     "gas_imports", "squared_dev", 
#     "min", "min"
# )

# pareto_config3 = pareto_results['costo_seguridad_dependencia']['pareto_df']


plot_pareto_front_2d(
    df_rgs, 
    pareto_results['desal_cf_vs_cost']["pareto_df"],
    "total_costs", "desal_cf", 
    "min", "max"
)

plot_pareto_front_2d(
    df_rgs, 
    pareto_results['cost_vs_nivell_min']["pareto_df"],
    "total_costs", "min_level", 
    "min", "max"
)

plot_pareto_front_2d(
    df_rgs, 
    pareto_results['cost_vs_nivell_mitja']["pareto_df"],
    "total_costs", "mean_level", 
    "min", "max"
)

#%%

def analisi_rellevancia_completa(df_tests, parametre, metriques_objectiu, valor_ref=0, pareto_config=None):
    """Anàlisi completa de rellevància d'un paràmetre."""
    
    print(f"\n{'='*60}")
    print(f"ANÀLISI DE RELLEVÀNCIA: {parametre}")
    print(f"{'='*60}")
    
    # 1. Correlació
    print("\n1. CORRELACIONS")
    for m in metriques_objectiu:
        corr = df_tests[parametre].corr(df_tests[m])
        stars = '***' if abs(corr) > 0.5 else '**' if abs(corr) > 0.3 else '*' if abs(corr) > 0.1 else ''
        print(f"   {m}: {corr:.3f} {stars}")
    
    # 2. Comparació de grups
    print(f"\n2. COMPARACIÓ GRUPS (ref={valor_ref})")
    grup_ref = df_tests[df_tests[parametre] == valor_ref]
    grup_actiu = df_tests[df_tests[parametre] != valor_ref]
    
    if len(grup_ref) > 0 and len(grup_actiu) > 0:
        for m in metriques_objectiu:
            diff = grup_actiu[m].mean() - grup_ref[m].mean()
            print(f"   {m}: Δ = {diff:+.2f}")
    
    # 3. Variabilitat en Pareto (si es proporciona config)
    if pareto_config:
        print(f"\n3. VARIABILITAT EN FRONT DE PARETO")
        pareto_df = extract_pareto_front(df_tests, pareto_config['objectives'], pareto_config['directions'])
        
        std_pareto = pareto_df[parametre].std()
        std_total = df_tests[parametre].std()
        
        print(f"   Std dins Pareto: {std_pareto:.3f}")
        print(f"   Std total: {std_total:.3f}")
        print(f"   Ratio: {std_pareto/std_total:.2f}" if std_total > 0 else "   Ratio: N/A")
    
    # 4. Recomanació
    print(f"\n4. RECOMANACIÓ")
    corr_max = max(abs(df_tests[parametre].corr(df_tests[m])) for m in metriques_objectiu)
    
    if corr_max < 0.1:
        print(f"   → FIXAR a {valor_ref}: Impacte negligible")
    elif corr_max < 0.3:
        print(f"   → CONSIDERAR FIXAR: Impacte menor")
    else:
        print(f"   → MANTENIR COM A VARIABLE: Impacte significatiu")

# Ús
analisi_rellevancia_completa(
    df_tests,
    'derivada_nivell',
    ['min_level', 'mean_level', 'restriction_days', 'restriction_savings', 'gas_imports', 'desal_cf', 'total_costs','seasonal_amplitude_out'],
    valor_ref=0,
    pareto_config={"objectives": ["gas_imports", "min_level"], "directions": ["min", "max"]}
)

#%%
def analisi_rellevancia_completa(df_tests, parametre, metriques_objectiu, valor_ref=None, pareto_config=None):
    """
    Anàlisi completa de rellevància d'un paràmetre.
    
    Args:
        df_tests: DataFrame amb resultats dels escenaris
        parametre: Nom de la variable de decisió a analitzar
        metriques_objectiu: Llista de mètriques de sortida
        valor_ref: Valor de referència per comparació de grups.
                   Si None, usa anàlisi per tercils.
        pareto_config: Diccionari amb 'objectives' i 'directions' per anàlisi Pareto
    
    Returns:
        dict amb resultats de l'anàlisi
    """
    
    print(f"\n{'='*60}")
    print(f"ANÀLISI DE RELLEVÀNCIA: {parametre}")
    print(f"{'='*60}")
    
    resultats = {'parametre': parametre, 'correlacions': {}, 'diffs': {}}
    
    # =========================================================================
    # 1. CORRELACIÓ
    # =========================================================================
    print("\n1. CORRELACIONS")
    corr_max = 0
    for m in metriques_objectiu:
        corr = df_tests[parametre].corr(df_tests[m])
        corr_max = max(corr_max, abs(corr))
        stars = '***' if abs(corr) > 0.5 else '**' if abs(corr) > 0.3 else '*' if abs(corr) > 0.1 else ''
        print(f"   {m}: {corr:.3f} {stars}")
        resultats['correlacions'][m] = corr
    
    resultats['corr_max'] = corr_max
    
    # =========================================================================
    # 2. COMPARACIÓ DE GRUPS
    # =========================================================================
    if valor_ref is not None:
        # Mètode A: Comparació amb valor de referència
        print(f"\n2. COMPARACIÓ GRUPS (ref={valor_ref})")
        grup_ref = df_tests[df_tests[parametre] == valor_ref]
        grup_actiu = df_tests[df_tests[parametre] != valor_ref]
        
        print(f"   n(ref={valor_ref}): {len(grup_ref)}")
        print(f"   n(altres): {len(grup_actiu)}")
        
        if len(grup_ref) > 0 and len(grup_actiu) > 0:
            for m in metriques_objectiu:
                mean_ref = grup_ref[m].mean()
                mean_actiu = grup_actiu[m].mean()
                diff = mean_actiu - mean_ref
                diff_pct = (diff / mean_ref * 100) if mean_ref != 0 else 0
                print(f"   {m}: Δ = {diff:+.2f} ({diff_pct:+.1f}%)")
                resultats['diffs'][m] = {'abs': diff, 'pct': diff_pct}
        else:
            print(f"   ⚠ Insuficients dades per comparació")
    else:
        # Mètode B: Comparació per tercils
        print(f"\n2. COMPARACIÓ PER TERCILS")
        q33 = df_tests[parametre].quantile(0.33)
        q66 = df_tests[parametre].quantile(0.66)
        
        grup_baix = df_tests[df_tests[parametre] <= q33]
        grup_alt = df_tests[df_tests[parametre] >= q66]
        
        print(f"   Tercil baix ({parametre} ≤ {q33:.2f}): n={len(grup_baix)}")
        print(f"   Tercil alt ({parametre} ≥ {q66:.2f}): n={len(grup_alt)}")
        
        if len(grup_baix) > 0 and len(grup_alt) > 0:
            for m in metriques_objectiu:
                mean_baix = grup_baix[m].mean()
                mean_alt = grup_alt[m].mean()
                diff = mean_alt - mean_baix
                diff_pct = (diff / mean_baix * 100) if mean_baix != 0 else 0
                print(f"   {m}: Δ(alt-baix) = {diff:+.2f} ({diff_pct:+.1f}%)")
                resultats['diffs'][m] = {'abs': diff, 'pct': diff_pct}
        else:
            print(f"   ⚠ Insuficients dades per comparació")
    
    # =========================================================================
    # 3. VARIABILITAT EN PARETO
    # =========================================================================
    if pareto_config:
        print(f"\n3. VARIABILITAT EN FRONT DE PARETO")
        pareto_df = extract_pareto_front(df_tests, pareto_config['objectives'], pareto_config['directions'])
        
        std_pareto = pareto_df[parametre].std()
        std_total = df_tests[parametre].std()
        mean_pareto = pareto_df[parametre].mean()
        mean_total = df_tests[parametre].mean()
        
        ratio = std_pareto / std_total if std_total > 0 else float('nan')
        
        print(f"   Mitjana Pareto: {mean_pareto:.3f} (vs total: {mean_total:.3f})")
        print(f"   Std Pareto: {std_pareto:.3f} (vs total: {std_total:.3f})")
        print(f"   Ratio Std: {ratio:.2f}")
        
        resultats['pareto'] = {
            'mean_pareto': mean_pareto,
            'mean_total': mean_total,
            'std_pareto': std_pareto,
            'std_total': std_total,
            'ratio': ratio
        }
        
        if ratio < 0.5:
            print(f"   → El paràmetre CONVERGEIX en el front òptim")
        elif std_pareto < 0.01 * mean_pareto:
            print(f"   → Valors quasi CONSTANTS en el Pareto")
        else:
            print(f"   → Alta variabilitat, NO determina l'optimalitat")
    
    # =========================================================================
    # 4. RECOMANACIÓ
    # =========================================================================
    print(f"\n4. RECOMANACIÓ")
    
    # Criteri basat en correlació màxima
    if corr_max < 0.1:
        recomanacio = "FIXAR"
        motiu = "Impacte negligible (|r| < 0.1)"
        if valor_ref is not None:
            recomanacio += f" a {valor_ref}"
        else:
            recomanacio += f" al valor per defecte"
    elif corr_max < 0.3:
        recomanacio = "CONSIDERAR FIXAR"
        motiu = "Impacte menor (0.1 ≤ |r| < 0.3)"
    else:
        recomanacio = "MANTENIR COM A VARIABLE"
        motiu = f"Impacte significatiu (|r| = {corr_max:.2f})"
    
    print(f"   → {recomanacio}")
    print(f"      {motiu}")
    
    resultats['recomanacio'] = recomanacio
    resultats['motiu'] = motiu
    
    return resultats


# =============================================================================
# FUNCIÓ PER ANALITZAR MÚLTIPLES VARIABLES
# =============================================================================
def analisi_rellevancia_multiple(df_tests, variables_config, metriques_objectiu, pareto_config=None):
    """
    Analitza la rellevància de múltiples variables de decisió.
    
    Args:
        df_tests: DataFrame amb resultats
        variables_config: Dict amb {nom_variable: valor_ref} (None per tercils)
        metriques_objectiu: Llista de mètriques de sortida
        pareto_config: Configuració del front de Pareto
    
    Returns:
        DataFrame resum amb recomanacions
    """
    
    resultats_tots = []
    
    for var, val_ref in variables_config.items():
        if var not in df_tests.columns:
            print(f"⚠ Variable '{var}' no trobada al DataFrame")
            continue
        
        res = analisi_rellevancia_completa(
            df_tests, var, metriques_objectiu, 
            valor_ref=val_ref, pareto_config=pareto_config
        )
        
        resultats_tots.append({
            'Variable': var,
            'Valor_ref': val_ref,
            'Corr_max': res['corr_max'],
            'Recomanació': res['recomanacio']
        })
    
    df_resum = pd.DataFrame(resultats_tots)
    
    print(f"\n{'='*60}")
    print("RESUM DE RECOMANACIONS")
    print(f"{'='*60}")
    print(df_resum.to_string(index=False))
    
    return df_resum


# =============================================================================
# ÚS
# =============================================================================

# Configuració de variables amb els seus valors de referència
variables_config = {
    # Variables amb valor de referència clar (0 = desactivat)
    'derivada_nivell': 0,
    'seasonal_amplitude': 0,
    
    # Variables sense valor de referència clar (usar tercils)
    'seasonal_phase': None,
    'potencia_solar': None,
    'potencia_eolica': None,
    'potencia_baterias': None,
    'max_desalation': None,
    'midpoint_estimation': None,
    'overflow_threshold': None,
    'regen_base_pct': None,
    'llindar_regen_max': None,
    'min_run_hours': None,
    'max_regen': None,
    
}

# metriques = ['min_level', 'mean_level', 'restriction_days', 'gas_imports', 'desal_cf']
metriques = ['squared_dev','total_costs', 'gas_imports', 'min_level', 'mean_level', 'restriction_days', 'restriction_savings', 'desal_cf', 'seasonal_amplitude_out']

pareto_config = {"objectives": ["squared_dev","total_costs", "gas_imports"], "directions": ["min", "min", "min"]}
# pareto_config = {"objectives": ["squared_dev","total_costs"], "directions": ["min", "min"]}

#%%
# Anàlisi individual
resultats = analisi_rellevancia_completa(
    df_tests,
    'L_emergencia',
    metriques,
    valor_ref=16,
    pareto_config=pareto_config
)

# Anàlisi múltiple
df_resum = analisi_rellevancia_multiple(
    df_tests, 
    variables_config, 
    metriques, 
    pareto_config
)


#%%

def analitzar_convergencia_pareto(df_tests, objectives, directions, steps=10):
    """
    Analitza l'estabilitat del front de Pareto incrementant el nombre de mostres.
    """
    n_total = len(df_tests)
    sample_sizes = np.linspace(n_total // steps, n_total, steps, dtype=int)
    
    resultats = []
    
    for n in sample_sizes:
        # Submostrar (mantenint ordre original per reproductibilitat)
        df_sub = df_tests.iloc[:n]
        
        # Extreure Pareto
        pareto = extract_pareto_front(df_sub, objectives, directions)
        
        # Mètriques del front
        stats = {
            'n_samples': n,
            'n_pareto': len(pareto),
            'pct_pareto': len(pareto) / n * 100
        }
        
        # Valors extrems de cada objectiu en el Pareto
        for obj in objectives:
            stats[f'{obj}_min'] = pareto[obj].min()
            stats[f'{obj}_max'] = pareto[obj].max()
        
        resultats.append(stats)
    
    df_conv = pd.DataFrame(resultats)
    
    # Calcular variació respecte al valor final
    print("\n=== CONVERGÈNCIA DEL FRONT DE PARETO ===\n")
    print(f"{'N':>10} | {'Pareto':>8} | ", end="")
    for obj in objectives:
        print(f'{obj[:12]:>14} |', end="")
    print()
    print("-" * (25 + 17 * len(objectives)))
    
    for i, row in df_conv.iterrows():
        print(f"{int(row['n_samples']):>10} | {int(row['n_pareto']):>8} | ", end="")
        for obj in objectives:
            val = row[f'{obj}_min']
            val_final = df_conv.iloc[-1][f'{obj}_min']
            var_pct = abs(val - val_final) / val_final * 100 if val_final != 0 else 0
            print(f"{val:>8.1f} ({var_pct:>3.0f}%) |", end="")
        print()
    
    return df_conv

# Ús
objectives = ['gas_imports', 'squared_dev', 'total_costs']
directions = ['min', 'min', 'min']

df_conv = analitzar_convergencia_pareto(df_tests, objectives, directions, steps=10)
   
    
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


class OptimizadorEnergetico:
    """
    Classe per optimitzar escenaris energètics usant diferents metaheurístiques.
    """
    
    def __init__(self, datos_base, funcion_procesar_escenario):
        self.datos_base = datos_base
        self.procesar_escenario = funcion_procesar_escenario
        self.mejor_resultado = None
        self.historial_evaluaciones = []
        self.variables = None
        self.cache_resultados = {}
        
    def definir_variables_decision(self):
        """Defineix els rangs de les variables de decisió."""
        self.variables = {
            'potencia_solar': {'min': 400, 'max': 35000},
            'potencia_eolica': {'min': 1400, 'max': 30000},
            'potencia_baterias': {'min': 500, 'max': 10000},
            # 'max_desalation': {'min': 32, 'max': 240},
            # 'overflow_threshold_pct': {'min': 40, 'max': 95},
            'overflow_threshold_pct': {'min': 50, 'max': 95},
            'min_run_hours': {'min': 6, 'max': 12, 'tipo': 'integer'},
            # 'midpoint_estimation': {'min': 10, 'max': 95},
            'midpoint_estimation': {'min': 50, 'max': 95},
            # 'seasonal_phase_months': {'min': 0, 'max': 11.9},
            # 'seasonal_desal_amplitude': {'min': 0, 'max': 1.0},
            # 'max_regen': {'min': 0.173, 'max': 0.350},
            'llindar_activacio_desal_max': {'min': 1, 'max': 5, 'tipo': 'integer'},
            'llindar_activacio_regen_max': {'min': 1, 'max': 4, 'tipo': 'integer'},
            # 'x1_base_eme': {'min': 10, 'max': 20, 'tipo': 'integer'},
            # 'x2_gap_exc': {'min': 5, 'max': 20, 'tipo': 'integer'},
            'x1_base_eme': {'min': 16, 'max': 16, 'tipo': 'integer'},
            'x2_gap_exc': {'min': 9, 'max': 9, 'tipo': 'integer'},
            'x3_gap_ale': {'min': 5, 'max': 20, 'tipo': 'integer'},
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
            
            umbrales = increments_a_llindars(
                params.pop('x1_base_eme'),
                params.pop('x2_gap_exc'),
                params.pop('x3_gap_ale'),
                params.pop('x4_gap_pre')
            )
            
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
    
    def optimizar_nsga2(self, ngen=30, pop_size=50, seed=42, parallel=True):  # AFEGIT parallel
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
            
            # individual.resultado = resultado  # 👈 CLAVE
            
            if resultado is None:
                return (1e10, 1e10, 1e10)
            
            # Penalitzar desbordament
            if resultado['level_final'].max() > 100.5:
                return (1e10, 1e10, 1e10)
            
            # Objectius: emissions (min), mean_sq_dev (min), costos (min)
            emisiones = resultado['energy_data']['Gas+Imports'].sum() / 1e6  # TWh
            mean_sq_dev = ((100 - resultado['level_final'])**2).sum() / len(resultado['level_final'])
            costos = resultado['costes']['total'] / 1e6  # M€
            
            return (emisiones, mean_sq_dev, costos)
        
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
        
        # Població inicial
        pop = toolbox.population(n=pop_size)
        
        # Hall of Fame (Pareto front)
        hof = tools.ParetoFront()
        
        # =========================================================================
        # AVALUAR POBLACIÓ INICIAL
        # =========================================================================
        print("Avaluant població inicial...")
        fitnesses = evaluar_batch(pop)  # CANVIAT: usa evaluar_batch
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
    
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
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Seleccionar nova generació
            pop = toolbox.select(pop + offspring, pop_size)
            hof.update(pop)
            
            # Mostrar progrés
            if (gen + 1) % 5 == 0 or gen == 0:
                fits = [ind.fitness.values for ind in pop]
                print(f"  Gen {gen+1}: Pareto size={len(hof)}, "
                      f"Emis[TWh]=[{min(f[0] for f in fits):.1f}, {max(f[0] for f in fits):.1f}], "
                      f"MSD=[{min(f[1] for f in fits):.1f}, {max(f[1] for f in fits):.1f}], "
                      f"Cost[M€]=[{min(f[2] for f in fits):.1f}, {max(f[2] for f in fits):.1f}]")
        
        print(f"\nNSGA-II completat! Solucions Pareto: {len(hof)}")
        
        return pop, hof    

    def hof_to_dataframe_old(self, hof):
        """Converteix el Hall of Fame a DataFrame."""
        records = []
        for ind in hof:
            params = self.decodificar_individuo(ind)
            params['obj_emissions'] = ind.fitness.values[0]
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
            
            # # PRIMER: Re-avaluar amb còpia neta (sense objectius)
            # resultado = self.evaluar_escenario(params.copy())
            # Recuperar del cache (o re-avaluar si no existeix)
            resultado = self.cache_resultados.get(cache_key)
            if resultado is None:
                resultado = self.evaluar_escenario(params.copy())            
            
            # DESPRÉS: Afegir objectius optimitzats
            params['obj_gasimports'] = ind.fitness.values[0]
            params['obj_msqdev'] = ind.fitness.values[1]
            params['obj_costs'] = ind.fitness.values[2]
            
            # resultado = getattr(ind, "resultado", None)
            
            
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
                # params['gas_imports'] = resultado['energy_data']['Gas+Imports'].sum() / 1e6
                
                # Llindars
                umbrales = increments_a_llindars(
                    int(round(ind[self.var_names.index('x1_base_eme')])),
                    int(round(ind[self.var_names.index('x2_gap_exc')])),
                    int(round(ind[self.var_names.index('x3_gap_ale')])),
                    int(round(ind[self.var_names.index('x4_gap_pre')]))
                )
                params['L_emergencia'] = umbrales['Emergencia']
                params['L_excepcionalitat'] = umbrales['Excepcionalitat']
                params['L_alerta'] = umbrales['Alerta']
                params['L_prealerta'] = umbrales['Prealerta']
                
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
        # 'nucleares_activas': [True, True, True],
        # 'potencia_cogeneracion': 943.503,
        'nucleares_activas': [False, False, False],
        'potencia_cogeneracion': 122.4,

        'duracion_horas': 4,
        'potencia_autoconsumo': 1188.6267,
        'demanda_electrica': 1,
        'CF_eolica_obj': None,
        'usar_CF_automatic': True,
        'trend_time_window': 24*30,
        'k_deriv': 0,
        'regen_base_pct': 0.5,
        # 'llindar_activacio_regen_max': 1,
        'seasonal_phase_months': 0,
        'seasonal_desal_amplitude': 0,
        'max_desalation': 64,
        'max_regen': 0.250,
        # 'overflow_threshold': 90,
        # 'umbrales_sequia': increments_a_llindars(16, 9, 15, 20),  # Valors per defecte
    }
    
    optimizador = OptimizadorEnergetico(datos_base, procesar_escenario_func)
    optimizador.definir_variables_decision()
    
    return optimizador

#%%
# =============================================================================
# EXEMPLE D'ÚS
# =============================================================================

if __name__ == "__main__":
    
    # Configurar
    optimizador = configurar_optimizador(procesar_escenario, precomputed, datos)
    
    
    # Opció 1: Differential Evolution (escalar, més ràpid)
    print("\n" + "="*60)
    print("DIFFERENTIAL EVOLUTION")
    print("="*60)
    t0 = time.time()
    resultado_de, params_de = optimizador.optimizar_differential_evolution(
        maxiter=25, #20,
        popsize=20,
        seed=42
    )
    print(f"Temps: {time.time()-t0:.1f}s") #654s 2 iteraciones

# -----------------------------------------------------------------------------
    
    # # Opció 2: NSGA-II (multiobjectiu)
    # print("\n" + "="*60)
    # print("NSGA-II")
    # print("="*60)
    # t0 = time.time()
    # pop, hof = optimizador.optimizar_nsga2(
    #     ngen=500,
    #     pop_size=64, #32,
    #     seed=42,
    #     parallel=True
    # )
    # print(f"Temps: {time.time()-t0:.1f}s") #99s 1 iteracion #903s 20 iter
    # # 4.5h 64 pop, 1000 gen
    
    # # Convertir resultats a DataFrame
    # df_pareto_nsga = optimizador.hof_to_dataframe(hof)
    # print(f"\nSolucions Pareto trobades: {len(df_pareto_nsga)}")
    # print(df_pareto_nsga.head())

# -----------------------------------------------------------------------------    
    # df_pareto_nsga.to_parquet('df_nsga_1k_14v.parquet')
    # df_nsga = pd.read_parquet('df_nsga_3k_9v.parquet')  # ✅ Funció "read_parquet"

# front_nsga[front_nsga.min_level > 50].midpoint_estimation.min()
# front_nsga[front_nsga.mean_level > 80].midpoint_estimation.min()
# front_nsga[front_nsga.min_level > 50].overflow_threshold.min()
# front_nsga[front_nsga.mean_level > 80].overflow_threshold.min()
# front_nsga[front_nsga.mean_level > 80].L_prealerta.min()
# front_nsga[front_nsga.mean_level > 80].L_alerta.min()

# front_nsga[front_nsga.mean_level > 80].L_prealerta.mean()

#%%
def plot_pareto_nsga2(hof, objetivos=['Gas_Imports', 'MSD', 'Costs']):
    """Gràfic 2D i 3D del front de Pareto."""
    
    # Extreure fitness values
    fits = np.array([ind.fitness.values for ind in hof])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 2D: Emissions vs Min Level
    axes[0].scatter(fits[:, 0], fits[:, 1], c=fits[:, 2], cmap='viridis', s=50)
    axes[0].set_xlabel(objetivos[0])
    axes[0].set_ylabel(objetivos[1])
    axes[0].set_title(f'{objetivos[0]} vs {objetivos[1]}')
    axes[0].colorbar = plt.colorbar(axes[0].collections[0], ax=axes[0], label=objetivos[2])
    
    # 2D: Emissions vs Costs
    axes[1].scatter(fits[:, 0], fits[:, 2], c=fits[:, 1], cmap='viridis', s=50)
    axes[1].set_xlabel(objetivos[0])
    axes[1].set_ylabel(objetivos[2])
    axes[1].set_title(f'{objetivos[0]} vs {objetivos[2]}')
    
    # 2D: Min Level vs Costs
    axes[2].scatter(fits[:, 1], fits[:, 2], c=fits[:, 0], cmap='viridis', s=50)
    axes[2].set_xlabel(objetivos[1])
    axes[2].set_ylabel(objetivos[2])
    axes[2].set_title(f'{objetivos[1]} vs {objetivos[2]}')
    
    plt.tight_layout()
    plt.show()
    
    # 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(fits[:, 0], fits[:, 1], fits[:, 2], c=fits[:, 1], cmap='viridis', s=50)
    ax.set_xlabel(objetivos[0])
    ax.set_ylabel(objetivos[1])
    ax.set_zlabel(objetivos[2])
    ax.set_title('Front de Pareto 3D')
    plt.colorbar(sc, label=objetivos[1])
    plt.show()

# Ús:
plot_pareto_nsga2(hof)

#%%
import plotly.graph_objects as go

def plot_pareto_nsga2_plotly(hof, objetivos=['Gas_Imports', 'MSD', 'Costs']):
    """Gràfic 2D i 3D del front de Pareto."""
    
    # Extreure fitness values
    fits = np.array([ind.fitness.values for ind in hof])
    
    # Gràfic 3D
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=fits[:, 0],
        y=fits[:, 1],
        z=fits[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=fits[:, 1],  # Color per segon eix
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=objetivos[1])
        ),
        name='Solucions Pareto'
    )])
    
    fig_3d.update_layout(
        title='Front de Pareto 3D',
        scene=dict(
            xaxis_title=objetivos[0],
            yaxis_title=objetivos[1],
            zaxis_title=objetivos[2]
        ),
        width=900,
        height=700
    )
    
    # Gràfic 2D (totes les combinacions)
    fig_2d = go.Figure()
    
    # Afegeix totes les parelles 2D
    from itertools import combinations
    parelles = list(combinations(range(3), 2))
    
    for i, (ax1, ax2) in enumerate(parelles):
        fig_2d.add_trace(go.Scatter(
            x=fits[:, ax1],
            y=fits[:, ax2],
            mode='markers',
            name=f'{objetivos[ax1]} vs {objetivos[ax2]}',
            marker=dict(size=8),
            visible=True if i==0 else 'legendonly'  # Només primera visible
        ))
    
    fig_2d.update_layout(
        title='Front de Pareto 2D (totes les combinacions)',
        xaxis_title=objetivos[0],
        yaxis_title=objetivos[1],
        width=1000,
        height=600
    )
    
    # Mostrar gràfics
    fig_3d.show(renderer="browser")
    fig_2d.show(renderer="browser")

    return fig_3d, fig_2d

# Ús
fig3d, fig2d = plot_pareto_nsga2_plotly(hof)

# fig_3d.write_html("pareto_3d.html")  # Obre després amb navegador
# fig_2d.write_html("pareto_2d.html")

#%%

def analitzar_regions_pareto(df_tests, objectives, directions, variables_decisio):
    """
    Analitza la distribució de les variables de decisió dins del front de Pareto
    per identificar regions d'interès i possibles acotacions.
    """
    
    # 1. Extreure front de Pareto
    pareto_df = extract_pareto_front(df_tests, objectives, directions)
    
    print(f"Total escenaris: {len(df_tests)}")
    print(f"Solucions Pareto: {len(pareto_df)} ({100*len(pareto_df)/len(df_tests):.1f}%)")
    print("\n" + "="*80)
    
    resultats = []
    
    for var in variables_decisio:
        if var not in df_tests.columns:
            continue
            
        # Estadístiques totals
        total_min = df_tests[var].min()
        total_max = df_tests[var].max()
        total_mean = df_tests[var].mean()
        total_std = df_tests[var].std()
        
        # Estadístiques Pareto
        pareto_min = pareto_df[var].min()
        pareto_max = pareto_df[var].max()
        pareto_mean = pareto_df[var].mean()
        pareto_std = pareto_df[var].std()
        
        # Percentils Pareto (per acotar)
        p5 = pareto_df[var].quantile(0.05)
        p95 = pareto_df[var].quantile(0.95)
        
        # Ràtio de concentració
        rang_total = total_max - total_min
        rang_pareto = pareto_max - pareto_min
        concentracio = 1 - (rang_pareto / rang_total) if rang_total > 0 else 0
        
        # Desplaçament (la mitjana Pareto vs total)
        desplacament = (pareto_mean - total_mean) / total_std if total_std > 0 else 0
        
        resultats.append({
            'variable': var,
            'total_min': total_min,
            'total_max': total_max,
            'pareto_min': pareto_min,
            'pareto_max': pareto_max,
            'pareto_p5': p5,
            'pareto_p95': p95,
            'concentracio': concentracio,
            'desplacament': desplacament
        })
        
        # Recomanació
        if concentracio > 0.5:
            rec = f"⭐ ACOTAR a [{p5:.1f}, {p95:.1f}]"
        elif abs(desplacament) > 1:
            direccio = "ALTS" if desplacament > 0 else "BAIXOS"
            rec = f"↗ Prefereix valors {direccio}"
        else:
            rec = "— Mantenir rang actual"
        
        print(f"\n{var}:")
        print(f"  Rang total:  [{total_min:.1f}, {total_max:.1f}]")
        print(f"  Rang Pareto: [{pareto_min:.1f}, {pareto_max:.1f}]")
        print(f"  Pareto P5-P95: [{p5:.1f}, {p95:.1f}]")
        print(f"  Concentració: {concentracio:.1%}")
        print(f"  Desplaçament: {desplacament:+.2f} σ")
        print(f"  Recomanació: {rec}")
    
    return pd.DataFrame(resultats), pareto_df


def plot_distribuicions_pareto(df_tests, pareto_df, variables_decisio, ncols=3):
    """Visualitza distribucions total vs Pareto per cada variable."""
    
    n_vars = len(variables_decisio)
    nrows = (n_vars + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    for i, var in enumerate(variables_decisio):
        if var not in df_tests.columns:
            continue
            
        ax = axes[i]
        
        # Histogrames superposats
        ax.hist(df_tests[var], bins=30, alpha=0.5, density=True, label='Total', color='gray')
        ax.hist(pareto_df[var], bins=30, alpha=0.7, density=True, label='Pareto', color='green')
        
        # Línies de mitjana
        ax.axvline(df_tests[var].mean(), color='gray', linestyle='--', linewidth=2)
        ax.axvline(pareto_df[var].mean(), color='green', linestyle='-', linewidth=2)
        
        ax.set_xlabel(var)
        ax.set_ylabel('Densitat')
        ax.legend()
        ax.set_title(f'{var}')
    
    # Amagar eixos sobrants
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def suggerir_nous_bounds(df_resultats, marge=0.1):
    """Genera nous bounds acotats basats en l'anàlisi."""
    
    print("\n" + "="*80)
    print("NOUS BOUNDS SUGGERITS (P5-P95 amb marge del 10%)")
    print("="*80)
    
    nous_bounds = {}
    
    for _, row in df_resultats.iterrows():
        var = row['variable']
        p5 = row['pareto_p5']
        p95 = row['pareto_p95']
        rang = p95 - p5
        
        nou_min = max(row['total_min'], p5 - marge * rang)
        nou_max = min(row['total_max'], p95 + marge * rang)
        
        nous_bounds[var] = {'min': nou_min, 'max': nou_max}
        
        reduccio = 1 - (nou_max - nou_min) / (row['total_max'] - row['total_min'])
        
        print(f"{var}:")
        print(f"  Original: [{row['total_min']:.1f}, {row['total_max']:.1f}]")
        print(f"  Nou:      [{nou_min:.1f}, {nou_max:.1f}] (reducció {reduccio:.0%})")
    
    return nous_bounds


# =============================================================================
# ÚS
# =============================================================================

# Definir objectius i variables
objectives = ['gas_imports', 'squared_dev', 'total_costs']
directions = ['min', 'min', 'min']

variables_decisio = [
    'potencia_solar', 'potencia_eolica', 'potencia_baterias',
    'max_desalation', 'midpoint_estimation', 'overflow_threshold',
    'min_run_hours', 'seasonal_phase', 'seasonal_amplitude'
]

# Analitzar
df_resultats, pareto_df = analitzar_regions_pareto(
    df_tests, objectives, directions, variables_decisio
)

# Visualitzar
plot_distribuicions_pareto(df_tests, pareto_df, variables_decisio)

# Suggerir nous bounds
nous_bounds = suggerir_nous_bounds(df_resultats)