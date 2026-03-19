# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 16:07:01 2025

@author: Víctor García
"""
import numpy as np
import pandas as pd
from numba import jit, njit
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional


# =============================================================================
# FUNCIONS AUXILIARS
# =============================================================================

# Funciones de carga y entrega para baterías e hidrógeno (potencia de carga y descarga desacopladas)

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
    hydro_min_lookup: np.ndarray,
    # hydro_min_for_level,
    # --- Parámetros de Control ---
    max_salto_pct_mensual: float = 10.0,
    puntos_optimizacion: int = 0,
    
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
            # hydro_min_for_level=hydro_min_for_level,
            hydro_min_lookup=hydro_min_lookup,  # NOU            
            puntos_optimizacion=puntos_optimizacion,
            # reajustar_por_overload_numpy=reajustar_por_overload_suavizado, #reajustar_por_overload_numpy,
            reajustar_por_overload_numpy=reajustar_por_overload_suavizado_optimizado, #reajustar_por_overload_numpy,
            suavizar_excedente_numpy=suavizar_excedente_rampa_optimizado, #suavizar_excedente_numpy,
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
            max_delta_mwh=0, #max_delta_int_mwh,
            # hydro_min_for_level=hydro_min_for_level,
            hydro_min_lookup=hydro_min_lookup,  # NOU
            puntos_optimizacion=puntos_optimizacion,
            # reajustar_por_overload_numpy=reajustar_por_overload_suavizado, #reajustar_por_overload_numpy,
            reajustar_por_overload_numpy=reajustar_por_overload_suavizado_optimizado, #reajustar_por_overload_numpy,
            suavizar_excedente_numpy=suavizar_excedente_rampa_optimizado, #suavizar_excedente_numpy,
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
    # hydro_min_for_level,
    hydro_min_lookup: np.ndarray,    
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
    # hydro_min = hydro_min_for_level(nivel_prom) * potencia_max
    # hydro_min = hydro_min_lookup[int(np.clip(nivel_prom, 0, 100))] * potencia_max
    idx = int(np.clip(nivel_prom * 10, 0, len(hydro_min_lookup) - 1))  # *10 perquè resolució 0.1%
    hydro_min = hydro_min_lookup[idx] * potencia_max    
    energia_obj = energia_mes + hydro_storage
    
    if puntos_optimizacion > 0:
        # potencias = np.linspace(hydro_min, potencia_max, puntos_optimizacion)
        # generaciones = [np.clip(gap_slice * f_cuenca, hydro_min, p).sum() for p in potencias]
        # deltas_posibles = energia_mes - np.array(generaciones)
        # error_obj = np.abs(deltas_posibles + hydro_storage)
        # hid_max_optimo = potencias[np.argmin(error_obj)]
        hid_max_optimo = optimitzar_hid_max_numba(gap_slice, f_cuenca, hydro_min, potencia_max, energia_obj, puntos_optimizacion)
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

@njit(cache=True, fastmath=True)
def optimitzar_hid_max_numba(
    gap_slice: np.ndarray,
    f_cuenca: float,
    hydro_min: float,
    potencia_max: float,
    energia_obj: float,
    n_punts: int
) -> float:
    """Cerca manual compilada amb Numba."""
    if n_punts <= 1:
        return hydro_min
    
    pas = (potencia_max - hydro_min) / (n_punts - 1)
    n = len(gap_slice)
    
    millor_error = 1e18
    millor_p = hydro_min
    
    for i in range(n_punts):
        p = hydro_min + i * pas
        
        gen = 0.0
        for j in range(n):
            val = gap_slice[j] * f_cuenca
            if val < hydro_min:
                val = hydro_min
            elif val > p:
                val = p
            gen += val
        
        error = abs(gen - energia_obj)
        if error < millor_error:
            millor_error = error
            millor_p = p
    
    return millor_p

# Función para redistribuir exceso (delta positivo), permitiendo usar gaps negativos si hace falta
# def reajustar_por_overload_numpy(
#     hidraulica_actual: np.ndarray, 
#     gap_mes: np.ndarray, 
#     energia_a_repartir: float, 
#     potencia_max: float
# ) -> np.ndarray:
#     """
#     Versión NumPy-optimizada para repartir energía excedente.
#     Reparte una cantidad de energía tratando primero de llenar la capacidad
#     en horas con dèficit (gap > 0), y luego de forma uniforme si es necesario.

#     :param hidraulica_actual: Array NumPy con la generación hidráulica actual del mes.
#     :param gap_mes: Array NumPy con los valores de 'gap' para el mes.
#     :param energia_a_repartir: Exceso de energía a repartir (MWh).
#     :param potencia_max: Potencia máxima por hora.
#     :return: Array NumPy con la generación ajustada.
#     """
#     # Creamos una copia para no modificar el array original fuera de la función
#     hid_ajustada = hidraulica_actual.copy()
#     energia_restante = energia_a_repartir

#     # Iteramos un número fijo de veces para asegurar la convergencia sin bucles infinitos
#     for _ in range(10):  # 10 iteraciones suelen ser suficientes para converger
#         if energia_restante < 1e-6:
#             break

#         # 1. Identificar horas candidatas (aquellas con capacidad disponible)
#         # Usamos una máscara booleana, que es mucho más eficiente
#         capacidad_restante_total = potencia_max - hid_ajustada
#         mascara_candidatas = capacidad_restante_total > 1e-6

#         # Si no hay ninguna hora con capacidad, salimos
#         if not mascara_candidatas.any():
#             break
            
#         # Extraemos los valores solo de las horas candidatas para los cálculos
#         gaps_candidatos = gap_mes[mascara_candidatas]
#         cap_rest_candidatas = capacidad_restante_total[mascara_candidatas]

#         # 2. Calcular los pesos para repartir la energía
#         # Priorizamos horas con gap positivo
#         gaps_positivos = np.clip(gaps_candidatos, 0, None)
#         total_pos = gaps_positivos.sum()

#         if total_pos > 1e-6:
#             # Pesos proporcionales a los gaps positivos
#             pesos = gaps_positivos / total_pos
#         else:
#             # Si no hay gaps positivos, reparto uniforme entre todas las candidatas
#             num_candidatas = len(gaps_candidatos)
#             pesos = np.ones(num_candidatas) / num_candidatas if num_candidatas > 0 else np.array([])
        
#         if pesos.size == 0:
#             break

#         # 3. Calcular cuánta energía añadir en esta iteración
#         # Intentamos repartir la energía restante según los pesos
#         intento_reparto = energia_restante * pesos
        
#         # La cantidad a añadir está limitada por la capacidad restante de cada hora
#         energia_anadir = np.minimum(intento_reparto, cap_rest_candidatas)

#         # 4. Actualizar la generación y la energía restante
#         # Usamos la máscara booleana para actualizar solo las horas candidatas en el array original
#         hid_ajustada[mascara_candidatas] += energia_anadir
        
#         energia_restante -= energia_anadir.sum()

#     return hid_ajustada

from scipy.ndimage import gaussian_filter1d

def suavizar_excedente_rampa(
    hidraulica_actual: np.ndarray, 
    energia_a_repartir: float, 
    hydro_min: float,
    potencia_max: float,
    max_rampa_pct: float = 0.15,
    sigma_suavizado: float = 3.0,
    max_iteraciones: int = 50
) -> np.ndarray:
    """
    Distribueix energia excedent amb suavitzat temporal per evitar "parets".
    
    Combina dues estratègies:
    1. Filtre gaussià per suavitzar el perfil
    2. Restricció de rampa màxima entre hores consecutives
    
    Args:
        hidraulica_actual: Generació hidràulica actual (MWh/h)
        energia_a_repartir: Energia a distribuir (MWh)
        hydro_min: Mínim tècnic de generació (MW)
        potencia_max: Potència màxima per hora (MW)
        max_rampa_pct: Màxim canvi permès entre hores com a fracció de potencia_max
                       (0.15 = 15% de la potència màxima per hora)
        sigma_suavizado: Amplada del filtre gaussià (hores). Més alt = més suau.
        max_iteraciones: Iteracions màximes per convergència
    
    Returns:
        Array amb generació suavitzada
    """
    n = len(hidraulica_actual)
    max_rampa = max_rampa_pct * potencia_max
    
    # Energia objectiu total
    energia_objetivo = hidraulica_actual.sum() + energia_a_repartir
    
    # Partir d'un perfil inicial: el actual més un increment uniforme
    incremento_uniforme = energia_a_repartir / n
    hid_suavizada = hidraulica_actual + incremento_uniforme
    
    for iteracion in range(max_iteraciones):
        hid_anterior = hid_suavizada.copy()
        
        # Pas 1: Aplicar filtre gaussià per suavitzar
        hid_suavizada = gaussian_filter1d(hid_suavizada, sigma=sigma_suavizado, mode='nearest')
        
        # Pas 2: Aplicar restricció de rampa
        for i in range(1, n):
            delta = hid_suavizada[i] - hid_suavizada[i-1]
            if abs(delta) > max_rampa:
                # Limitar el canvi
                hid_suavizada[i] = hid_suavizada[i-1] + np.sign(delta) * max_rampa
        
        # Pas 3: Clip als límits físics
        hid_suavizada = np.clip(hid_suavizada, hydro_min, potencia_max)
        
        # Pas 4: Escalar per mantenir l'energia total
        energia_actual = hid_suavizada.sum()
        if energia_actual > 1e-6:
            factor_escala = energia_objetivo / energia_actual
            hid_suavizada = hid_suavizada * factor_escala
            # Re-clip després d'escalar
            hid_suavizada = np.clip(hid_suavizada, hydro_min, potencia_max)
        
        # Comprovar convergència
        cambio_max = np.abs(hid_suavizada - hid_anterior).max()
        if cambio_max < 0.1:  # Convergit
            break
    
    # Ajust final d'energia (pot haver-hi error petit pel clipping)
    diferencia_energia = energia_objetivo - hid_suavizada.sum()
    if abs(diferencia_energia) > 1e-3:
        # Repartir la diferència uniformement on hi hagi marge
        if diferencia_energia > 0:
            marge = potencia_max - hid_suavizada
        else:
            marge = hid_suavizada - hydro_min
        
        marge_total = marge.sum()
        if marge_total > 1e-6:
            ajust = diferencia_energia * (marge / marge_total)
            hid_suavizada += ajust
            hid_suavizada = np.clip(hid_suavizada, hydro_min, potencia_max)
    
    return hid_suavizada


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


def _crear_lookup_energia(serie_mensual: pd.Series) -> dict:
    """
    Converteix una sèrie mensual en un diccionari {(year, month): valor}.
    Accés O(1) en lloc de O(n).
    """
    return {
        (idx.year, idx.month): valor 
        for idx, valor in serie_mensual.items()
    }

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



def suavizar_excedente_waterfill(
    hidraulica_actual: np.ndarray, 
    energia_a_repartir: float, 
    hydro_min: float,
    potencia_max: float,
    warn_no_repartida: bool = True
) -> np.ndarray:
    """
    Distribueix energia excedent usant un algoritme water-fill real.
    
    L'objectiu és suavitzar el perfil de generació, omplint primer les hores
    amb menor generació fins a nivelar-les amb les següents, com si s'omplís
    un recipient amb aigua.
    
    Args:
        hidraulica_actual: Generació hidràulica actual (MWh/h)
        energia_a_repartir: Energia exacta a distribuir (MWh)
        hydro_min: Mínim tècnic de generació (MW)
        potencia_max: Potència màxima per hora (MW)
        warn_no_repartida: Si True, avisa si queda energia sense repartir
    
    Returns:
        Array amb generació ajustada i suavitzada
    """
    hid_ajustada = hidraulica_actual.copy()
    n = len(hid_ajustada)
    
    # Assegurar mínim tècnic (i comptabilitzar l'energia afegida)
    deficit_minim = np.maximum(hydro_min - hid_ajustada, 0)
    energia_para_minimo = deficit_minim.sum()
    
    if energia_para_minimo > 0:
        hid_ajustada = np.maximum(hid_ajustada, hydro_min)
        # Nota: aquesta energia s'afegeix "gratis" perquè és obligatòria
        # Si vols ser estricte, podries descomptar-la de energia_a_repartir
    
    energia_restante = energia_a_repartir
    
    # Water-fill: ordenar per nivell i omplir progressivament
    while energia_restante > 1e-6:
        # Trobar el nivell actual més baix entre les no saturades
        capacidad = potencia_max - hid_ajustada
        no_saturadas = capacidad > 1e-6
        
        if not no_saturadas.any():
            break
        
        niveles = hid_ajustada.copy()
        niveles[~no_saturadas] = np.inf  # Ignorar saturades
        
        nivel_min = niveles.min()
        
        # Trobar totes les hores al nivell mínim
        en_nivel_min = np.abs(hid_ajustada - nivel_min) < 1e-6
        en_nivel_min &= no_saturadas  # Només les no saturades
        num_en_min = en_nivel_min.sum()
        
        if num_en_min == 0:
            break
        
        # Trobar el següent nivell (per saber fins on omplir)
        niveles_superiors = niveles[niveles > nivel_min + 1e-6]
        if len(niveles_superiors) > 0:
            siguiente_nivel = niveles_superiors.min()
        else:
            siguiente_nivel = potencia_max
        
        # Calcular quanta energia cal per pujar totes les del nivell mínim
        # fins al següent nivell
        diferencia_nivel = siguiente_nivel - nivel_min
        energia_para_nivelar = diferencia_nivel * num_en_min
        
        # Quanta energia podem realment afegir?
        energia_a_usar = min(energia_restante, energia_para_nivelar)
        incremento_por_hora = energia_a_usar / num_en_min
        
        # Aplicar, respectant potencia_max
        incremento_real = np.minimum(
            incremento_por_hora, 
            capacidad[en_nivel_min]
        )
        hid_ajustada[en_nivel_min] += incremento_real
        energia_restante -= incremento_real.sum()
    
    # Warning si queda energia
    if warn_no_repartida and energia_restante > 1e-3:
        import warnings
        pct = energia_restante / energia_a_repartir * 100
        warnings.warn(
            f"suavizar_excedente: {energia_restante:.1f} MWh no repartits "
            f"({pct:.1f}%). Totes les hores saturades."
        )
    
    return hid_ajustada


def reajustar_por_overload_suavizado(
    hidraulica_actual: np.ndarray, 
    gap_mes: np.ndarray, 
    energia_a_repartir: float, 
    potencia_max: float,
    max_rampa_pct: float = 0.15
) -> np.ndarray:
    """
    Distribueix energia excedent en situació d'overload mantenint
    continuïtat temporal per evitar "parets" de generació.
    
    Estratègia:
    1. Crea un "perfil objectiu" basat en el gap (on hi ha més dèficit, més generació)
    2. Suavitza aquest perfil per garantir transicions graduals
    3. Escala per encaixar l'energia total requerida
    
    Args:
        hidraulica_actual: Generació hidràulica actual (MWh/h)
        gap_mes: Dèficit horari del mes (demanda - generació)
        energia_a_repartir: Energia extra a distribuir per overload (MWh)
        potencia_max: Potència màxima per hora (MW)
        max_rampa_pct: Màxim canvi permès entre hores (fracció de potencia_max)
    
    Returns:
        Array amb generació ajustada i suavitzada
    """
    n = len(hidraulica_actual)
    max_rampa = max_rampa_pct * potencia_max
    hid_ajustada = hidraulica_actual.copy()
    
    # Calcular capacitat disponible per hora
    capacidad_disponible = potencia_max - hid_ajustada
    capacidad_disponible = np.maximum(capacidad_disponible, 0)
    
    capacidad_total = capacidad_disponible.sum()
    if capacidad_total < 1e-6 or energia_a_repartir < 1e-6:
        return hid_ajustada
    
    # === ESTRATÈGIA: Perfil de distribució basat en gap suavitzat ===
    
    # Pas 1: Crear un perfil de "prioritat" basat en el gap
    # Gap positiu = més prioritat, però no volem salts bruscos
    prioridad_bruta = gap_mes.copy()
    
    # Pas 2: Suavitzar el perfil de prioritat amb mitjana mòbil
    ventana = min(7, n // 10)  # Finestra adaptativa
    if ventana >= 3:
        kernel = np.ones(ventana) / ventana
        prioridad_suavizada = np.convolve(prioridad_bruta, kernel, mode='same')
    else:
        prioridad_suavizada = prioridad_bruta
    
    # Pas 3: Convertir prioritat en increment desitjat (normalitzat)
    # Desplacem perquè el mínim sigui 0 i tot positiu
    prioridad_suavizada = prioridad_suavizada - prioridad_suavizada.min()
    
    # Afegir una base uniforme per evitar zeros (tots reben alguna cosa)
    prioridad_suavizada = prioridad_suavizada + prioridad_suavizada.mean() * 0.3
    
    suma_prioridad = prioridad_suavizada.sum()
    if suma_prioridad < 1e-6:
        # Fallback a distribució uniforme
        incremento_deseado = np.full(n, energia_a_repartir / n)
    else:
        incremento_deseado = (prioridad_suavizada / suma_prioridad) * energia_a_repartir
    
    # Pas 4: Limitar per capacitat disponible
    incremento_limitado = np.minimum(incremento_deseado, capacidad_disponible)
    
    # Pas 5: Aplicar restricció de rampa al perfil resultant
    hid_propuesta = hid_ajustada + incremento_limitado
    
    # Forward pass
    for i in range(1, n):
        delta = hid_propuesta[i] - hid_propuesta[i-1]
        if delta > max_rampa:
            hid_propuesta[i] = hid_propuesta[i-1] + max_rampa
        elif delta < -max_rampa:
            hid_propuesta[i] = hid_propuesta[i-1] - max_rampa
    
    # Backward pass (per simetria)
    for i in range(n-2, -1, -1):
        delta = hid_propuesta[i] - hid_propuesta[i+1]
        if delta > max_rampa:
            hid_propuesta[i] = hid_propuesta[i+1] + max_rampa
        elif delta < -max_rampa:
            hid_propuesta[i] = hid_propuesta[i+1] - max_rampa
    
    # Pas 6: Assegurar que no baixem del nivell actual ni superem el màxim
    hid_propuesta = np.maximum(hid_propuesta, hid_ajustada)  # No reduir
    hid_propuesta = np.minimum(hid_propuesta, potencia_max)  # No superar màxim
    
    # Pas 7: Escalar per ajustar a l'energia objectiu
    energia_afegida = hid_propuesta.sum() - hid_ajustada.sum()
    
    if energia_afegida < energia_a_repartir * 0.99:
        # Encara queda energia per repartir, fer una segona passada
        energia_restant = energia_a_repartir - energia_afegida
        capacidad_restante = potencia_max - hid_propuesta
        capacidad_restante = np.maximum(capacidad_restante, 0)
        
        cap_total_restant = capacidad_restante.sum()
        if cap_total_restant > 1e-6:
            # Repartir uniformement el restant (ja no queda més remei)
            incremento_extra = energia_restant * (capacidad_restante / cap_total_restant)
            incremento_extra = np.minimum(incremento_extra, capacidad_restante)
            hid_propuesta += incremento_extra
    
    return hid_propuesta


@njit(cache=True, fastmath=True)
def _convolve_same_numba(arr: np.ndarray, ventana: int) -> np.ndarray:
    """
    Replica exactament np.convolve(arr, kernel, mode='same')
    amb kernel uniforme de mida 'ventana'.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    half = ventana // 2
    
    for i in range(n):
        # np.convolve amb mode='same' centra el kernel
        suma = 0.0
        count = 0
        for j in range(ventana):
            idx = i - half + j
            if 0 <= idx < n:
                suma += arr[idx]
                count += 1
        # np.convolve sempre divideix per ventana, no per count
        result[i] = suma / ventana
    
    return result


@njit(cache=True, fastmath=True)
def _aplicar_rampa_bidireccional(hid: np.ndarray, max_rampa: float) -> np.ndarray:
    """Aplica restricció de rampa en ambdues direccions."""
    n = len(hid)
    
    for i in range(1, n):
        delta = hid[i] - hid[i-1]
        if delta > max_rampa:
            hid[i] = hid[i-1] + max_rampa
        elif delta < -max_rampa:
            hid[i] = hid[i-1] - max_rampa
    
    for i in range(n - 2, -1, -1):
        delta = hid[i] - hid[i+1]
        if delta > max_rampa:
            hid[i] = hid[i+1] + max_rampa
        elif delta < -max_rampa:
            hid[i] = hid[i+1] - max_rampa
    
    return hid


@njit(cache=True, fastmath=True)
def _core_overload_suavizado_v2(
    hidraulica_actual: np.ndarray,
    gap_mes: np.ndarray,
    energia_a_repartir: float,
    potencia_max: float,
    max_rampa: float,
    ventana: int
) -> np.ndarray:
    """
    Nucli optimitzat - versió corregida per ser equivalent a l'original.
    """
    n = len(hidraulica_actual)
    hid_ajustada = hidraulica_actual.copy()
    
    # Calcular capacitat disponible
    capacidad_disponible = np.empty(n, dtype=np.float64)
    capacidad_total = 0.0
    for i in range(n):
        cap = potencia_max - hid_ajustada[i]
        if cap < 0:
            cap = 0.0
        capacidad_disponible[i] = cap
        capacidad_total += cap
    
    if capacidad_total < 1e-6 or energia_a_repartir < 1e-6:
        return hid_ajustada
    
    # === Pas 1: Crear prioritat suavitzada (EXACTAMENT com l'original) ===
    prioridad_bruta = gap_mes.copy()
    
    # Suavitzar amb convolve equivalent
    if ventana >= 3:
        prioridad_suavizada = _convolve_same_numba(prioridad_bruta, ventana)
    else:
        prioridad_suavizada = prioridad_bruta.copy()
    
    # Desplaçar perquè mínim sigui 0
    min_val = prioridad_suavizada[0]
    for i in range(1, n):
        if prioridad_suavizada[i] < min_val:
            min_val = prioridad_suavizada[i]
    
    for i in range(n):
        prioridad_suavizada[i] = prioridad_suavizada[i] - min_val
    
    # Afegir base uniforme (mean calculat DESPRÉS de restar min, com l'original)
    mean_val = 0.0
    for i in range(n):
        mean_val += prioridad_suavizada[i]
    mean_val /= n
    
    suma_prior = 0.0
    for i in range(n):
        prioridad_suavizada[i] = prioridad_suavizada[i] + mean_val * 0.3
        suma_prior += prioridad_suavizada[i]
    
    # === Pas 2: Calcular increment desitjat ===
    if suma_prior < 1e-6:
        inc_uniforme = energia_a_repartir / n
        for i in range(n):
            prioridad_suavizada[i] = inc_uniforme
    else:
        for i in range(n):
            prioridad_suavizada[i] = (prioridad_suavizada[i] / suma_prior) * energia_a_repartir
    
    # Limitar per capacitat disponible
    for i in range(n):
        if prioridad_suavizada[i] > capacidad_disponible[i]:
            prioridad_suavizada[i] = capacidad_disponible[i]
    
    # === Pas 3: Crear perfil proposat ===
    hid_propuesta = np.empty(n, dtype=np.float64)
    for i in range(n):
        hid_propuesta[i] = hid_ajustada[i] + prioridad_suavizada[i]
    
    # === Pas 4: Aplicar restricció de rampa ===
    hid_propuesta = _aplicar_rampa_bidireccional(hid_propuesta, max_rampa)
    
    # === Pas 5: Clip als límits ===
    for i in range(n):
        if hid_propuesta[i] < hid_ajustada[i]:
            hid_propuesta[i] = hid_ajustada[i]
        if hid_propuesta[i] > potencia_max:
            hid_propuesta[i] = potencia_max
    
    # === Pas 6: Repartir energia restant ===
    energia_afegida = 0.0
    for i in range(n):
        energia_afegida += hid_propuesta[i] - hid_ajustada[i]
    
    energia_restant = energia_a_repartir - energia_afegida
    
    if energia_restant > energia_a_repartir * 0.01:
        cap_total_restant = 0.0
        for i in range(n):
            cap = potencia_max - hid_propuesta[i]
            if cap < 0:
                cap = 0.0
            capacidad_disponible[i] = cap
            cap_total_restant += cap
        
        if cap_total_restant > 1e-6:
            for i in range(n):
                extra = energia_restant * (capacidad_disponible[i] / cap_total_restant)
                if extra > capacidad_disponible[i]:
                    extra = capacidad_disponible[i]
                hid_propuesta[i] += extra
    
    return hid_propuesta


def reajustar_por_overload_suavizado_optimizado(
    hidraulica_actual: np.ndarray, 
    gap_mes: np.ndarray, 
    energia_a_repartir: float, 
    potencia_max: float,
    max_rampa_pct: float = 0.15,
    ventana_suavizado: int = None  # None = calcular dinàmicament com l'original
) -> np.ndarray:
    """
    Distribueix energia excedent en situació d'overload mantenint
    continuïtat temporal. Versió optimitzada amb Numba.
    
    Aquesta versió és funcionalment equivalent a reajustar_por_overload_suavizado.
    
    Args:
        hidraulica_actual: Generació hidràulica actual (MWh/h)
        gap_mes: Dèficit horari del mes
        energia_a_repartir: Energia extra a distribuir (MWh)
        potencia_max: Potència màxima per hora (MW)
        max_rampa_pct: Màxim canvi entre hores (fracció de potencia_max)
        ventana_suavizado: Amplada finestra. Si None, usa min(7, n//10) com l'original.
    
    Returns:
        np.ndarray: Generació ajustada i suavitzada (mateix format que l'original)
    """
    n = len(hidraulica_actual)
    max_rampa = max_rampa_pct * potencia_max
    
    # Calcular ventana dinàmicament si no s'especifica (com l'original)
    if ventana_suavizado is None:
        ventana = min(7, n // 10)
    else:
        ventana = ventana_suavizado
    
    # Assegurar tipus correctes per Numba
    hid = np.ascontiguousarray(hidraulica_actual, dtype=np.float64)
    gap = np.ascontiguousarray(gap_mes, dtype=np.float64)
    
    return _core_overload_suavizado_v2(
        hid, gap,
        float(energia_a_repartir),
        float(potencia_max),
        float(max_rampa),
        int(ventana)
    )

@njit(cache=True, fastmath=True)
def _gaussian_blur_1d_numba(arr: np.ndarray, sigma: float) -> np.ndarray:
    """
    Aproximació eficient del filtre gaussià 1D.
    Usa kernel truncat a 3*sigma per velocitat.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    
    # Mida del kernel (truncat a 3 sigmes)
    radius = int(3.0 * sigma + 0.5)
    if radius < 1:
        radius = 1
    
    # Pre-calcular pesos gaussians
    kernel_size = 2 * radius + 1
    kernel = np.empty(kernel_size, dtype=np.float64)
    kernel_sum = 0.0
    
    for i in range(kernel_size):
        x = i - radius
        weight = np.exp(-0.5 * (x / sigma) ** 2)
        kernel[i] = weight
        kernel_sum += weight
    
    # Normalitzar kernel
    for i in range(kernel_size):
        kernel[i] /= kernel_sum
    
    # Aplicar convolució
    for i in range(n):
        suma = 0.0
        for k in range(kernel_size):
            idx = i + k - radius
            # Mode 'nearest' - replicar extrems
            if idx < 0:
                idx = 0
            elif idx >= n:
                idx = n - 1
            suma += arr[idx] * kernel[k]
        result[i] = suma
    
    return result


@njit(cache=True, fastmath=True)
def _aplicar_rampa_forward(hid: np.ndarray, max_rampa: float) -> None:
    """Aplica restricció de rampa només forward, in-place."""
    n = len(hid)
    for i in range(1, n):
        delta = hid[i] - hid[i-1]
        if delta > max_rampa:
            hid[i] = hid[i-1] + max_rampa
        elif delta < -max_rampa:
            hid[i] = hid[i-1] - max_rampa


@njit(cache=True, fastmath=True)
def _core_suavizar_excedente_rampa(
    hidraulica_actual: np.ndarray,
    energia_a_repartir: float,
    hydro_min: float,
    potencia_max: float,
    max_rampa: float,
    sigma: float,
    max_iteraciones: int,
    tolerancia_convergencia: float
) -> np.ndarray:
    """
    Nucli optimitzat per suavitzar excedent amb restricció de rampa.
    """
    n = len(hidraulica_actual)
    
    # Energia objectiu total
    energia_objetivo = 0.0
    for i in range(n):
        energia_objetivo += hidraulica_actual[i]
    energia_objetivo += energia_a_repartir
    
    # Partir d'un perfil inicial: actual + increment uniforme
    incremento_uniforme = energia_a_repartir / n
    hid_suavizada = np.empty(n, dtype=np.float64)
    for i in range(n):
        hid_suavizada[i] = hidraulica_actual[i] + incremento_uniforme
    
    # Array temporal per comparar convergència
    hid_anterior = np.empty(n, dtype=np.float64)
    
    for iteracion in range(max_iteraciones):
        # Guardar estat anterior
        for i in range(n):
            hid_anterior[i] = hid_suavizada[i]
        
        # Pas 1: Aplicar filtre gaussià
        hid_suavizada = _gaussian_blur_1d_numba(hid_suavizada, sigma)
        
        # Pas 2: Aplicar restricció de rampa (forward only per velocitat)
        _aplicar_rampa_forward(hid_suavizada, max_rampa)
        
        # Pas 3: Clip als límits físics
        for i in range(n):
            if hid_suavizada[i] < hydro_min:
                hid_suavizada[i] = hydro_min
            elif hid_suavizada[i] > potencia_max:
                hid_suavizada[i] = potencia_max
        
        # Pas 4: Escalar per mantenir energia total
        energia_actual = 0.0
        for i in range(n):
            energia_actual += hid_suavizada[i]
        
        if energia_actual > 1e-6:
            factor_escala = energia_objetivo / energia_actual
            for i in range(n):
                hid_suavizada[i] *= factor_escala
            
            # Re-clip després d'escalar
            for i in range(n):
                if hid_suavizada[i] < hydro_min:
                    hid_suavizada[i] = hydro_min
                elif hid_suavizada[i] > potencia_max:
                    hid_suavizada[i] = potencia_max
        
        # Comprovar convergència
        cambio_max = 0.0
        for i in range(n):
            diff = hid_suavizada[i] - hid_anterior[i]
            if diff < 0:
                diff = -diff
            if diff > cambio_max:
                cambio_max = diff
        
        if cambio_max < tolerancia_convergencia:
            break
    
    # Ajust final d'energia
    energia_final = 0.0
    for i in range(n):
        energia_final += hid_suavizada[i]
    
    diferencia_energia = energia_objetivo - energia_final
    
    if diferencia_energia > 1e-3 or diferencia_energia < -1e-3:
        # Calcular marge disponible
        if diferencia_energia > 0:
            # Cal afegir energia
            marge_total = 0.0
            for i in range(n):
                marge = potencia_max - hid_suavizada[i]
                if marge < 0:
                    marge = 0.0
                marge_total += marge
            
            if marge_total > 1e-6:
                for i in range(n):
                    marge = potencia_max - hid_suavizada[i]
                    if marge < 0:
                        marge = 0.0
                    ajust = diferencia_energia * (marge / marge_total)
                    hid_suavizada[i] += ajust
                    if hid_suavizada[i] > potencia_max:
                        hid_suavizada[i] = potencia_max
        else:
            # Cal treure energia
            marge_total = 0.0
            for i in range(n):
                marge = hid_suavizada[i] - hydro_min
                if marge < 0:
                    marge = 0.0
                marge_total += marge
            
            if marge_total > 1e-6:
                diferencia_abs = -diferencia_energia
                for i in range(n):
                    marge = hid_suavizada[i] - hydro_min
                    if marge < 0:
                        marge = 0.0
                    ajust = diferencia_abs * (marge / marge_total)
                    hid_suavizada[i] -= ajust
                    if hid_suavizada[i] < hydro_min:
                        hid_suavizada[i] = hydro_min
    
    return hid_suavizada


def suavizar_excedente_rampa_optimizado(
    hidraulica_actual: np.ndarray, 
    energia_a_repartir: float, 
    hydro_min: float,
    potencia_max: float,
    max_rampa_pct: float = 0.15,
    sigma_suavizado: float = 3.0,
    max_iteraciones: int = 50
) -> np.ndarray:
    """
    Distribueix energia excedent amb suavitzat temporal per evitar "parets".
    Versió optimitzada amb Numba - funcionalment equivalent a l'original.
    
    Args:
        hidraulica_actual: Generació hidràulica actual (MWh/h)
        energia_a_repartir: Energia a distribuir (MWh)
        hydro_min: Mínim tècnic de generació (MW)
        potencia_max: Potència màxima per hora (MW)
        max_rampa_pct: Màxim canvi permès entre hores (fracció de potencia_max)
        sigma_suavizado: Amplada del filtre gaussià (hores)
        max_iteraciones: Iteracions màximes per convergència
    
    Returns:
        np.ndarray: Generació suavitzada (intercanviable amb l'original)
    """
    max_rampa = max_rampa_pct * potencia_max
    
    # Assegurar tipus correctes per Numba
    hid = np.ascontiguousarray(hidraulica_actual, dtype=np.float64)
    
    return _core_suavizar_excedente_rampa(
        hid,
        float(energia_a_repartir),
        float(hydro_min),
        float(potencia_max),
        float(max_rampa),
        float(sigma_suavizado),
        int(max_iteraciones),
        0.1  # tolerancia_convergencia
    )

def remove_restrictions_seasonal(
    base_level: pd.Series,
    consumo_base_diario_hm3: float = 2.05,  # hm3/dia (Mitjana anual)
    max_capacity_int: float = 693,
    umbrales_sequia: dict = None,
    pesos_sectores: dict = None,
    patrones_estacionales: pd.DataFrame = None,
    restricciones_sectoriales: pd.DataFrame = None
) -> tuple[pd.Series, pd.Series]:
    """
    Calcula l'estalvi d'aigua i reconstrueix el nivell 'natural' dels embassaments
    incorporant l'ESTACIONALITAT de la demanda i les restriccions SECTORIALS.

    Args:
        base_level (pd.Series): Nivell dels embassaments en % (sèrie temporal).
        consumo_base_diario_hm3 (float): Consum mitjà anual de referència (hm³/dia).
        max_capacity_int (float): Capacitat total dels embassaments interns en hm³.
        umbrales_sequia (dict, optional): Llindars de % per a cada fase.
        pesos_sectores (dict, optional): Pes de cada sector sobre el total (0-1).
                                         Ex: {'Urbà': 0.6, 'Regadiu': 0.3, ...}
        patrones_estacionales (pd.DataFrame, optional): Coeficients mensuals (Index 1-12).
        restricciones_sectoriales (pd.DataFrame, optional): % restricció per Fase i Sector.

    Returns:
        tuple[pd.Series, pd.Series]: 
            - Sèrie amb l'estalvi acumulat en hm³.
            - Sèrie amb el nivell simulat (%) sense restriccions.
    """
    # Consum estacional
    # consumo_base_diario_estacional_hm3 = np.array([
    #     1.37, 1.37, 1.42, 1.74, 2.32, 3.03, 
    #     3.51, 3.24, 2.36, 1.48, 1.37, 1.37
    # ])

    # --- 1. Definició de Dades per Defecte (Basades en el Pla de Gestió) ---
    
    if umbrales_sequia is None:
        umbrales_sequia = {
            'Prealerta': 60.0, 'Alerta': 40.0, 'Excepcionalitat': 25.0,
            'Emergencia': 16.0, 'Emergencia_2': 11.0, 'Emergencia_3': 5.5
        }

    # Pesos estimats
    # He posat uns valors orientatius perquè sumin 1.0
    if pesos_sectores is None:
        pesos_sectores = {
            'Urbà': 0.415, #0.544, #excloc les pèrdues
            'Regadiu': 0.344, 
            'Ramaderia': 0.02, 
            'Ind_Bens': 0.0785, 
            'Ind_Turisme': 0.0135
        }

    # Taula de coeficients mensuals
    if patrones_estacionales is None:
        data_patrons = {
            'Urbà': [0.96, 0.96, 0.96, 0.96, 1.08, 1.08, 1.08, 1.08, 0.96, 0.96, 0.96, 0.96],
            'Regadiu': [0.13, 0.13, 0.19, 0.63, 1.26, 2.26, 2.89, 2.51, 1.50, 0.25, 0.13, 0.13],
            'Ramaderia': [1.0]*12,
            'Ind_Bens': [1.0]*12,
            'Ind_Turisme': [0.51, 0.51, 0.60, 0.69, 0.94, 1.54, 2.15, 2.15, 1.28, 0.69, 0.51, 0.43]
        }
        patrones_estacionales = pd.DataFrame(data_patrons, index=range(1, 13)) # Index 1=Gener

    # Taula de restriccions (La del PES que m'has passat)
    if restricciones_sectoriales is None:
        # Definir les reduccions per fase (0.25 = 25% reducció)
        # Nota: He assumit 0% per Normalitat i Prealerta per simplificar, ajusta si cal.
        data_restr = {
            'Urbà':        {'Normalitat':0, 'Prealerta':0.025, 'Alerta':0.05, 'Excepcionalitat':0.075, 'Emergencia':0.10, 'Emergencia_2':0.12, 'Emergencia_3':0.14}, # Ajustar restricció urbana
            'Regadiu':     {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.25, 'Excepcionalitat':0.40, 'Emergencia':0.80, 'Emergencia_2':0.80, 'Emergencia_3':0.80},
            'Ramaderia':   {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.10, 'Excepcionalitat':0.30, 'Emergencia':0.50, 'Emergencia_2':0.50, 'Emergencia_3':0.50},
            'Ind_Bens':    {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.05, 'Excepcionalitat':0.15, 'Emergencia':0.25, 'Emergencia_2':0.25, 'Emergencia_3':0.25},
            'Ind_Turisme': {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.05, 'Excepcionalitat':0.15, 'Emergencia':0.25, 'Emergencia_2':0.25, 'Emergencia_3':0.25}
        }
        
        # Transposem perquè l'índex siguin les fases
        restricciones_sectoriales = pd.DataFrame(data_restr).T
        # Assegurem que les columnes de fases coincideixin amb els noms que fem servir
        # (Aquí caldria assegurar que el df tingui com a columnes 'Normalitat', 'Alerta', etc.)
        # Per simplicitat en l'exemple, ho tractarem com un diccionari de diccionaris a sota.

    # --- 2. Preparació de la Sèrie Temporal ---

    # Resamplejar a diari (ffill per omplir forats)
    df_diari = pd.DataFrame({'nivel_pct': base_level}).resample('D').ffill().ffill()
    
    # Determinar el Mes (per estacionalitat) i la Fase (per restricció)
    df_diari['mes'] = df_diari.index.month
    
    def get_fase_sequia(nivel):
        if pd.isna(nivel): return 'Normalitat'
        if nivel < umbrales_sequia['Emergencia_3']: return 'Emergencia_3'
        if nivel < umbrales_sequia['Emergencia_2']: return 'Emergencia_2'
        if nivel < umbrales_sequia['Emergencia']: return 'Emergencia' # Simplifico EM1,2,3 a 'Emergencia' per l'exemple
        if nivel < umbrales_sequia['Excepcionalitat']: return 'Excepcionalitat'
        if nivel < umbrales_sequia['Alerta']: return 'Alerta'
        if nivel < umbrales_sequia['Prealerta']: return 'Prealerta'
        return 'Normalitat'

    df_diari['fase'] = df_diari['nivel_pct'].apply(get_fase_sequia)

    # --- 3. Càlcul de l'Estalvi Sectorial (El nucli de la funció) ---
    
    # Inicialitzem l'estalvi diari total a 0
    df_diari['ahorro_diario_hm3'] = 0.0

    # Iterem per cada sector (Urbà, Reg, etc.)
    for sector, pes in pesos_sectores.items():
        
        # A. Volum Base del sector (Càrrega base plana)
        volum_base_sector = consumo_base_diario_hm3 * pes
        
        # B. Sèrie de Coeficients Estacionals per a aquest sector (segons el mes del dia)
        # map: agafa la columna 'mes' (1..12) i li assigna el valor del df patrones
        coef_estacional = df_diari['mes'].map(patrones_estacionales[sector])
        
        # C. Sèrie de Restriccions per a aquest sector (segons la fase del dia)
        # map: agafa la fase i busca el valor al diccionari de restriccions
        # (Si el df de restriccions és complex, el convertim a dict per fer map ràpid)
        restr_dict = restricciones_sectoriales.loc[sector] if isinstance(restricciones_sectoriales, pd.DataFrame) else data_restr[sector]
        factor_restriccio = df_diari['fase'].map(restr_dict).fillna(0.0) # 0.0 vol dir sense restricció
        
        # D. Càlcul de l'estalvi d'aquest sector
        # Estalvi = Demanda_Potencial * %_Restricció
        # Demanda_Potencial = Base * Coef_Estacional
        estalvi_sector = (volum_base_sector * coef_estacional) * factor_restriccio
        
        # Sumem a l'estalvi total
        df_diari['ahorro_diario_hm3'] += estalvi_sector

    # --- 4. Resultats i Retorn ---

    # Acumulem l'estalvi al llarg del temps
    df_diari['ahorro_acumulado_hm3'] = df_diari['ahorro_diario_hm3'].cumsum()

    # Reindexar per coincidir amb l'índex original (si no era diari)
    ahorro_final_acumulado = df_diari['ahorro_acumulado_hm3'].reindex(base_level.index, method='ffill').fillna(0)
    # ahorro_final_acumulado = (df_diari['ahorro_acumulado_hm3'].reindex(base_level.index).interpolate(method='linear').fillna(0))


    # Calcular nivell simulat sense restriccions
    # Si hem estalviat aigua, vol dir que sense restriccions el nivell seria MÉS BAIX.
    pct_per_hm3 = 100.0 / max_capacity_int
    ahorro_acumulado_pct = ahorro_final_acumulado * pct_per_hm3
    nivel_simulado = base_level - ahorro_acumulado_pct

    return ahorro_final_acumulado, nivel_simulado

def calcular_estalvi_pct(
    mes: int,
    nivel_pct: float,
    umbrales_sequia: dict = None,
    restricciones_sectoriales: dict = None
) -> float:
    """
    Retorna el percentatge d'estalvi hídric esperat donats el mes i el nivell.
    
    Args:
        mes: Mes de l'any (1-12)
        nivel_pct: Nivell actual dels embassaments (%)
        umbrales_sequia: Llindars de cada fase (parametritzable)
        restricciones_sectoriales: % restricció per sector i fase (parametritzable)
    
    Returns:
        float: Fracció d'estalvi sobre el consum base (0.0 - 1.0)
    """
    
    # === PARÀMETRES FIXOS (estructura del sistema) ===
    
    PESOS_SECTORES = {
        'Urba': 0.415,
        'Regadiu': 0.344,
        'Ramaderia': 0.02,
        'Ind_Bens': 0.0785,
        'Ind_Turisme': 0.0135
    }
    
    PATRONES_ESTACIONALES = {
        'Urba':        [0.96, 0.96, 0.96, 0.96, 1.08, 1.08, 1.08, 1.08, 0.96, 0.96, 0.96, 0.96],
        'Regadiu':     [0.13, 0.13, 0.19, 0.63, 1.26, 2.26, 2.89, 2.51, 1.50, 0.25, 0.13, 0.13],
        'Ramaderia':   [1.0]*12,
        'Ind_Bens':    [1.0]*12,
        'Ind_Turisme': [0.51, 0.51, 0.60, 0.69, 0.94, 1.54, 2.15, 2.15, 1.28, 0.69, 0.51, 0.43]
    }
    
    # === PARÀMETRES CONFIGURABLES (política) ===
    
    if umbrales_sequia is None:
        umbrales_sequia = {
            'Emergencia_3': 5.5, 'Emergencia_2': 11.0, 'Emergencia': 16.0,
            'Excepcionalitat': 25.0, 'Alerta': 40.0, 'Prealerta': 60.0
        }
    
    if restricciones_sectoriales is None:
        restricciones_sectoriales = {
            'Urba':        {'Normalitat':0, 'Prealerta':0.025, 'Alerta':0.05, 'Excepcionalitat':0.075, 'Emergencia':0.10, 'Emergencia_2':0.12, 'Emergencia_3':0.14},
            'Regadiu':     {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.25, 'Excepcionalitat':0.40, 'Emergencia':0.80, 'Emergencia_2':0.80, 'Emergencia_3':0.80},
            'Ramaderia':   {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.10, 'Excepcionalitat':0.30, 'Emergencia':0.50, 'Emergencia_2':0.50, 'Emergencia_3':0.50},
            'Ind_Bens':    {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.05, 'Excepcionalitat':0.15, 'Emergencia':0.25, 'Emergencia_2':0.25, 'Emergencia_3':0.25},
            'Ind_Turisme': {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.05, 'Excepcionalitat':0.15, 'Emergencia':0.25, 'Emergencia_2':0.25, 'Emergencia_3':0.25}
        }
    
    # === CÀLCUL ===
    
    # 1. Determinar fase de sequera
    if nivel_pct < umbrales_sequia['Emergencia_3']:
        fase = 'Emergencia_3'
    elif nivel_pct < umbrales_sequia['Emergencia_2']:
        fase = 'Emergencia_2'
    elif nivel_pct < umbrales_sequia['Emergencia']:
        fase = 'Emergencia'
    elif nivel_pct < umbrales_sequia['Excepcionalitat']:
        fase = 'Excepcionalitat'
    elif nivel_pct < umbrales_sequia['Alerta']:
        fase = 'Alerta'
    elif nivel_pct < umbrales_sequia['Prealerta']:
        fase = 'Prealerta'
    else:
        fase = 'Normalitat'
    
    # 2. Calcular estalvi ponderat per sectors
    estalvi_total = 0.0
    idx_mes = mes - 1  # Convertir 1-12 a índex 0-11
    
    for sector, pes in PESOS_SECTORES.items():
        coef_estacional = PATRONES_ESTACIONALES[sector][idx_mes]
        pct_restriccio = restricciones_sectoriales[sector].get(fase, 0.0)
        
        # Estalvi = pes_sector × demanda_estacional × restricció
        estalvi_sector = pes * coef_estacional * pct_restriccio
        estalvi_total += estalvi_sector
    
    return estalvi_total


def sigmoid_factor(
    level_pct: float,
    midpoint: float = 75.0,
    steepness: float = 0.2
) -> float:
    """
    Mapea un nivel (%) a un factor [0,1] siguiendo una sigmoide centrada en midpoint:
      - Para level_pct ≪ midpoint → factor ≈ 1
      - Para level_pct ≫ midpoint → factor ≈ 0
    steepness controla la pendiente en torno al midpoint.
    """
    x = (level_pct - midpoint) * steepness
    return 1.0 / (1.0 + np.exp(x))

@njit
def sigmoid_factor_numba(
    level_pct: float,
    midpoint: float = 75.0,
    steepness: float = 0.2,
    f_min: float = 0.18,
) -> float:
    """
    Versión optimizada con Numba de sigmoid_factor.
    Inclou mínim tècnic
    """
    x = (level_pct - midpoint) * steepness
    f_base = 1.0 / (1.0 + np.exp(x))
    return f_min + (1.0 - f_min) * f_base  # Rang: [f_min, 1.0]

# Versión completamente optimizada para arrays
@njit
def seasonal_factor_array(
    months: np.ndarray,
    days: np.ndarray,
    days_in_months: np.ndarray,
    phase_months: float = 0.0,
    amplitude: float = 0.2
) -> np.ndarray:
    """
    Versión vectorizada para procesar arrays completos.
    """
    result = np.empty_like(months, dtype=np.float64)
    for i in range(len(months)):
        month_decimal = months[i] + (days[i] - 1) / days_in_months[i]
        angle = 2 * np.pi * (month_decimal - phase_months) / 12.0
        result[i] = 1.0 - amplitude * (1 + np.cos(angle)) / 2
    return result

# def simulate_full_water_management(
#     surpluses: pd.Series,
#     level_base: pd.Series,
#     thermal_generation: pd.Series,  # NUEVO: Generación térmica disponible
#     base_hydro_generation: pd.Series,  # NUEVO: Generación hidro base
#     max_capacity_int: float,
#     consumo_base_diario_hm3: float,
#     # Parámetros de desalación
#     max_desal_mw: float = 30,
#     min_run_hours: int = 4,
#     midpoint: float = 75,
#     steepness: float = 0.2,
#     save_hm3_per_mwh: float = 1/3000,
#     # NUEVOS: Parámetros estacionales
#     seasonal_phase_months: float = 0.0,  # Desplazamiento de fase (0 = máximo en enero)
#     seasonal_amplitude: float = 0.0,     # Amplitud (0.2 = varía entre 0.8 y 1.0)
#     # NUEVOS: Parámetros de turbinación extra
#     max_hydro_capacity_mw: float = None,
#     overflow_threshold_pct: float = 95.0,  # Umbral para activar turbinación extra (toda la capacidad disponible)
#     sensitivity_mwh_per_percent: float = 2238,  # MWh para reducir 1% del nivel
#     # Parámetros de restricciones
#     umbrales_sequia: dict = None,
#     ahorro_por_fase: dict = None
# ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:  # MODIFICADO: Retorna 5 series
#     """
#     Simula la evolución del nivel de los embalses aplicando dinámicamente tres medidas:
#     1. DESALACIÓN: Cuando hay excedentes y niveles bajos/medios (con factor estacional)
#     2. RESTRICCIONES: Según fase de sequía (niveles críticos bajos)
#     3. TURBINACIÓN EXTRA: Cuando niveles superan el umbral (prevención sobrellenado)
    
#     NUEVO: Incluye gestión de sobrellenado mediante turbinación extra que sustituye
#     generación térmica respetando la capacidad hidráulica máxima. Se activa completamente
#     (100% de la capacidad disponible) cuando el nivel supera el umbral configurado.
    
#     NUEVO: Factor estacional que modula la desalación según la época del año.
#     """
#     # --- Inicialización y Parámetros por Defecto ---      
#     if umbrales_sequia is None:
#         umbrales_sequia = {
#             'Prealerta': 60.0, 'Alerta': 40.0, 'Excepcionalitat': 25.0,
#             'Emergencia': 16.0, 'Emergencia_2': 11.0, 'Emergencia_3': 5.5
#         }
        
#     # if ahorro_por_fase is None:
#     #     ahorro_por_fase = {
#     #         'Normalitat': 0.0, 'Prealerta': 0.0, 'Alerta': 0.08, 'Excepcionalidad': 0.12,
#     #         'Emergencia I': 0.23, 'Emergencia II': 0.31, 'Emergencia III': 0.38
#     #     }

#     # NUEVO: Validación parámetros turbºinación
#     if max_hydro_capacity_mw is None:
#         raise ValueError("max_hydro_capacity_mw es requerido para la gestión de sobrellenado")

#     # --- Pre-cálculos ---
#     pct_per_hm3 = 100.0 / max_capacity_int
#     consumo_base_horario_hm3 = consumo_base_diario_hm3 / 24.0

#     # Identificar periodos válidos para desalación (lógica vectorizada ya aplicada)
#     mask = surpluses > 0
#     run_id = mask.ne(mask.shift()).cumsum()
#     run_lengths = run_id.value_counts()
#     block_lengths = run_id.map(run_lengths)
#     valid_mask = (mask) & (block_lengths >= min_run_hours)
#     valid_desal_hours = surpluses.index[valid_mask]

#     # --- INICIO DE LA MODIFICACIÓN (Opción 2: NumPy) ---

#     # 1. Convertir las Series de Pandas a arrays de NumPy para un acceso mucho más rápido en el bucle.
#     level_np = level_base.copy().to_numpy(dtype=float)
#     desal_mw_np = np.zeros_like(level_np)
#     restriction_savings_hm3_np = np.zeros_like(level_np)
#     # NUEVO: Arrays para turbinación extra
#     extra_hydro_mw_np = np.zeros_like(level_np)
#     thermal_reduced_mw_np = np.zeros_like(level_np)
    
#     surpluses_np = surpluses.to_numpy()
#     # NUEVO: Arrays para generación
#     thermal_np = thermal_generation.to_numpy()
#     base_hydro_np = base_hydro_generation.to_numpy()
    
#     # 2. Crear una máscara booleana de NumPy para las horas de desalación válidas.
#     original_index = level_base.index
#     is_valid_desal_hour_np = original_index.isin(valid_desal_hours)

#     # --- Función auxiliar para fases de sequía (sin cambios) ---
#     def get_fase_sequia(nivel_pct):
#         if nivel_pct < umbrales_sequia['Emergencia III']: return 'Emergencia III'
#         elif nivel_pct < umbrales_sequia['Emergencia II']: return 'Emergencia II'
#         elif nivel_pct < umbrales_sequia['Emergencia I']: return 'Emergencia I'
#         elif nivel_pct < umbrales_sequia['Excepcionalidad']: return 'Excepcionalidad'
#         elif nivel_pct < umbrales_sequia['Alerta']: return 'Alerta'
#         elif nivel_pct < umbrales_sequia['Prealerta']: return 'Prealerta'
#         else: return 'Normalitat'


#     # Antes del bucle
#     months = np.array([ts.month for ts in original_index], dtype=np.float64)
#     days = np.array([ts.day for ts in original_index], dtype=np.float64)
#     days_in_months = np.array([ts.days_in_month for ts in original_index], dtype=np.float64)
#     seasonal_factors = seasonal_factor_array(months, days, days_in_months, seasonal_phase_months, seasonal_amplitude)
    

#     # --- Bucle de Simulación Horaria (Ampliado con Factor Estacional) ---
#     for i in range(len(level_np)):
#         # Acceso directo al array de NumPy por índice entero (muy rápido)
#         current_level_pct = level_np[i]
#         current_timestamp = original_index[i]
        
#         # 1. CÁLCULO DE AHORRO POR RESTRICCIONES (se aplica siempre)
#         fase = get_fase_sequia(current_level_pct)
#         ahorro_pct = ahorro_por_fase[fase]
#         ahorro_hora_hm3 = consumo_base_horario_hm3 * ahorro_pct
#         restriction_savings_hm3_np[i] = ahorro_hora_hm3

#         # 2. CÁLCULO DE APORTE POR DESALACIÓN (sólo si hay excedente)
#         desal_hora_hm3 = 0.0
#         if is_valid_desal_hour_np[i]:
#             f_desal = sigmoid_factor(current_level_pct, midpoint, steepness)
            
#             # NUEVO: Aplicar factor estacional solo en condiciones de normalidad
#             # if fase == 'Normalitat':
#             #     f_seasonal = seasonal_factor(current_timestamp, seasonal_phase_months, seasonal_amplitude)
#             #     f_desal *= f_seasonal
            
#             # En el bucle
#             if fase == 'Normalitat' or fase == 'Prealerta':
#                 f_desal *= seasonal_factors[i]

            
#             cap_desal = max_desal_mw * f_desal
#             mw_usados = min(surpluses_np[i], cap_desal)
#             desal_mw_np[i] = mw_usados
#             desal_hora_hm3 = mw_usados * save_hm3_per_mwh

#         # 3. NUEVO: CÁLCULO DE TURBINACIÓN EXTRA (prevención sobrellenado)
#         extra_turbine_hm3 = 0.0
#         if current_level_pct >= overflow_threshold_pct:
#             # Calcular capacidad hidro disponible
#             current_hydro_total = base_hydro_np[i] + extra_hydro_mw_np[i]
#             available_hydro_mw = max_hydro_capacity_mw - current_hydro_total
            
#             # Calcular cuánto podemos turbinar (limitado por térmica y capacidad hidro)
#             max_possible_turbine_mw = min(thermal_np[i], available_hydro_mw)
            
#             if max_possible_turbine_mw > 0:
#                 # Usar toda la capacidad disponible (sin sigmoide)
#                 target_turbine_mw = max_possible_turbine_mw
                
#                 # Registrar la turbinación extra
#                 extra_hydro_mw_np[i] = target_turbine_mw
#                 thermal_reduced_mw_np[i] = target_turbine_mw
                
#                 # Convertir a reducción de nivel (efecto negativo en hm³)
#                 extra_turbine_hm3 = -target_turbine_mw / sensitivity_mwh_per_percent

#         # 4. ACTUALIZACIÓN DEL NIVEL (Suma algebraica de los tres efectos)
#         # Positivo: ahorro + desalación (añaden agua/reducen consumo)
#         # Negativo: turbinación extra (reduce nivel del embalse)
#         total_hm3_change = ahorro_hora_hm3 + desal_hora_hm3 + extra_turbine_hm3
        
#         if abs(total_hm3_change) > 1e-6:  # Solo actualizar si hay cambio significativo
#             delta_pct = total_hm3_change * pct_per_hm3
#             # Actualización del slice del array (la operación más crítica y ahora muy rápida)
#             level_np[i:] += delta_pct
            
#     # 5. Convertir los arrays de NumPy de vuelta a Series de Pandas con el índice original.
#     level_final = pd.Series(level_np, index=original_index, name=level_base.name)
#     desal_final = pd.Series(desal_mw_np, index=original_index, name='desal_mw')
#     savings_final = pd.Series(restriction_savings_hm3_np, index=original_index, name='restriction_savings_hm3')
#     # NUEVO: Series para turbinación extra
#     extra_hydro_final = pd.Series(extra_hydro_mw_np, index=original_index, name='extra_hydro_mw')
#     thermal_reduced_final = pd.Series(thermal_reduced_mw_np, index=original_index, name='thermal_reduced_mw')
    
#     # --- FIN DE LA MODIFICACIÓN ---

#     return level_final, desal_final, savings_final, extra_hydro_final, thermal_reduced_final






def simulate_full_water_management(
    # =========================================================================
    # INPUTS DEL MODEL ELÈCTRIC (sèries horàries)
    # =========================================================================
    surpluses: pd.Series,
    # """Excedents elèctrics horaris [MW]. Positiu = generació > demanda."""
    
    level_base: pd.Series,
    # """Nivell base dels embassaments [%]. Sèrie 'fictícia' sense intervencions."""
    
    thermal_generation: pd.Series,
    # """Generació tèrmica disponible per substituir [MW]."""
    
    base_hydro_generation: pd.Series,
    # """Generació hidràulica base del model elèctric [MW]."""
    
    # =========================================================================
    # PARÀMETRES FÍSICS DEL SISTEMA HÍDRIC
    # =========================================================================
    max_capacity_int: float,
    # """Capacitat total dels embassaments de les conques internes [hm³]."""
    
    consumo_base_diario_estacional_hm3: np.ndarray = None,  # Array de 12 valors [hm³/dia]
    # Consum hídric mitjà diari de referència per a cada mes (estacionalitat)
    consumo_base_diario_hm3: float = 2.05,
    # """Consum hídric mitjà diari de referència [hm³/dia]."""    
    
    # =========================================================================
    # PARÀMETRES DE REGENERACIÓ EN CONTINU (NOU)
    # =========================================================================
    max_regen_hm3_dia: float = 0.173,
    # """Capacitat màxima de regeneració [hm³/dia]."""
    
    regen_cost_mwh_per_hm3: float = 1200.0,
    # """Cost energètic de la regeneració [MWh/hm³]."""
    
    regen_min_pct: float = 0.20,
    # """Mínim tècnic de la planta de regeneració [fracció]."""
    
    regen_base_pct: float = 0.9,
    # """Nivell operatiu base en normalitat [fracció]."""
    
    regen_rampa_pct: float = 0.15,
    # """Rampa màxima per hora [fracció de potència nominal].""" 
    
    llindar_activacio_regen_max: int = 2,
    # Llindar a partir del qual passa a règim de plena potència
    
    # =========================================================================
    # PARÀMETRES DE DESSALINITZACIÓ
    # =========================================================================
    max_desal_mw: float = 32.0,
    # """Potència màxima de les plantes dessalinitzadores [MW]."""
    
    min_run_hours: int = 6,
    # """Hores mínimes consecutives d'excedent per activar dessalinització."""
    
    midpoint: float = 75.0,
    # """Punt d'inflexió de la sigmoide: nivell [%] on f_desal = 0.5."""
    
    steepness: float = 0.2,
    # """Pendent de la sigmoide. Més alt = transició més brusca."""
    
    save_hm3_per_mwh: float = 1/3500,
    # """Factor de conversió energia→aigua [hm³/MWh]. ~3.5 kWh/m³ típic."""
    
    rampa_desal_pct: float = 0.30,  # 30% de potència nominal per hora
    # Rampa màxima de dessalinització per hora [fracció de potència nominal].
    
    desal_min_pct: float = 0.18, # 18% de la potència nominal de mínim tècnic

    llindar_activacio_desal_max: int = 2,
    # Llindar a partir del qual passa a règim de plena potència    
    
    # desal_derivada_sensibilitat: float = 2.0,
    # # """Sensibilitat a la derivada del nivell (típic: 1-5)."""    
    # desal_derivada_amplitud_up: float = 0.15,
    # desal_derivada_amplitud_down: float = 0.5,
    # # """Amplitud de modulació per derivada (0.5 → factor entre 0.5 i 1.5)."""
    
    k_deriv: float = 0.0,

    finestra_hores: int = 24*30,
    # Finestra horaria pel càlcul de la tendencia del nivell (168h = 1 setmana)   
    
    # =========================================================================
    # PARÀMETRES D'ESTACIONALITAT DE LA DESSALINITZACIÓ
    # =========================================================================
    llindar_activacio_desal_estacional: int = 1,
    # Llindar que separa el regim estacional del règim de crisi.
    
    seasonal_phase_months: float = 0.0,
    # """Desplaçament de fase [mesos]. 0 = màxim en gener."""
    
    seasonal_amplitude: float = 0.0,
    # """Amplitud de variació estacional [0-1]. 0 = sense variació."""
    
    # =========================================================================
    # PARÀMETRES DE TURBINACIÓ PREVENTIVA (ANTI-OVERFLOW)
    # =========================================================================
    max_hydro_capacity_mw: float = None,
    # """Potència hidràulica màxima instal·lada [MW]. Requerit."""
    
    overflow_threshold_pct: float = 90.0,
    # """Llindar de nivell [%] per activar turbinació preventiva."""
    
    # VALOR A MODIFICAR , HACERLO DEPENDIENTE DE LA SENSIBILIDAD DE LAS CIC
    # sensibility_int * max_capacity_int * 0.01 = 2238.39
    sensitivity_mwh_per_percent: float = 2238.39,
    # """Energia necessària per reduir 1% el nivell [MWh/%]."""
    
    # =========================================================================
    # PARÀMETRES DE RESTRICCIONS DE CONSUM (POLÍTICA DE SEQUERA)
    # =========================================================================
    umbrales_sequia: Optional[dict] = None,
    # """Llindars de nivell [%] per a cada fase de sequera."""
    
    restricciones_sectoriales: Optional[dict] = None,
    # """Percentatge de restricció per sector i fase. Si None, usa valors per defecte."""

) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Simula l'evolució del nivell dels embassaments aplicant tres polítiques de gestió:
    
    1. RESTRICCIONS DE CONSUM: Estalvi forçat segons fase de sequera i estacionalitat
    2. REGENERACIÓ EN CONTINU: Aportació constant modulada per situació    
    3. DESSALINITZACIÓ OPORTUNISTA: Activació oportunista quan hi ha excedents elèctrics
    4. TURBINACIÓ PREVENTIVA: Buidatge controlat quan nivell supera el llindar
    
    El model avança hora a hora, actualitzant el nivell i propagant els canvis
    a tot el període futur (com una integral acumulativa).
    
    Returns:
        Tuple amb 5 pd.Series:
        - level_final: Nivell dels embassaments amb intervencions [%]
        - regen_final: Aportació per regeneració [hm³]        
        - desal_final: Potència de dessalinització usada [MW]
        - savings_final: Estalvi hídric per restriccions [hm³]
        - extra_hydro_final: Generació hidràulica extra [MW]
        - thermal_reduced_final: Generació tèrmica substituïda [MW]
    """
    
    # =========================================================================
    # VALIDACIONS
    # =========================================================================
    
    if max_hydro_capacity_mw is None:
        raise ValueError("max_hydro_capacity_mw és requerit per a la gestió de sobrellenament")
    
    # =========================================================================
    # VALORS PER DEFECTE DE LA DEMANDA D'AIGUA
    # =========================================================================    
    if consumo_base_diario_estacional_hm3 is None:
        consumo_base_diario_estacional_hm3 = np.array([
            1.37, 1.37, 1.42, 1.74, 2.32, 3.03, 
            3.51, 3.24, 2.36, 1.48, 1.37, 1.37
        ])
        consumo_base_diario_hm3 = np.mean(consumo_base_diario_estacional_hm3)
    
    # =========================================================================
    # VALORS PER DEFECTE DE POLÍTICA DE SEQUERA
    # =========================================================================
    
    if umbrales_sequia is None:
        umbrales_sequia = {
            'Emergencia_3': 5.5, 'Emergencia_2': 11.0, 'Emergencia': 16.0,
            'Excepcionalitat': 25.0, 'Alerta': 40.0, 'Prealerta': 60.0
        }

    # =========================================================================
    # PRE-CÀLCUL DE LA MATRIU D'ESTALVI (1 cop per escenari)
    # =========================================================================
    
    matriu_estalvi, llindars = generar_matriu_estalvi(
        umbrales_sequia=umbrales_sequia,
        restricciones_sectoriales=restricciones_sectoriales
    )

    matriu_estalvi_regadiu = generar_matriu_estalvi_sector(
        umbrales_sequia, restricciones_sectoriales, sector='Regadiu')
    
    matriu_estalvi_ramaderia = generar_matriu_estalvi_sector(
        umbrales_sequia, restricciones_sectoriales, sector='Ramaderia')
    
    matriu_estalvi_agro = matriu_estalvi_regadiu + matriu_estalvi_ramaderia
    
    matriu_estalvi_urba = generar_matriu_estalvi_sector(
        umbrales_sequia, restricciones_sectoriales, sector='Urba')    
   
    # =========================================================================
    # PRE-CÀLCULS (fora del bucle per eficiència)
    # =========================================================================
    
    # Conversions d'unitats
    pct_per_hm3 = 100.0 / max_capacity_int
    consumo_base_horario_hm3 = consumo_base_diario_hm3 / 24.0
    # consumo_base_horario_hm3 = consumo_base_diario_estacional_hm3 / 24.0

    # Regeneració: conversions
    max_regen_hm3_hora = max_regen_hm3_dia / 24.0
    regen_rampa_hm3_hora = regen_rampa_pct * max_regen_hm3_hora
    
    # Dessalinització: rampa màxima
    desal_rampa_mw = rampa_desal_pct * max_desal_mw    
    
    # Identificar blocs d'excedents vàlids per dessalinització
    # (només blocs de >= min_run_hours hores consecutives)
    mask_surplus = surpluses > 0
    run_id = mask_surplus.ne(mask_surplus.shift()).cumsum()
    run_lengths = run_id.value_counts()
    block_lengths = run_id.map(run_lengths)
    valid_desal_mask = (mask_surplus) & (block_lengths >= min_run_hours)
    
    # Convertir a arrays NumPy per velocitat
    original_index = level_base.index
    level_np = level_base.copy().to_numpy(dtype=float)
    surpluses_np = surpluses.to_numpy()
    thermal_np = thermal_generation.to_numpy()
    base_hydro_np = base_hydro_generation.to_numpy()
    is_valid_desal_np = valid_desal_mask.to_numpy()
    
    # Arrays de sortida
    n = len(level_np)
    regen_hm3_np = np.zeros(n, dtype=np.float64)    
    desal_mw_np = np.zeros(n, dtype=np.float64)
    restriction_savings_hm3_np = np.zeros(n, dtype=np.float64)
    restriction_agro_savings_hm3_np = np.zeros(n, dtype=np.float64)
    restriction_urba_savings_hm3_np = np.zeros(n, dtype=np.float64)
    extra_hydro_mw_np = np.zeros(n, dtype=np.float64)
    # thermal_reduced_mw_np = np.zeros(n, dtype=np.float64)
    deltas_pct_np = np.zeros(n, dtype=np.float64)
    level_base_np = level_base.to_numpy(dtype=np.float64)  # Guardar original
    delta_acumulat = 0.0  # <<< Acumulador escalar  
    level_calculat_np = np.zeros(n, dtype=np.float64)  # Historial del nivell amb intervencions
    surpluses_net_np = surpluses_np.copy()
    spillage_hm3_np = np.zeros(n, dtype=np.float64)  # Vessaments forçats
    
    # Pre-calcular factors estacionals (si aplica)
    if seasonal_amplitude > 0:
        months = np.array([ts.month for ts in original_index], dtype=np.float64)
        days = np.array([ts.day for ts in original_index], dtype=np.float64)
        days_in_months = np.array([ts.days_in_month for ts in original_index], dtype=np.float64)
        seasonal_factors = seasonal_factor_array(
            months, days, days_in_months, seasonal_phase_months, seasonal_amplitude
        )
    else:
        seasonal_factors = np.ones(n, dtype=np.float64)
    
    # Pre-extreure mesos per al càlcul d'estalvi
    months_array = np.array([ts.month for ts in original_index], dtype=np.int32)
    
    # Variables d'estat inicial per rampes
    regen_anterior_hm3 = regen_base_pct * max_regen_hm3_hora  # Començar al 50%
    desal_anterior_mw = 0.0    
    
    
    # =========================================================================
    # BUCLE DE SIMULACIÓ HORÀRIA
    # =========================================================================
    
    for i in range(n):
        # current_level_pct = level_np[i]
        current_level_pct = level_base_np[i] + delta_acumulat  # <<< CANVI
        current_month = months_array[i]
        
        # Guardar per poder calcular derivada més endavant
        level_calculat_np[i] = current_level_pct
        
        # Calcular derivada del nivell (mitjana mòbil de la darrera setmana)
        if i >= finestra_hores and k_deriv > 0:
            derivada_nivel = (current_level_pct - level_np[i-finestra_hores]) / finestra_hores  # %/hora (mitjana 24h)
        else:
            derivada_nivel = 0.0        
        
        # ---------------------------------------------------------------------
        # 1. ESTALVI PER RESTRICCIONS DE CONSUM
        # ---------------------------------------------------------------------
        # Ara usem la funció millorada amb estacionalitat sectorial
        # estalvi_pct = calcular_estalvi_pct(
        #     mes=current_month,
        #     nivel_pct=current_level_pct,
        #     umbrales_sequia=umbrales_sequia,
        #     restricciones_sectoriales=restricciones_sectoriales
        # )
        
        # Pas A: Obtenir índex de fase de sequera (0-6) a partir del nivell
        fase_idx = get_fase_idx_numba(current_level_pct, llindars)
        # Pas B: Lookup a la matriu pre-calculada
        estalvi_pct = matriu_estalvi[current_month - 1, fase_idx]
        
        ahorro_hora_hm3 = consumo_base_horario_hm3 * estalvi_pct
        # ahorro_hora_hm3 = consumo_base_horario_hm3[current_month - 1] * estalvi_pct
        restriction_savings_hm3_np[i] = ahorro_hora_hm3
        
        estalvi_agro_pct = matriu_estalvi_agro[current_month - 1, fase_idx]
        ahorro_agro_hora_hm3 = consumo_base_horario_hm3 * estalvi_agro_pct
        # ahorro_agro_hora_hm3 = consumo_base_horario_hm3[current_month - 1] * estalvi_agro_pct
        restriction_agro_savings_hm3_np[i] = ahorro_agro_hora_hm3        

        estalvi_urba_pct = matriu_estalvi_urba[current_month - 1, fase_idx]
        ahorro_urba_hora_hm3 = consumo_base_horario_hm3 * estalvi_urba_pct
        # ahorro_urba_hora_hm3 = consumo_base_horario_hm3[current_month - 1] * estalvi_urba_pct
        restriction_urba_savings_hm3_np[i] = ahorro_urba_hora_hm3        

        # ---------------------------------------------------------------------
        # 2. REGENERACIÓ EN CONTINU
        # ---------------------------------------------------------------------
        # Determinar nivell objectiu segons situació
        
        if current_level_pct >= overflow_threshold_pct:
            # Sobrecapacitat: mínim tècnic
            regen_objectiu_hm3 = regen_min_pct * max_regen_hm3_hora
        elif fase_idx >= llindar_activacio_regen_max:  #2 = Alerta o pitjor
            # Situació de sequera: màxima potència
            regen_objectiu_hm3 = max_regen_hm3_hora
        else:
            # Normalitat/Prealerta: operació base
            regen_objectiu_hm3 = regen_base_pct * max_regen_hm3_hora
            # regen_objectiu_hm3 = 0.9 * max_regen_hm3_hora
        
        # Aplicar rampa
        if regen_objectiu_hm3 > regen_anterior_hm3:
            regen_actual_hm3 = min(regen_objectiu_hm3, regen_anterior_hm3 + regen_rampa_hm3_hora)
        else:
            regen_actual_hm3 = max(regen_objectiu_hm3, regen_anterior_hm3 - regen_rampa_hm3_hora)

        # regen_actual_hm3 = calcular_regeneracio_hora(
        #     current_level_pct,
        #     fase_idx,
        #     regen_anterior_hm3,
        #     overflow_threshold_pct,
        #     max_regen_hm3_hora,
        #     regen_min_pct,
        #     regen_base_pct,
        #     regen_rampa_hm3_hora
        # )
        
        regen_hm3_np[i] = regen_actual_hm3
        regen_anterior_hm3 = regen_actual_hm3
        
        # Calcular consumo eléctrico de la regeneración
        consumo_regen_mwh = regen_actual_hm3 * regen_cost_mwh_per_hm3
        excedent_disponible = max(surpluses_np[i] - consumo_regen_mwh,0)  # Excedente neto
        surpluses_net_np[i] -= consumo_regen_mwh

        
        # ---------------------------------------------------------------------
        # 3. DESSALINITZACIÓ OPORTUNISTA
        # ---------------------------------------------------------------------
        desal_hora_hm3 = 0.0
        # desal_minim_mw = max_desal_mw * 0.18 # mínim tècnic
        # desal_minim_hm3 = max_desal_mw * 0.18 * 24 * save_hm3_per_mwh # mínim tècnic
        
        if fase_idx >= llindar_activacio_desal_max:
            desal_ideal_mw = max_desal_mw * 0.9

        
        elif is_valid_desal_np[i]:
          
            
            # Factor sigmoide basat en nivell: més dessalinització quan nivell és baix
            # f_nivel = sigmoid_factor_numba(current_level_pct, midpoint, steepness, desal_min_pct)
            derivada_nivel = min(derivada_nivel, 0.0)
            f_nivel = sigmoid_factor_numba(current_level_pct + k_deriv*derivada_nivel*finestra_hores, midpoint, steepness, desal_min_pct)
                       
            # Modulació estacional (només en normalitat/prealerta)
            f_est = seasonal_factors[i] if fase_idx <= llindar_activacio_desal_estacional else 1.0
            
            
            # # Factor basat en derivada (més urgent si baixant)
            # f_derivada = factor_derivada_numba(
            #     derivada_nivel, 
            #     desal_derivada_sensibilitat, 
            #     desal_derivada_amplitud_up,
            #     desal_derivada_amplitud_down
            # )
              
            # Factor combinat
            f_desal = f_nivel * f_est           
          
            # Limitació tècnica
            f_desal = min(1.0, max(desal_min_pct,f_desal))
            
            # Calcular MW usats i convertir a hm³
            desal_ideal_mw = max_desal_mw * f_desal
            # desal_ideal_mw = min(surpluses_np[i], desal_ideal_mw)
            desal_ideal_mw = min(excedent_disponible, desal_ideal_mw)
            
        else:
            # Mode base, en mínim tècnic/econòmic
            desal_ideal_mw = desal_min_pct * max_desal_mw
            
        # Aplicar mínim tècnic
        desal_ideal_mw = max(desal_ideal_mw, desal_min_pct * max_desal_mw)        
        # Aplicar rampa
        desal_actual_mw = np.clip(
        desal_ideal_mw,
        desal_anterior_mw - desal_rampa_mw,
        desal_anterior_mw + desal_rampa_mw
    )
        # Actualitzar estat
        # desal_actual_mw = max(0.0, desal_actual_mw)  # No pot ser negatiu
        desal_mw_np[i] = desal_actual_mw
        desal_hora_hm3 = desal_actual_mw * save_hm3_per_mwh
        desal_anterior_mw = desal_actual_mw
        surpluses_net_np[i] -= desal_mw_np[i]
    
        # ---------------------------------------------------------------------
        # 4. TURBINACIÓ PREVENTIVA (ANTI-OVERFLOW)
        # ---------------------------------------------------------------------
        extra_turbine_hm3 = 0.0
        
        if current_level_pct >= overflow_threshold_pct:
            # Capacitat hidro disponible
            current_hydro_total = base_hydro_np[i] + extra_hydro_mw_np[i]
            available_hydro_mw = max_hydro_capacity_mw - current_hydro_total
            
            # Limitat per tèrmica disponible i capacitat hidro
            max_possible_turbine_mw = min(thermal_np[i], available_hydro_mw)
            
            if max_possible_turbine_mw > 0:
                extra_hydro_mw_np[i] = max_possible_turbine_mw
                # thermal_reduced_mw_np[i] = max_possible_turbine_mw
                
                # Convertir a reducció de nivell (efecte negatiu)
                extra_turbine_hm3 = -max_possible_turbine_mw / (sensitivity_mwh_per_percent * pct_per_hm3)
                
        # ---------------------------------------------------------------------
        # 4b. VESSAMENT FORÇAT (si nivell > 100% malgrat turbinació)
        # ---------------------------------------------------------------------
        spillage_hm3 = 0.0
        
        if current_level_pct >= 100.0:
            exces_pct = current_level_pct - 99.0
            spillage_hm3 = exces_pct / pct_per_hm3
            spillage_hm3_np[i] = spillage_hm3
        
        # ---------------------------------------------------------------------
        # 5. GUARDAR DELTA (acumulació diferida)
        # ---------------------------------------------------------------------
        total_hm3_change = ahorro_hora_hm3 + regen_actual_hm3 + desal_hora_hm3 + extra_turbine_hm3 - spillage_hm3        
        deltas_pct_np[i] = total_hm3_change * pct_per_hm3
        delta_acumulat += deltas_pct_np[i]
         
            
        # ---------------------------------------------------------------------
        # 5. ACTUALITZACIÓ DEL NIVELL
        # ---------------------------------------------------------------------
        # # Balanç: (+) estalvi i dessalinització, (-) turbinació
        # total_hm3_change = ahorro_hora_hm3 + regen_actual_hm3 + desal_hora_hm3 + extra_turbine_hm3
        
        # if abs(total_hm3_change) > 1e-9:
        #     delta_pct = total_hm3_change * pct_per_hm3
        #     # Propagar el canvi a totes les hores futures
        #     level_np[i:] += delta_pct

        # # ---------------------------------------------------------------------
        # # 5. GUARDAR DELTA (acumulació diferida)
        # # ---------------------------------------------------------------------            
        # total_hm3_change = (
        #     ahorro_hora_hm3 +
        #     regen_actual_hm3 +
        #     desal_hora_hm3 +
        #     extra_turbine_hm3 +
        #     spillage_hm3
        # )
        # deltas_pct_np[i] = total_hm3_change * pct_per_hm3
        # delta_acumulat += deltas_pct_np[i]  # <<< Acumular per la pròxima iteració           
    
    
    # =========================================================================
    # CONVERSIÓ A SERIES PANDAS
    # =========================================================================
    level_np = level_base_np + np.cumsum(deltas_pct_np)
    level_final = pd.Series(level_np, index=original_index, name='level_pct')
    regen_final = pd.Series(regen_hm3_np, index=original_index, name='regen_hm3')    
    desal_final = pd.Series(desal_mw_np, index=original_index, name='desal_mw')
    savings_final = pd.Series(restriction_savings_hm3_np, index=original_index, name='restriction_savings_hm3')
    savings_agro_final = pd.Series(restriction_agro_savings_hm3_np, index=original_index, name='restriction_agro_savings_hm3')
    savings_urba_final = pd.Series(restriction_urba_savings_hm3_np, index=original_index, name='restriction_urba_savings_hm3')
    extra_hydro_final = pd.Series(extra_hydro_mw_np, index=original_index, name='extra_hydro_mw')
    surpluses_net_final = pd.Series(surpluses_net_np, index=original_index, name='surpluses_net_mw')
    spillage_final = pd.Series(spillage_hm3_np, index=original_index, name='spillage_hm3')
    # thermal_reduced_final = pd.Series(thermal_reduced_mw_np, index=original_index, name='thermal_reduced_mw')
    
    return level_final, regen_final, desal_final, savings_final, savings_agro_final, savings_urba_final, extra_hydro_final, surpluses_net_final, spillage_final #, thermal_reduced_final


def generar_matriu_estalvi(
    umbrales_sequia: dict = None,
    restricciones_sectoriales: dict = None
) -> tuple:
    """
    Genera la matriu de lookup d'estalvi_pct.
    
    CRIDAR UNA VEGADA PER ESCENARI, abans del bucle de simulació.
    
    Args:
        umbrales_sequia: Llindars de cada fase (parametritzable per escenari)
        restricciones_sectoriales: % restricció per sector i fase (parametritzable)
    
    Returns:
        tuple: (matriu_estalvi, llindars_array)
            - matriu_estalvi: np.ndarray[12, 7] amb estalvi_pct per cada (mes, fase)
            - llindars_array: np.ndarray[6] amb llindars ordenats ascendent
    """
    
    # === PARÀMETRES FIXOS (estructura del sistema, no canvien entre escenaris) ===
    
    PESOS_SECTORS = {
        'Urba': 0.415,
        'Regadiu': 0.344,
        'Ramaderia': 0.02,
        'Ind_Bens': 0.0785,
        'Ind_Turisme': 0.0135
    }
    
    ESTACIONALITAT = {
        'Urba':        [0.96, 0.96, 0.96, 0.96, 1.08, 1.08, 1.08, 1.08, 0.96, 0.96, 0.96, 0.96],
        'Regadiu':     [0.13, 0.13, 0.19, 0.63, 1.26, 2.26, 2.89, 2.51, 1.50, 0.25, 0.13, 0.13],
        'Ramaderia':   [1.0]*12,
        'Ind_Bens':    [1.0]*12,
        'Ind_Turisme': [0.51, 0.51, 0.60, 0.69, 0.94, 1.54, 2.15, 2.15, 1.28, 0.69, 0.51, 0.43]
    }
    
    FASES = ['Normalitat', 'Prealerta', 'Alerta', 'Excepcionalitat', 
             'Emergencia', 'Emergencia_2', 'Emergencia_3']
    
    # === PARÀMETRES CONFIGURABLES (poden canviar entre escenaris) ===
    
    if umbrales_sequia is None:
        umbrales_sequia = {
            'Emergencia_3': 5.5, 'Emergencia_2': 11.0, 'Emergencia': 16.0,
            'Excepcionalitat': 25.0, 'Alerta': 40.0, 'Prealerta': 60.0
        }
    
    if restricciones_sectoriales is None:
        restricciones_sectoriales = {
            'Urba':        {'Normalitat':0, 'Prealerta':0.025, 'Alerta':0.05, 'Excepcionalitat':0.075, 'Emergencia':0.10, 'Emergencia_2':0.12, 'Emergencia_3':0.14},
            'Regadiu':     {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.25, 'Excepcionalitat':0.40, 'Emergencia':0.80, 'Emergencia_2':0.80, 'Emergencia_3':0.80},
            'Ramaderia':   {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.10, 'Excepcionalitat':0.30, 'Emergencia':0.50, 'Emergencia_2':0.50, 'Emergencia_3':0.50},
            'Ind_Bens':    {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.05, 'Excepcionalitat':0.15, 'Emergencia':0.25, 'Emergencia_2':0.25, 'Emergencia_3':0.25},
            'Ind_Turisme': {'Normalitat':0, 'Prealerta':0.000, 'Alerta':0.05, 'Excepcionalitat':0.15, 'Emergencia':0.25, 'Emergencia_2':0.25, 'Emergencia_3':0.25}
        }
    
    # === GENERACIÓ DE LA MATRIU ===
    
    matriu = np.zeros((12, 7), dtype=np.float64)
    
    for mes_idx in range(12):  # 0 = Gener, 11 = Desembre
        for fase_idx, fase in enumerate(FASES):  # 0 = Normalitat, 6 = Emergencia_3
            
            estalvi = 0.0
            for sector, pes in PESOS_SECTORS.items():
                coef_estacional = ESTACIONALITAT[sector][mes_idx]
                pct_restriccio = restricciones_sectoriales[sector].get(fase, 0.0)
                estalvi += pes * coef_estacional * pct_restriccio
            
            matriu[mes_idx, fase_idx] = estalvi
    
    # === ARRAY DE LLINDARS (ordenat ascendent per a searchsorted) ===
    
    llindars = np.array([
        umbrales_sequia['Emergencia_3'],   # índex 0 → si nivel < aquest → fase_idx = 6
        umbrales_sequia['Emergencia_2'],   # índex 1 → si nivel < aquest → fase_idx = 5
        umbrales_sequia['Emergencia'],     # índex 2 → si nivel < aquest → fase_idx = 4
        umbrales_sequia['Excepcionalitat'],# índex 3 → si nivel < aquest → fase_idx = 3
        umbrales_sequia['Alerta'],         # índex 4 → si nivel < aquest → fase_idx = 2
        umbrales_sequia['Prealerta'],      # índex 5 → si nivel < aquest → fase_idx = 1
    ], dtype=np.float64)
    
    return matriu, llindars


def generar_matriu_estalvi_sector(
    umbrales_sequia: dict = None,
    restricciones_sectoriales: dict = None,
    sector: str = 'Regadiu'
) -> np.ndarray:
    """
    Genera matriu d'estalvi [12 mesos × 7 fases] per un sector específic.
    
    Args:
        umbrales_sequia: Llindars de nivell per cada fase
        restricciones_sectoriales: Restriccions per sector i fase
        sector: Nom del sector ('Regadiu', 'Urba', 'Ramaderia', 'Ind_Bens', 'Ind_Turisme')
    
    Returns:
        np.ndarray: Matriu [12, 7] amb percentatge d'estalvi del sector
    """

    if umbrales_sequia is None:
        umbrales_sequia = {
            'Emergencia_3': 5.5, 'Emergencia_2': 11.0, 'Emergencia': 16.0,
            'Excepcionalitat': 25.0, 'Alerta': 40.0, 'Prealerta': 60.0
        }
    
    # Pesos sectorials (fracció del consum total)
    pesos_sectors = {
        'Urba': 0.415,
        'Regadiu': 0.344,
        'Ramaderia': 0.02,
        'Ind_Bens': 0.0785,
        'Ind_Turisme': 0.0135
    }    
    
    # Estacionalitat per sector [12 mesos]  
    estacionalitat_sectors = {
        'Urba':        [0.96, 0.96, 0.96, 0.96, 1.08, 1.08, 1.08, 1.08, 0.96, 0.96, 0.96, 0.96],
        'Regadiu':     [0.13, 0.13, 0.19, 0.63, 1.26, 2.26, 2.89, 2.51, 1.50, 0.25, 0.13, 0.13],
        'Ramaderia':   [1.0]*12,
        'Ind_Bens':    [1.0]*12,
        'Ind_Turisme': [0.51, 0.51, 0.60, 0.69, 0.94, 1.54, 2.15, 2.15, 1.28, 0.69, 0.51, 0.43]
    }    
    
    # Restriccions per defecte
    if restricciones_sectoriales is None:
        restricciones_sectoriales = {
            'Urba':        {'Normalitat':0, 'Prealerta':0.025, 'Alerta':0.05,  'Excepcionalitat':0.075, 'Emergencia':0.10, 'Emergencia_2':0.12, 'Emergencia_3':0.14},
            'Regadiu':     {'Normalitat':0, 'Prealerta':0.00,  'Alerta':0.25,  'Excepcionalitat':0.40,  'Emergencia':0.80, 'Emergencia_2':0.80, 'Emergencia_3':0.80},
            'Ramaderia':   {'Normalitat':0, 'Prealerta':0.00,  'Alerta':0.10,  'Excepcionalitat':0.30,  'Emergencia':0.50, 'Emergencia_2':0.50, 'Emergencia_3':0.50},
            'Ind_Bens':    {'Normalitat':0, 'Prealerta':0.00,  'Alerta':0.05,  'Excepcionalitat':0.15,  'Emergencia':0.25, 'Emergencia_2':0.25, 'Emergencia_3':0.25},
            'Ind_Turisme': {'Normalitat':0, 'Prealerta':0.00,  'Alerta':0.05,  'Excepcionalitat':0.15,  'Emergencia':0.25, 'Emergencia_2':0.25, 'Emergencia_3':0.25}
        }
        
    
    # Validar que el sector existeix
    if sector not in pesos_sectors:
        raise ValueError(f"Sector '{sector}' no reconegut. Opcions: {list(pesos_sectors.keys())}")
    
    # Ordre de fases (de més severa a normalitat)
    # fases = ['Emergencia_3', 'Emergencia_2', 'Emergencia', 'Excepcionalitat', 'Alerta', 'Prealerta', 'Normalitat']
    fases = ['Normalitat', 'Prealerta', 'Alerta', 'Excepcionalitat', 'Emergencia', 'Emergencia_2', 'Emergencia_3']
    
    # Crear matriu [12 mesos, 7 fases]
    matriu = np.zeros((12, 7), dtype=np.float64)
    
    pes = pesos_sectors[sector]
    estacio = estacionalitat_sectors[sector]
    restriccions = restricciones_sectoriales[sector]
    
    for mes_idx in range(12):
        for fase_idx, fase in enumerate(fases):
            r_fase = restriccions.get(fase, 0)
            e_mes = estacio[mes_idx]
            
            # Estalvi = pes × estacionalitat × restricció
            matriu[mes_idx, fase_idx] = pes * e_mes * r_fase
    
    return matriu



@njit(cache=True)
def get_fase_idx_numba(nivel_pct: float, llindars: np.ndarray) -> int:
    """
    Retorna l'índex de fase (0-6) segons el nivell.
    
    Args:
        nivel_pct: Nivell actual (%)
        llindars: Array[6] ordenat ascendent [Em3, Em2, Em1, Exc, Alerta, Prealerta]
    
    Returns:
        int: 0=Normalitat, 1=Prealerta, 2=Alerta, 3=Excep, 4=Em1, 5=Em2, 6=Em3
    """
    if nivel_pct < llindars[0]:    # < Emergencia_3 (5.5)
        return 6
    elif nivel_pct < llindars[1]:  # < Emergencia_2 (11.0)
        return 5
    elif nivel_pct < llindars[2]:  # < Emergencia (16.0)
        return 4
    elif nivel_pct < llindars[3]:  # < Excepcionalitat (25.0)
        return 3
    elif nivel_pct < llindars[4]:  # < Alerta (40.0)
        return 2
    elif nivel_pct < llindars[5]:  # < Prealerta (60.0)
        return 1
    else:
        return 0  # Normalitat



@njit(cache=True, fastmath=True)
def calcular_regeneracio_hora(
    current_level_pct: float,
    fase_idx: int,
    regen_anterior_hm3: float,
    overflow_threshold_pct: float,
    max_regen_hm3_hora: float,
    regen_min_pct: float,
    regen_base_pct: float,
    regen_rampa_hm3_hora: float
) -> float:
    """
    Calcula la regeneració per a una hora donada.
    
    La planta de regeneració opera en tres modes:
    - Sobrecapacitat (nivell > overflow): mínim tècnic
    - Sequera (fase >= Alerta): màxima potència  
    - Normal/Prealerta: operació base (50%)
    
    Args:
        current_level_pct: Nivell actual dels embassaments [%]
        fase_idx: Índex de fase (0=Normal, 1=Prealerta, 2=Alerta, ..., 6=Emerg3)
        regen_anterior_hm3: Regeneració de l'hora anterior [hm³]
        overflow_threshold_pct: Llindar de sobrecapacitat [%]
        max_regen_hm3_hora: Capacitat màxima horària [hm³/h]
        regen_min_pct: Mínim tècnic [fracció, ex: 0.10]
        regen_base_pct: Operació base [fracció, ex: 0.50]
        regen_rampa_hm3_hora: Rampa màxima [hm³/h]
    
    Returns:
        float: Regeneració actual [hm³] (també serà el "anterior" de la pròxima hora)
    """
    
    # 1. Determinar nivell objectiu segons situació
    if current_level_pct >= overflow_threshold_pct:
        regen_objectiu = regen_min_pct * max_regen_hm3_hora
    elif fase_idx >= 2:  # Alerta (2) o pitjor (3,4,5,6)
        regen_objectiu = max_regen_hm3_hora
    else:  # Normalitat (0) o Prealerta (1)
        regen_objectiu = regen_base_pct * max_regen_hm3_hora
    
    # 2. Aplicar rampa (limitar canvi màxim per hora)
    delta = regen_objectiu - regen_anterior_hm3
    
    if delta > regen_rampa_hm3_hora:
        regen_actual = regen_anterior_hm3 + regen_rampa_hm3_hora
    elif delta < -regen_rampa_hm3_hora:
        regen_actual = regen_anterior_hm3 - regen_rampa_hm3_hora
    else:
        regen_actual = regen_objectiu
    
    return regen_actual


@njit(cache=True)
def factor_derivada_numba(derivada: float, sensibilitat: float = 2.0, amplitud_up: float = 0.15, amplitud_down: float = 0.5) -> float:
    """
    Modula la dessalinització segons la tendència del nivell.
    
    Args:
        derivada: Canvi de nivell (%/hora), negatiu = baixant
        sensibilitat: Com de reactiu és al canvi (típic: 1-5)
        amplitud: Rang de modulació (0.5 → factor entre 0.5 i 1.5)
    
    Returns:
        Factor multiplicatiu (>1 si baixant, <1 si pujant)
    """
    d_norm = derivada / 0.011
    
    # tanh(-derivada * sensibilitat) → positiu si derivada negativa (baixant)    
    if d_norm < 0.0:
        # Baixant → resposta forta
        return 1.0 + amplitud_down * np.tanh(-d_norm * sensibilitat)
    else:
        # Pujant → resposta suau
        return 1.0 + amplitud_up * np.tanh(-d_norm * sensibilitat)    

    # return 1.0 + amplitud * np.tanh(-d_norm * sensibilitat)

# @njit
# def factor_derivada_numba(derivada, sensibilitat, amplitud):
#     """
#     Retorna un 'boost' additiu.
#     Si derivada és 0 (estable) -> retorna 0.0
#     Si derivada baixa ràpid -> retorna fins a 'amplitud' (ex: 0.4)
#     """
    
#     # Si baixa, calculem el boost.
#     # Usem tanh per suavitzar la resposta fins a l'amplitud màxima.
#     return amplitud * np.tanh(-derivada * sensibilitat)

def increments_a_llindars(x1, x2, x3, x4):
    """
    Transforma increments en llindars absoluts.
    
    Args:
        x1: Llindar base Emergència [%]
        x2: Increment per Excepcionalitat [%]
        x3: Increment per Alerta [%]
        x4: Increment per Prealerta [%]
    
    Returns:
        dict amb umbrales_sequia
    """
    L_eme = x1
    L_exc = x1 + x2
    L_ale = x1 + x2 + x3
    L_pre = x1 + x2 + x3 + x4
    
    return {
        'Emergencia_3': L_eme * 0.35,   # Proporcions internes fixes
        'Emergencia_2': L_eme * 0.70,
        'Emergencia': L_eme,
        'Excepcionalitat': L_exc,
        'Alerta': L_ale,
        'Prealerta': L_pre
    }


# ----------------------------------------------------------------
# ----------------------------------------------------------------

