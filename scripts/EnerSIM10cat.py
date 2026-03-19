# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 12:45:37 2025

@author: victor.garcia.carrasco@upc.edu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from numba import njit
from typing import Dict, Tuple

from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sodapy import Socrata
from scipy.optimize import minimize_scalar
from statsmodels.tsa.seasonal import seasonal_decompose
from joblib import Parallel, delayed

#%%
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

def hydrogen(b, gap):
    """
    Simula la producción y consumo de hidrógeno en respuesta a excedentes o déficits energéticos.
    (para simular la producción de metano sintético basta con modificar el valor de la eficiencia)
    
    Parámetros:
    b (list): Características de la producción de hidrógeno
        b <sup> </sup>: Potencia máxima de producción de hidrógeno (kW)
        b[1]: Potencia máxima de consumo de hidrógeno (kW)
        b[2]: Capacidad total de almacenamiento de hidrógeno (kg)
        b[3]: Eficiencia de producción de hidrógeno (kg/kWh)
        b[4]: Consumo base de energía (kW)
        b[-1]: Estado de almacenamiento actual de hidrógeno (kg)
    gap (array): Diferencia entre generación y demanda en cada intervalo (kW)
                 Valores negativos indican excedente, positivos indican déficit
    
    Retorna:
    b_t (array): Estado de almacenamiento de hidrógeno en cada intervalo
    power_t (array): Potencia suministrada/absorbida para la producción/consumo de hidrógeno en cada intervalo
    """
    
    b_t = np.zeros(len(gap))  # Array para almacenar el estado de almacenamiento en cada intervalo
    power_t = np.zeros(len(gap))  # Array para almacenar la potencia de producción/consumo en cada intervalo

    max_production_power = b[0]
    max_consumption_power = b[1]
    total_storage_capacity = b[2]
    production_efficiency = b[3]
    base_consumption = b[4]
    current_storage = b[-1]

    for k, g in enumerate(gap):
        if g < 0:  # Fase de producción (hay excedente de energía)
            power = min(max_production_power, abs(g)) * np.sign(g)
            power = -min(abs(power), (total_storage_capacity - current_storage) / production_efficiency)
            current_storage -= power * production_efficiency
        elif g > base_consumption:  # Fase de consumo (hay déficit de energía)
            power = min(max_consumption_power, g - base_consumption)
            power = min(power, current_storage)
            current_storage -= power
        else:  # No hay consumo adicional, solo el consumo base
            power = 0

        # Registrar el nuevo estado de almacenamiento y la potencia suministrada/absorbida
        b_t[k] = current_storage
        power_t[k] = power
        
    return b_t, power_t

# Funciones de gestión de la generación hidráulica
def calcular_hidraulica(df_mes, hydro_min, hydro_max, f = 1):
    """
    Calcula la generación hidráulica horaria para un mes dado.
    
    :param df_mes: DataFrame con datos del mes
    :param hydro_max: Límite máximo de generación hidráulica
    :return: Serie con la generación hidráulica horaria
    """
    df_temp = df_mes.copy()
    df_temp['Hidráulica'] = (df_temp['gap'] * f).clip(lower=hydro_min,upper=hydro_max)
    return df_temp['Hidráulica']

def objetivo(hydro_max, df_mes, energia_objetivo, hydro_min, f = 1):
    """
    Función objetivo para la optimización.
    Calcula la diferencia entre la energía hidráulica estimada y la conocida.
    
    :param hydro_max: Límite máximo de generación hidráulica (variable a optimizar)
    :param hydro_min: Límite mínimo de generación hidráulica (calculado mediante otra función)
    :param df_mes: DataFrame con datos del mes
    :param energia_objetivo: Energía disponible para turbinar en el mes
    :return: Valor absoluto de la diferencia entre energía estimada y conocida
    """
    hidraulica = calcular_hidraulica(df_mes, hydro_min, hydro_max, f)
    energia_total = hidraulica.sum()  # Suma total de energía en MWh
    return abs(energia_total - energia_objetivo)

def process_month(month, df_mes, hydro_min, potencia_max_hidraulica):
    """
    Procesa un mes específico, optimizando la generación hidráulica.
    
    :param month: Fecha del mes a procesar
    :param df_mes: DataFrame con datos del mes
    :return: Serie con la generación hidráulica optimizada para el mes
    """
    # energia_turbinada_conocida = energia_turbinada_mensual.loc[month] * 1000
    # Busca el valor más cercano hacia atrás
    # energia_turbinada_conocida = energia_turbinada_mensual.asof(month)
    # energia_turbinada_conocida = energia_turbinada_mensual.reindex([month], method='ffill').iloc[0]
    # ---> BUSCAR POR AÑO Y MES, NO POR TIMESTAMP EXACTO
    mask = (
        (energia_turbinada_mensual.index.year  == month.year) & 
        (energia_turbinada_mensual.index.month == month.month)
    )
    energia_turbinada_conocida = energia_turbinada_mensual.loc[mask].iloc[0]


    
    resultado = minimize_scalar(
        objetivo,
        args=(df_mes, energia_turbinada_conocida, hydro_min),
        bounds=(hydro_min, potencia_max_hidraulica),
        method='bounded'
    )
        
    return calcular_hidraulica(df_mes, hydro_min, resultado.x)


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
        start_idx, end_idx = df_result.index.searchsorted([mes.start_time, mes.end_time], side='right')
        boundaries.append((mes.start_time, start_idx, end_idx))

    # --- 2. BUCLE PRINCIPAL SOBRE LOS MESES ---
    for month_start, start_idx, end_idx in boundaries:
        # ... (Lógica común para el mes sin cambios) ...
        mask_int = (energia_turbinada_mensual_internes.index.year == month_start.year) & (energia_turbinada_mensual_internes.index.month == month_start.month)
        energia_mes_int = energia_turbinada_mensual_internes[mask_int].values[0] if mask_int.any() else 0.0
        mask_ebro = (energia_turbinada_mensual_ebre.index.year == month_start.year) & (energia_turbinada_mensual_ebre.index.month == month_start.month)
        energia_mes_ebro = energia_turbinada_mensual_ebre[mask_ebro].values[0] if mask_ebro.any() else 0.0
        total_energia_mes = energia_mes_int + energia_mes_ebro
        f_int = energia_mes_int / total_energia_mes if total_energia_mes > 0 else 0.0
        f_ebro = 1 - f_int
        gap_slice = gap_np[start_idx:end_idx]

        # --- CICLO DE CÁLCULO PARA CADA CUENCA (Ejemplo con EBRO) ---
        
        # ### PASO 1: OPTIMIZACIÓN IDEAL ###
        nivel_prom_ebro = level_ebro_np[start_idx:end_idx].mean()
        hydro_min_ebro = hydro_min_for_level(nivel_prom_ebro) * potencia_max_ebro
        energia_obj_ebro = energia_mes_ebro + hydro_storage_ebro
        
        if puntos_optimizacion > 0:
            potencias = np.linspace(hydro_min_ebro, potencia_max_ebro, puntos_optimizacion)
            generaciones = [np.clip(gap_slice * f_ebro, hydro_min_ebro, p).sum() for p in potencias]
            deltas_posibles = energia_mes_ebro - np.array(generaciones)
            error_obj = np.abs(deltas_posibles + hydro_storage_ebro)
            hid_max_optimo_ebro = potencias[np.argmin(error_obj)]
        else:
            res = minimize_scalar(_objetivo_optimizador, bounds=(hydro_min_ebro, potencia_max_ebro), args=(gap_slice, f_ebro, hydro_min_ebro, energia_obj_ebro))
            hid_max_optimo_ebro = res.x

        hid_ebro_mes = np.clip(gap_slice * f_ebro, hydro_min_ebro, hid_max_optimo_ebro)
        delta_ebro = energia_mes_ebro - hid_ebro_mes.sum()

        # ### PASO 2: APLICAR POLÍTICA DE OVERLOAD (AJUSTE CONDICIONAL) ###
        if nivel_prom_ebro >= level_overflow_pct_ebro and delta_ebro > 100:
            hid_ebro_mes = reajustar_por_overload_numpy(hid_ebro_mes, gap_slice, delta_ebro, potencia_max_ebro)
            # hid_ebro_mes = suavizar_excedente_numpy(hid_ebro_mes, delta_ebro, hydro_min_ebro, potencia_max_ebro)
            delta_ebro = energia_mes_ebro - hid_ebro_mes.sum() # Recalcular delta tras el ajuste

        # ### PASO 3: FORZAR LÍMITE DE SUAVIZADO (RESTRICCIÓN DURA) ###
        if delta_ebro > max_delta_ebro_mwh:
            energia_a_anadir = delta_ebro - max_delta_ebro_mwh
            # hid_ebro_mes = reajustar_por_overload_numpy(hid_ebro_mes, gap_slice, energia_a_anadir, potencia_max_ebro)
            hid_ebro_mes = suavizar_excedente_numpy(hid_ebro_mes, energia_a_anadir, hydro_min_ebro, potencia_max_ebro)
        elif delta_ebro < -max_delta_ebro_mwh:
            energia_a_reducir = abs(delta_ebro) - max_delta_ebro_mwh
            hid_ebro_mes = reducir_generacion_numpy(hid_ebro_mes, energia_a_reducir, hydro_min_ebro)
        
        # ### PASO 4: CÁLCULO FINAL Y ACTUALIZACIÓN DE ESTADO ###
        delta_ebro_final = energia_mes_ebro - hid_ebro_mes.sum()
        
        hydro_storage_ebro += delta_ebro_final
        pct_ebro = (delta_ebro_final / sensibility_ebro / max_capacity_ebro) * 100
        level_ebro_np[start_idx:] += pct_ebro
        
        # ### PASO 5: SEGURO ANTI-OVERFLOW (>100%) ###
        nivel_max = level_ebro_np[start_idx:].max()
        if nivel_max > 100:
            pct_exceso = nivel_max - 100
            energia_exceso = (pct_exceso / 100 * max_capacity_ebro) * sensibility_ebro
            hydro_storage_ebro -= energia_exceso
            level_ebro_np[start_idx:] -= pct_exceso
        
        hydro_ebro_np[start_idx:end_idx] = hid_ebro_mes

        # --- (Repetir el mismo ciclo de 5 pasos para CUENCAS INTERNAS) ---
        # ... (código idéntico al de Ebro, pero con las variables _int) ...
        nivel_prom_int = level_int_np[start_idx:end_idx].mean()
        hydro_min_int = hydro_min_for_level(nivel_prom_int) * potencia_max_int
        energia_obj_int = energia_mes_int + hydro_storage_int
        if puntos_optimizacion > 0:
            potencias = np.linspace(hydro_min_int, potencia_max_int, puntos_optimizacion)
            generaciones = [np.clip(gap_slice * f_int, hydro_min_int, p).sum() for p in potencias]
            deltas_posibles = energia_mes_int - np.array(generaciones)
            error_obj = np.abs(deltas_posibles + hydro_storage_int)
            hid_max_optimo_int = potencias[np.argmin(error_obj)]
        else:
            res = minimize_scalar(_objetivo_optimizador, bounds=(hydro_min_int, potencia_max_int), args=(gap_slice, f_int, hydro_min_int, energia_obj_int))
            hid_max_optimo_int = res.x
        hid_int_mes = np.clip(gap_slice * f_int, hydro_min_int, hid_max_optimo_int)
        delta_int = energia_mes_int - hid_int_mes.sum()
        if nivel_prom_int >= level_overflow_pct_int and delta_int > 100:
            hid_int_mes = reajustar_por_overload_numpy(hid_int_mes, gap_slice, delta_int, potencia_max_int)
            delta_int = energia_mes_int - hid_int_mes.sum()
        if delta_int > max_delta_int_mwh:
            energia_a_anadir = delta_int - max_delta_int_mwh
            # hid_int_mes = reajustar_por_overload_numpy(hid_int_mes, gap_slice, energia_a_anadir, potencia_max_int)
            hid_int_mes = suavizar_excedente_numpy(hid_int_mes, energia_a_anadir, hydro_min_int, potencia_max_int)
        elif delta_int < -max_delta_int_mwh:
            energia_a_reducir = abs(delta_int) - max_delta_int_mwh
            hid_int_mes = reducir_generacion_numpy(hid_int_mes, energia_a_reducir, hydro_min_int)
        delta_int_final = energia_mes_int - hid_int_mes.sum()
        hydro_storage_int += delta_int_final
        pct_int = (delta_int_final / sensibility_int / max_capacity_int) * 100
        level_int_np[start_idx:] += pct_int
        nivel_max = level_int_np[start_idx:].max()
        if nivel_max > 100:
            pct_exceso = nivel_max - 100
            energia_exceso = (pct_exceso / 100 * max_capacity_int) * sensibility_int
            hydro_storage_int -= energia_exceso
            level_int_np[start_idx:] -= pct_exceso
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


def remove_restrictions(
    base_level: pd.Series,
    consumo_base_diario_hm3: float = 2.05, #hm3/dia
    max_capacity_int: float = 693,
    umbrales_sequia: dict = None,
    ahorro_por_fase: dict = None
) -> tuple[pd.Series, pd.Series]:
    """
    Calcula el ahorro de agua basándose en un consumo diario de referencia y las
    restricciones aplicadas en cada fase de la sequía.

    Args:
        base_level (pd.Series): Nivel de los embalses en % (serie de tiempo).
        consumo_base_diario_hm3 (float): Consumo diario de referencia en hm³/día
                                         en condiciones de normalidad.
        max_capacity_int (float): Capacidad total de los embalses internos en hm³.
        umbrales_sequia (dict, optional): Umbrales de % para cada fase.
        ahorro_por_fase (dict, optional): Porcentaje de ahorro para cada fase.

    Returns:
        tuple[pd.Series, pd.Series]: 
            - Una serie con el ahorro acumulado en hm³.
            - Una serie con el nivel simulado de los embalses (%) sin restricciones.
    """
    # --- Parámetros por defecto ---
    if umbrales_sequia is None:
        umbrales_sequia = {
            'Prealerta': 60.0,
            'Alerta': 40.0,
            'Excepcionalidad': 25.0,
            'Emergencia': 16.0,
            'Emergencia_2': 11.0,
            'Emergencia_3': 5.5
        }
    

    # if ahorro_por_fase is None:
    #     ahorro_por_fase = {
    #         'Normalitat': 0.0,
    #         'Prealerta': 0.0,
    #         'Alerta': 0.08,
    #         'Excepcionalidad': 0.15,
    #         'Emergencia': 0.25
    #     }
        
    if ahorro_por_fase is None:
        ahorro_por_fase = {
            'Normalitat': 0.0,
            'Prealerta': 0.01,
            'Alerta': 0.12,
            'Excepcionalidad': 0.21,
            'Emergencia': 0.38,
            'Emergencia_2': 0.40,
            'Emergencia_3': 0.42
        }        

    # --- Lógica de cálculo ---

    # 1. Resamplear el nivel a frecuencia diaria para aplicar ahorros diarios
    level_diario = base_level.resample('D').ffill().ffill()

    # 2. Determinar la fase de sequía para cada día
    def get_fase_sequia(nivel_pct):
        if pd.isna(nivel_pct): return 'Normalitat'
        if nivel_pct < umbrales_sequia['Emergencia']: return 'Emergencia'
        if nivel_pct < umbrales_sequia['Excepcionalidad']: return 'Excepcionalidad'
        if nivel_pct < umbrales_sequia['Alerta']: return 'Alerta'
        if nivel_pct < umbrales_sequia['Prealerta']: return 'Prealerta'
        return 'Normalitat'
    
    fases_diarias = level_diario.apply(get_fase_sequia)
    
    # 3. Mapear cada fase a su porcentaje de ahorro
    ahorro_pct_diario = fases_diarias.map(ahorro_por_fase)

    # 4. Calcular el volumen ahorrado cada día en hm³
    #    Este es el núcleo: el ahorro es un % del consumo base, todos los días.
    ahorro_diario_hm3 = consumo_base_diario_hm3 * ahorro_pct_diario

    # 5. Calcular el ahorro acumulado a lo largo del tiempo
    ahorro_acumulado_hm3 = ahorro_diario_hm3.cumsum()
    
    # 6. Reindexar para que coincida con la serie de entrada original (semanal)
    ahorro_final_acumulado = ahorro_acumulado_hm3.reindex(base_level.index, method='ffill').fillna(0)

    # 7. Calcular el nivel simulado "sin restricciones"
    pct_per_hm3 = 100.0 / max_capacity_int
    ahorro_acumulado_pct = ahorro_final_acumulado * pct_per_hm3
    nivel_simulado = base_level - ahorro_acumulado_pct

    return ahorro_final_acumulado, nivel_simulado


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

    # Calcular nivell simulat sense restriccions
    # Si hem estalviat aigua, vol dir que sense restriccions el nivell seria MÉS BAIX.
    pct_per_hm3 = 100.0 / max_capacity_int
    ahorro_acumulado_pct = ahorro_final_acumulado * pct_per_hm3
    nivel_simulado = base_level - ahorro_acumulado_pct

    return ahorro_final_acumulado, nivel_simulado

def remove_real_desal(
    base_level: pd.Series,
    dessalacio_diaria: pd.Series,
    max_capacity_int: float
) -> pd.Series:
    """
    Devuelve la serie de nivel de embalses internos 'sin' el efecto real de desalación.
    - base_level: porcentaje hora a hora (0–100)
    - dessalacio_diaria: hm3/día de desalación real (serie diaria)
    - max_capacity_int: capacidad interna total en hm3
    """
    # 1) pasar diaria a horaria hm3/h
    real_h = (
        dessalacio_diaria
        .reindex(pd.date_range(dessalacio_diaria.index.min(),
                               dessalacio_diaria.index.max(),
                               freq='D'),
                 fill_value=0)
        .resample('h').ffill()
        / 24.0
    )
    # 2) ahorro acumulado real en hm3
    real_acum = real_h.cumsum()
    # 3) % que supone cada hm3
    pct_per_hm3 = 100.0 / max_capacity_int
    # 4) resto horario de % real
    pct_real = real_acum * pct_per_hm3
    # 5) nivel sin real
    return base_level - pct_real.reindex(base_level.index, method='ffill')


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
    steepness: float = 0.2
) -> float:
    """
    Versión optimizada con Numba de sigmoid_factor.
    """
    x = (level_pct - midpoint) * steepness
    return 1.0 / (1.0 + np.exp(x))

def find_midpoint_for_target_factor(
    target_level_pct: float,
    target_factor: float,
    steepness: float = 0.2
) -> float:
    """
    Calcula el 'midpoint' necesario para que la función sigmoide dé un
    'target_factor' específico en un 'target_level_pct'.

    Args:
        target_level_pct (float): Nivel de embalse objetivo (ej: 75.0).
        target_factor (float): Factor de potencia deseado en ese nivel (ej: 0.25).
        steepness (float): La pendiente de la curva sigmoide.

    Returns:
        float: El valor del midpoint a utilizar en la función sigmoid_factor.
    """
    # Inversión de la fórmula sigmoide para despejar el midpoint
    # x = ln((1/f) - 1)
    x = np.log((1.0 / target_factor) - 1.0)
    midpoint = target_level_pct - (x / steepness)
    return midpoint


def simulate_full_water_management(
    surpluses: pd.Series,
    level_base: pd.Series,
    thermal_generation: pd.Series,  # NUEVO: Generación térmica disponible
    base_hydro_generation: pd.Series,  # NUEVO: Generación hidro base
    max_capacity_int: float,
    consumo_base_diario_hm3: float,
    # Parámetros de desalación
    max_desal_mw: float = 30,
    min_run_hours: int = 4,
    midpoint: float = 75,
    steepness: float = 0.2,
    save_hm3_per_mwh: float = 1/3000,
    # NUEVOS: Parámetros de turbinación extra
    max_hydro_capacity_mw: float = None,
    overflow_threshold_pct: float = 95.0,  # Umbral para activar turbinación extra (toda la capacidad disponible)
    sensitivity_mwh_per_percent: float = 3000.0,  # MWh para reducir 1% del nivel
    # Parámetros de restricciones
    umbrales_sequia: dict = None,
    ahorro_por_fase: dict = None
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:  # MODIFICADO: Retorna 5 series
    """
    Simula la evolución del nivel de los embalses aplicando dinámicamente tres medidas:
    1. DESALACIÓN: Cuando hay excedentes y niveles bajos/medios
    2. RESTRICCIONES: Según fase de sequía (niveles críticos bajos)
    3. TURBINACIÓN EXTRA: Cuando niveles superan el umbral (prevención sobrellenado)
    
    NUEVO: Incluye gestión de sobrellenado mediante turbinación extra que sustituye
    generación térmica respetando la capacidad hidráulica máxima. Se activa completamente
    (100% de la capacidad disponible) cuando el nivel supera el umbral configurado.
    """
    # --- Inicialización y Parámetros por Defecto ---
    if umbrales_sequia is None:
        umbrales_sequia = {
            'Emergencia III': 5.4, 'Emergencia II': 10.95, 'Emergencia I': 16.3,
            'Excepcionalidad': 25.0, 'Alerta': 40.0, 'Prealerta': 60.0
        }
    if ahorro_por_fase is None:
        ahorro_por_fase = {
            'Normalitat': 0.0, 'Prealerta': 0.0, 'Alerta': 0.08, 'Excepcionalidad': 0.12,
            'Emergencia I': 0.23, 'Emergencia II': 0.31, 'Emergencia III': 0.38
        }

    # NUEVO: Validación parámetros turbinación
    if max_hydro_capacity_mw is None:
        raise ValueError("max_hydro_capacity_mw es requerido para la gestión de sobrellenado")

    # --- Pre-cálculos ---
    pct_per_hm3 = 100.0 / max_capacity_int
    consumo_base_horario_hm3 = consumo_base_diario_hm3 / 24.0

    # Identificar periodos válidos para desalación (lógica vectorizada ya aplicada)
    mask = surpluses > 0
    run_id = mask.ne(mask.shift()).cumsum()
    run_lengths = run_id.value_counts()
    block_lengths = run_id.map(run_lengths)
    valid_mask = (mask) & (block_lengths >= min_run_hours)
    valid_desal_hours = surpluses.index[valid_mask]

    # --- INICIO DE LA MODIFICACIÓN (Opción 2: NumPy) ---

    # 1. Convertir las Series de Pandas a arrays de NumPy para un acceso mucho más rápido en el bucle.
    level_np = level_base.copy().to_numpy(dtype=float)
    desal_mw_np = np.zeros_like(level_np)
    restriction_savings_hm3_np = np.zeros_like(level_np)
    # NUEVO: Arrays para turbinación extra
    extra_hydro_mw_np = np.zeros_like(level_np)
    thermal_reduced_mw_np = np.zeros_like(level_np)
    
    surpluses_np = surpluses.to_numpy()
    # NUEVO: Arrays para generación
    thermal_np = thermal_generation.to_numpy()
    base_hydro_np = base_hydro_generation.to_numpy()
    
    # 2. Crear una máscara booleana de NumPy para las horas de desalación válidas.
    original_index = level_base.index
    is_valid_desal_hour_np = original_index.isin(valid_desal_hours)

    # --- Función auxiliar para fases de sequía (sin cambios) ---
    def get_fase_sequia(nivel_pct):
        if nivel_pct < umbrales_sequia['Emergencia III']: return 'Emergencia III'
        elif nivel_pct < umbrales_sequia['Emergencia II']: return 'Emergencia II'
        elif nivel_pct < umbrales_sequia['Emergencia I']: return 'Emergencia I'
        elif nivel_pct < umbrales_sequia['Excepcionalidad']: return 'Excepcionalidad'
        elif nivel_pct < umbrales_sequia['Alerta']: return 'Alerta'
        elif nivel_pct < umbrales_sequia['Prealerta']: return 'Prealerta'
        else: return 'Normalitat'

    # --- Bucle de Simulación Horaria (Ampliado con Turbinación Extra) ---
    for i in range(len(level_np)):
        # Acceso directo al array de NumPy por índice entero (muy rápido)
        current_level_pct = level_np[i]
        
        # 1. CÁLCULO DE AHORRO POR RESTRICCIONES (se aplica siempre)
        fase = get_fase_sequia(current_level_pct)
        ahorro_pct = ahorro_por_fase[fase]
        ahorro_hora_hm3 = consumo_base_horario_hm3 * ahorro_pct
        restriction_savings_hm3_np[i] = ahorro_hora_hm3

        # 2. CÁLCULO DE APORTE POR DESALACIÓN (sólo si hay excedente)
        desal_hora_hm3 = 0.0
        if is_valid_desal_hour_np[i]:
            f_desal = sigmoid_factor(current_level_pct, midpoint, steepness)
            cap_desal = max_desal_mw * f_desal
            mw_usados = min(surpluses_np[i], cap_desal)
            desal_mw_np[i] = mw_usados
            desal_hora_hm3 = mw_usados * save_hm3_per_mwh

        # 3. NUEVO: CÁLCULO DE TURBINACIÓN EXTRA (prevención sobrellenado)
        extra_turbine_hm3 = 0.0
        if current_level_pct >= overflow_threshold_pct:
            # Calcular capacidad hidro disponible
            current_hydro_total = base_hydro_np[i] + extra_hydro_mw_np[i]
            available_hydro_mw = max_hydro_capacity_mw - current_hydro_total
            
            # Calcular cuánto podemos turbinar (limitado por térmica y capacidad hidro)
            max_possible_turbine_mw = min(thermal_np[i], available_hydro_mw)
            
            if max_possible_turbine_mw > 0:
                # Usar toda la capacidad disponible (sin sigmoide)
                target_turbine_mw = max_possible_turbine_mw
                
                # Registrar la turbinación extra
                extra_hydro_mw_np[i] = target_turbine_mw
                thermal_reduced_mw_np[i] = target_turbine_mw
                
                # Convertir a reducción de nivel (efecto negativo en hm³)
                extra_turbine_hm3 = -target_turbine_mw / sensitivity_mwh_per_percent

        # 4. ACTUALIZACIÓN DEL NIVEL (Suma algebraica de los tres efectos)
        # Positivo: ahorro + desalación (añaden agua/reducen consumo)
        # Negativo: turbinación extra (reduce nivel del embalse)
        total_hm3_change = ahorro_hora_hm3 + desal_hora_hm3 + extra_turbine_hm3
        
        if abs(total_hm3_change) > 1e-6:  # Solo actualizar si hay cambio significativo
            delta_pct = total_hm3_change * pct_per_hm3
            # Actualización del slice del array (la operación más crítica y ahora muy rápida)
            level_np[i:] += delta_pct
            
    # 5. Convertir los arrays de NumPy de vuelta a Series de Pandas con el índice original.
    level_final = pd.Series(level_np, index=original_index, name=level_base.name)
    desal_final = pd.Series(desal_mw_np, index=original_index, name='desal_mw')
    savings_final = pd.Series(restriction_savings_hm3_np, index=original_index, name='restriction_savings_hm3')
    # NUEVO: Series para turbinación extra
    extra_hydro_final = pd.Series(extra_hydro_mw_np, index=original_index, name='extra_hydro_mw')
    thermal_reduced_final = pd.Series(thermal_reduced_mw_np, index=original_index, name='thermal_reduced_mw')
    
    # --- FIN DE LA MODIFICACIÓN ---

    return level_final, desal_final, savings_final, extra_hydro_final, thermal_reduced_final

def seasonal_factor(
    timestamp: pd.Timestamp,
    phase_months: float = 0.0,
    amplitude: float = 0.2
) -> float:
    """
    Calcula un factor estacional sinusoidal que varía entre (1-amplitude) y 1.
    
    Args:
        timestamp: Momento temporal para calcular el factor
        phase_months: Desplazamiento de fase en meses (0 = máximo en enero)
        amplitude: Amplitud de la variación (0.2 = varía entre 0.8 y 1.0)
    
    Returns:
        float: Factor estacional entre (1-amplitude) y 1
    """
    # Convertir timestamp a mes decimal (1.0 = enero, 12.0 = diciembre)
    month_decimal = timestamp.month + (timestamp.day - 1) / timestamp.days_in_month
    
    # Calcular ángulo en radianes (2π por año, ajustado por fase)
    angle = 2 * np.pi * (month_decimal - phase_months) / 12.0
    
    # Sinusoide que va de (1-amplitude) a 1
    return 1.0 - amplitude * (1 + np.cos(angle)) / 2


@njit
def seasonal_factor_numba(
    month: float,
    day: float,
    days_in_month: float,
    phase_months: float = 0.0,
    amplitude: float = 0.2
) -> float:
    """
    Versión optimizada con Numba de seasonal_factor.
    Recibe componentes numéricos del timestamp.
    """
    # Convertir a mes decimal
    month_decimal = month + (day - 1) / days_in_month
    
    # Calcular ángulo en radianes
    angle = 2 * np.pi * (month_decimal - phase_months) / 12.0
    
    # Sinusoide que va de (1-amplitude) a 1
    return 1.0 - amplitude * (1 + np.cos(angle)) / 2


# Función wrapper para mantener compatibilidad con pandas
def seasonal_factor(
    timestamp: pd.Timestamp,
    phase_months: float = 0.0,
    amplitude: float = 0.2
) -> float:
    """
    Wrapper que mantiene la interfaz original usando la función optimizada.
    """
    return seasonal_factor_numba(
        float(timestamp.month),
        float(timestamp.day),
        float(timestamp.days_in_month),
        phase_months,
        amplitude
    )


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

def simulate_full_water_management(
    surpluses: pd.Series,
    level_base: pd.Series,
    thermal_generation: pd.Series,  # NUEVO: Generación térmica disponible
    base_hydro_generation: pd.Series,  # NUEVO: Generación hidro base
    max_capacity_int: float,
    consumo_base_diario_hm3: float,
    # Parámetros de desalación
    max_desal_mw: float = 30,
    min_run_hours: int = 4,
    midpoint: float = 75,
    steepness: float = 0.2,
    save_hm3_per_mwh: float = 1/3000,
    # NUEVOS: Parámetros estacionales
    seasonal_phase_months: float = 0.0,  # Desplazamiento de fase (0 = máximo en enero)
    seasonal_amplitude: float = 0.0,     # Amplitud (0.2 = varía entre 0.8 y 1.0)
    # NUEVOS: Parámetros de turbinación extra
    max_hydro_capacity_mw: float = None,
    overflow_threshold_pct: float = 95.0,  # Umbral para activar turbinación extra (toda la capacidad disponible)
    sensitivity_mwh_per_percent: float = 3000.0,  # MWh para reducir 1% del nivel
    # Parámetros de restricciones
    umbrales_sequia: dict = None,
    ahorro_por_fase: dict = None
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:  # MODIFICADO: Retorna 5 series
    """
    Simula la evolución del nivel de los embalses aplicando dinámicamente tres medidas:
    1. DESALACIÓN: Cuando hay excedentes y niveles bajos/medios (con factor estacional)
    2. RESTRICCIONES: Según fase de sequía (niveles críticos bajos)
    3. TURBINACIÓN EXTRA: Cuando niveles superan el umbral (prevención sobrellenado)
    
    NUEVO: Incluye gestión de sobrellenado mediante turbinación extra que sustituye
    generación térmica respetando la capacidad hidráulica máxima. Se activa completamente
    (100% de la capacidad disponible) cuando el nivel supera el umbral configurado.
    
    NUEVO: Factor estacional que modula la desalación según la época del año.
    """
    # --- Inicialización y Parámetros por Defecto ---
    if umbrales_sequia is None:
        umbrales_sequia = {
            'Emergencia III': 5.4, 'Emergencia II': 10.95, 'Emergencia I': 16.3,
            'Excepcionalidad': 25.0, 'Alerta': 40.0, 'Prealerta': 60.0
        }
    if ahorro_por_fase is None:
        ahorro_por_fase = {
            'Normalitat': 0.0, 'Prealerta': 0.0, 'Alerta': 0.08, 'Excepcionalidad': 0.12,
            'Emergencia I': 0.23, 'Emergencia II': 0.31, 'Emergencia III': 0.38
        }

    # NUEVO: Validación parámetros turbinación
    if max_hydro_capacity_mw is None:
        raise ValueError("max_hydro_capacity_mw es requerido para la gestión de sobrellenado")

    # --- Pre-cálculos ---
    pct_per_hm3 = 100.0 / max_capacity_int
    consumo_base_horario_hm3 = consumo_base_diario_hm3 / 24.0

    # Identificar periodos válidos para desalación (lógica vectorizada ya aplicada)
    mask = surpluses > 0
    run_id = mask.ne(mask.shift()).cumsum()
    run_lengths = run_id.value_counts()
    block_lengths = run_id.map(run_lengths)
    valid_mask = (mask) & (block_lengths >= min_run_hours)
    valid_desal_hours = surpluses.index[valid_mask]

    # --- INICIO DE LA MODIFICACIÓN (Opción 2: NumPy) ---

    # 1. Convertir las Series de Pandas a arrays de NumPy para un acceso mucho más rápido en el bucle.
    level_np = level_base.copy().to_numpy(dtype=float)
    desal_mw_np = np.zeros_like(level_np)
    restriction_savings_hm3_np = np.zeros_like(level_np)
    # NUEVO: Arrays para turbinación extra
    extra_hydro_mw_np = np.zeros_like(level_np)
    thermal_reduced_mw_np = np.zeros_like(level_np)
    
    surpluses_np = surpluses.to_numpy()
    # NUEVO: Arrays para generación
    thermal_np = thermal_generation.to_numpy()
    base_hydro_np = base_hydro_generation.to_numpy()
    
    # 2. Crear una máscara booleana de NumPy para las horas de desalación válidas.
    original_index = level_base.index
    is_valid_desal_hour_np = original_index.isin(valid_desal_hours)

    # --- Función auxiliar para fases de sequía (sin cambios) ---
    def get_fase_sequia(nivel_pct):
        if nivel_pct < umbrales_sequia['Emergencia III']: return 'Emergencia III'
        elif nivel_pct < umbrales_sequia['Emergencia II']: return 'Emergencia II'
        elif nivel_pct < umbrales_sequia['Emergencia I']: return 'Emergencia I'
        elif nivel_pct < umbrales_sequia['Excepcionalidad']: return 'Excepcionalidad'
        elif nivel_pct < umbrales_sequia['Alerta']: return 'Alerta'
        elif nivel_pct < umbrales_sequia['Prealerta']: return 'Prealerta'
        else: return 'Normalitat'


    # Antes del bucle
    months = np.array([ts.month for ts in original_index], dtype=np.float64)
    days = np.array([ts.day for ts in original_index], dtype=np.float64)
    days_in_months = np.array([ts.days_in_month for ts in original_index], dtype=np.float64)
    seasonal_factors = seasonal_factor_array(months, days, days_in_months, seasonal_phase_months, seasonal_amplitude)
    

    # --- Bucle de Simulación Horaria (Ampliado con Factor Estacional) ---
    for i in range(len(level_np)):
        # Acceso directo al array de NumPy por índice entero (muy rápido)
        current_level_pct = level_np[i]
        current_timestamp = original_index[i]
        
        # 1. CÁLCULO DE AHORRO POR RESTRICCIONES (se aplica siempre)
        fase = get_fase_sequia(current_level_pct)
        ahorro_pct = ahorro_por_fase[fase]
        ahorro_hora_hm3 = consumo_base_horario_hm3 * ahorro_pct
        restriction_savings_hm3_np[i] = ahorro_hora_hm3

        # 2. CÁLCULO DE APORTE POR DESALACIÓN (sólo si hay excedente)
        desal_hora_hm3 = 0.0
        if is_valid_desal_hour_np[i]:
            f_desal = sigmoid_factor_numba(current_level_pct, midpoint, steepness)
            
            # NUEVO: Aplicar factor estacional solo en condiciones de normalidad
            # if fase == 'Normalitat':
            #     f_seasonal = seasonal_factor(current_timestamp, seasonal_phase_months, seasonal_amplitude)
            #     f_desal *= f_seasonal
            
            # En el bucle
            if fase == 'Normalitat' or fase == 'Prealerta':
                f_desal *= seasonal_factors[i]

            
            cap_desal = max_desal_mw * f_desal
            mw_usados = min(surpluses_np[i], cap_desal)
            desal_mw_np[i] = mw_usados
            desal_hora_hm3 = mw_usados * save_hm3_per_mwh

        # 3. NUEVO: CÁLCULO DE TURBINACIÓN EXTRA (prevención sobrellenado)
        extra_turbine_hm3 = 0.0
        if current_level_pct >= overflow_threshold_pct:
            # Calcular capacidad hidro disponible
            current_hydro_total = base_hydro_np[i] + extra_hydro_mw_np[i]
            available_hydro_mw = max_hydro_capacity_mw - current_hydro_total
            
            # Calcular cuánto podemos turbinar (limitado por térmica y capacidad hidro)
            max_possible_turbine_mw = min(thermal_np[i], available_hydro_mw)
            
            if max_possible_turbine_mw > 0:
                # Usar toda la capacidad disponible (sin sigmoide)
                target_turbine_mw = max_possible_turbine_mw
                
                # Registrar la turbinación extra
                extra_hydro_mw_np[i] = target_turbine_mw
                thermal_reduced_mw_np[i] = target_turbine_mw
                
                # Convertir a reducción de nivel (efecto negativo en hm³)
                extra_turbine_hm3 = -target_turbine_mw / sensitivity_mwh_per_percent

        # 4. ACTUALIZACIÓN DEL NIVEL (Suma algebraica de los tres efectos)
        # Positivo: ahorro + desalación (añaden agua/reducen consumo)
        # Negativo: turbinación extra (reduce nivel del embalse)
        total_hm3_change = ahorro_hora_hm3 + desal_hora_hm3 + extra_turbine_hm3
        
        if abs(total_hm3_change) > 1e-6:  # Solo actualizar si hay cambio significativo
            delta_pct = total_hm3_change * pct_per_hm3
            # Actualización del slice del array (la operación más crítica y ahora muy rápida)
            level_np[i:] += delta_pct
            
    # 5. Convertir los arrays de NumPy de vuelta a Series de Pandas con el índice original.
    level_final = pd.Series(level_np, index=original_index, name=level_base.name)
    desal_final = pd.Series(desal_mw_np, index=original_index, name='desal_mw')
    savings_final = pd.Series(restriction_savings_hm3_np, index=original_index, name='restriction_savings_hm3')
    # NUEVO: Series para turbinación extra
    extra_hydro_final = pd.Series(extra_hydro_mw_np, index=original_index, name='extra_hydro_mw')
    thermal_reduced_final = pd.Series(thermal_reduced_mw_np, index=original_index, name='thermal_reduced_mw')
    
    # --- FIN DE LA MODIFICACIÓN ---

    return level_final, desal_final, savings_final, extra_hydro_final, thermal_reduced_final


def analyze_water_management_results(
    level_final: pd.Series,
    desal_final: pd.Series,
    savings_final: pd.Series,
    extra_hydro_final: pd.Series,
    thermal_reduced_final: pd.Series,
    overflow_threshold: float = 95.0
) -> dict:
    """
    NUEVA: Función auxiliar para analizar los resultados de la gestión integral.
    """
    results = {
        'summary': {
            'final_max_level_pct': level_final.max(),
            'final_min_level_pct': level_final.min(),
            'hours_above_threshold': (level_final > overflow_threshold).sum(),
            'total_desal_mwh': desal_final.sum(),
            'total_savings_hm3': savings_final.sum(),
            'total_extra_hydro_mwh': extra_hydro_final.sum(),
            'total_thermal_reduced_mwh': thermal_reduced_final.sum()
        },
        'monthly_stats': {
            'extra_hydro_by_month': extra_hydro_final.groupby([
                extra_hydro_final.index.year, extra_hydro_final.index.month
            ]).sum().to_dict(),
            'desal_by_month': desal_final.groupby([
                desal_final.index.year, desal_final.index.month  
            ]).sum().to_dict(),
            'level_max_by_month': level_final.groupby([
                level_final.index.year, level_final.index.month
            ]).max().to_dict()
        },
        'feasibility': {
            'max_level_exceeded': level_final.max() > 100.0,
            'status': 'infactible' if level_final.max() > 100.0 else 'factible'
        }
    }
    
    return results

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

def _crear_lookup_energia(serie_mensual: pd.Series) -> dict:
    """
    Converteix una sèrie mensual en un diccionari {(year, month): valor}.
    Accés O(1) en lloc de O(n).
    """
    return {
        (idx.year, idx.month): valor 
        for idx, valor in serie_mensual.items()
    }

# Configurar el estilo de seaborn para un aspecto más profesional
sn.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


#%% (OPCIONAL) Carga de datos energéticos del portal de transparencia "j7xc-3kfh"
energia = pd.read_csv('Indicadors_Energetics_Catalunya_20250616.csv')
energia.set_index("Data", inplace=True)

# Convertir el índice a datetime (asumiendo formato mes/año)
energia.index = pd.to_datetime(energia.index, format="%m/%Y")

# Ajustar al fin de mes
energia.index = energia.index + pd.offsets.MonthEnd(0)

# Ordenar el DataFrame por el nuevo índice
energia.sort_index(inplace=True)


energia_turbinada_mensual = energia.PBEE_Hidroelectr * 1000

# energia.CDEEBC_SaldoIntercanviElectr.sum() / energia.CDEEBC_DemandaElectr.sum()

#%% Estimación de la sensibilidad a partir de datos mensuales (mediana de diferencias y pendiente de regresión simple)
# NO EJECUTAR (OBSOLETO)

embalses = pd.read_excel('Embalses_1988-2025.xlsx', decimal=',')

# embassaments_ebre = embalses.loc[embalses['EMBALSE_NOMBRE'].isin(["Camarasa", "Canelles", "Ciurana", "Santa Ana", "Oliana", "Terradets", "Ribarroja", "Cavallers", "San Lorenzo", "Guiamets", "Sallente", "Sistema Valle de Arán", "Sistema Lagos Espot", "Escales", "Rialb", "Tremp ó Talárn" , "Sistema Capdella", "Certescáns"])]
embassaments_ebre = embalses.loc[embalses['EMBALSE_NOMBRE'].isin(["Baserca", "Camarasa", "Canelles", "Ciurana", "Santa Ana", "Oliana", "Terradets", "Ribarroja", "Cavallers", "San Lorenzo", "Guiamets", "Sallente", "Sistema Valle de Arán", "Sistema Lagos Espot", "Escales", "Rialb", "Tremp ó Talárn" , "Sistema Capdella", "Certescáns"])]
# embassaments_ebre = embalses.loc[embalses['EMBALSE_NOMBRE'].isin(["Baserca", "Camarasa", "Canelles", "Santa Ana", "Oliana", "Terradets", "Ribarroja", "Cavallers", "San Lorenzo", "Guiamets", "Escales", "Rialb", "Tremp ó Talárn"])]
embassaments_internes = embalses[embalses["AMBITO_NOMBRE"] == "Cuencas Internas de Cataluña"]
embassaments = pd.concat((embassaments_ebre,embassaments_internes))

capacidad_esp = embalses.groupby('FECHA')['AGUA_TOTAL'].sum()
niveles_esp = embalses.groupby('FECHA')['AGUA_ACTUAL'].sum()
niveles_esp = niveles_esp / capacidad_esp * 100
# test = niveles_esp.resample('D').ffill()
niveles_esp = niveles_esp.resample('D').asfreq().interpolate(method='linear')

capacidad_total = embassaments.groupby('FECHA')['AGUA_TOTAL'].sum()
capacidad_actual = embassaments.groupby('FECHA')['AGUA_ACTUAL'].sum()
capacidad = pd.concat([capacidad_total, capacidad_actual], axis=1)
capacidad.columns = ['Capacitat màxima', 'Capacitat actual']
capacidad['Nivel'] = capacidad['Capacitat actual'] / capacidad['Capacitat màxima']
max_capacity = capacidad['Capacitat màxima'].iloc[-1]

capacidad_total = embassaments_ebre.groupby('FECHA')['AGUA_TOTAL'].sum()
capacidad_actual = embassaments_ebre.groupby('FECHA')['AGUA_ACTUAL'].sum()
capacidad_ebre = pd.concat([capacidad_total, capacidad_actual], axis=1)
capacidad_ebre.columns = ['Capacitat màxima', 'Capacitat actual']
capacidad_ebre['Nivel'] = capacidad_ebre['Capacitat actual'] / capacidad_ebre['Capacitat màxima']
max_capacity_ebro = capacidad_ebre['Capacitat màxima'].iloc[-1]

capacidad_total = embassaments_internes.groupby('FECHA')['AGUA_TOTAL'].sum()
capacidad_actual = embassaments_internes.groupby('FECHA')['AGUA_ACTUAL'].sum()
capacidad_internes = pd.concat([capacidad_total, capacidad_actual], axis=1)
capacidad_internes.columns = ['Capacitat màxima', 'Capacitat actual']
capacidad_internes['Nivel'] = capacidad_internes['Capacitat actual'] / capacidad_internes['Capacitat màxima']
max_capacity_int = capacidad_internes['Capacitat màxima'].iloc[-1]

# capacidad.resample('M').mean().plot(ylabel = 'Volum embassat [hm3]', xlabel='')
# capacidad.resample('M').last()['2006-10-01':].plot(ylabel = 'Volum embassat [hm3]', xlabel='')

#%% - Cargar datos diarios del SAIH per la Conca Catalana de l'Ebre
# Tabla de correspondencia
codigo_a_nombre = {
    '002': 'Flix',
    '004': 'Ribarroja',
    '043': 'Guiamets',
    '050': 'Escales',
    '051': 'Canelles',
    '052': 'Santa Ana',
    '058': 'Talarn',
    '059': 'Terradets',
    '060': 'Camarasa',
    '061': 'Sant Llorenç',
    '062': 'Oliana',
    '063': 'Cavallers',
    '065': 'Baserca',
    '076': 'Rialb',
    '080': 'Tavascan'
    }

# Cargar todas las hojas del Excel
excel_file = pd.ExcelFile('EmbalsesEbro_diario_2025_08_24.xlsx')  # sustituye por tu nombre de archivo

# DataFrames finales
df_volumen_ebre = pd.DataFrame()
df_capacidad = pd.DataFrame()

for hoja in excel_file.sheet_names:
    df_hoja = excel_file.parse(hoja)
    
    # Extraer código del nombre de la hoja (ej: "E063_Cavallers" → "063")
    codigo = hoja.split('_')[0][1:4]  # Quita la 'E' y coge 3 dígitos
    nombre = codigo_a_nombre.get(codigo, hoja)  # Usa el nombre mapeado, o el original si no está

    # Verificar que existan las columnas
    if 'FECHA_GRUPO' in df_hoja.columns and 'ACUMULADO (hm³)' in df_hoja.columns:
        # Datos de volumen
        df_temp = df_hoja[['FECHA_GRUPO', 'ACUMULADO (hm³)']].copy()
        df_temp['FECHA_GRUPO'] = pd.to_datetime(df_temp['FECHA_GRUPO'], dayfirst=True)
        df_temp = df_temp.set_index('FECHA_GRUPO')
        df_temp.columns = [nombre]
        df_volumen_ebre = pd.concat([df_volumen_ebre, df_temp], axis=1)

    # Extraer capacidad (último valor no NaN de ACUMULADO, o de una fila fija si es constante)
    capacidad = df_hoja['ACUMULADO (hm³)'].dropna().iloc[-1]  # último valor no NaN
    df_capacidad[nombre] = [capacidad]

# Interpolar datos intermedios en el volumen
df_volumen_ebre = df_volumen_ebre.sort_index().interpolate(method='linear', limit_area='inside')
df_volumen_ebre = df_volumen_ebre.dropna()

max_capacity_ebro = 2284 #2258.6
df_pct_ebre = df_volumen_ebre.sum(axis=1) / max_capacity_ebro
df_pct_ebre.index.name = 'Data'
# Reindexar a frecuencia horaria y luego interpolar
df_pct_ebre_h = df_pct_ebre.resample('h').asfreq().interpolate(method='linear')

# # Aplanar df_capacidad (es un DataFrame con una fila)
# df_capacidad = df_capacidad.T  # Transponer: cada embalse como fila
# df_capacidad.index.name = 'Embalse'
# df_capacidad.columns = ['Capacidad (hm³)']

#%% Cargar datos diarios de Transparencia Gencat per les Conques Internes Catalanes
end_dataset_date = '2025-08-01'
end_dataset_date = '2025-11-01'
# Configuración de la conexión a Socrata
domain = "analisi.transparenciacatalunya.cat"
dataset_id = "gn9e-3qhr"

# Crear cliente Socrata
client = Socrata(domain, None)

# Obtener los datos (limita a 150000 registros, puedes ajustar este número)
df = client.get(dataset_id, limit=200000)

# Convertir results a un DataFrame de pandas
df = pd.DataFrame.from_records(df)

# Convertir la columna 'dia' al formato deseado
df['dia'] = pd.to_datetime(df['dia']).dt.date

# Convertir 'dia' a datetime64 y las columnas numéricas a float64
df['dia'] = pd.to_datetime(df['dia'])
numeric_columns = ['nivell_absolut', 'percentatge_volum_embassat', 'volum_embassat']
df[numeric_columns] = df[numeric_columns].astype(float)

df.columns = ['Dia','Embassament','Nivell absolut (msnm)','Percentatge volum embassat (%)','Volum embassat (hm3)']

df['Dia'] = pd.to_datetime(df['Dia']).dt.strftime('%Y-%m-%d')
df['Dia'] = pd.to_datetime(df['Dia'])
df.set_index('Dia', inplace=True)
# df.set_index(pd.to_datetime(df['Dia']), inplace=True)

# Diccionario de mapeo
mapeo = {
    'Embassament de Foix (Castellet i la Gornal)': 'Foix',
    'Embassament de Sant Ponç (Clariana de Cardener)': 'Sant Ponç',
    'Embassament de Riudecanyes': 'Riudecanyes',
    'Embassament de Sau (Vilanova de Sau)': 'Sau',
    'Embassament de Siurana (Cornudella de Montsant)': 'Siurana',
    'Embassament de la Llosa del Cavall (Navès)': 'La Llosa del Cavall',
    'Embassament de Darnius Boadella (Darnius)': 'Boadella',
    'Embassament de Susqueda (Osor)': 'Susqueda',
    'Embassament de la Baells (Cercs)': 'La Baells'
}

# Aplicar el cambio al campo
df['Embassament'] = df['Embassament'].map(mapeo)

# Agrupar por día y calcular la suma de las columnas numéricas
df_volumen_int = df.groupby('Dia').agg({
    'Volum embassat (hm3)': 'sum'
})
df_volumen_int.columns = ['']

max_capacity_int = 693 #693.0
df_pct_int = df_volumen_int / max_capacity_int
# Reindexar a frecuencia horaria y luego interpolar
df_pct_int_h = df_pct_int.resample('h').asfreq().interpolate(method='linear')


df_pct_total = (df_volumen_int.squeeze() + df_volumen_ebre.sum(axis=1)) / (max_capacity_int + max_capacity_ebro)
df_pct_total = df_pct_total.dropna()
df_pct_total_h = df_pct_total.resample('h').asfreq().interpolate(method='linear')

# # Crear una copia para no modificar la serie original
# serie_csv = df_volumen_int.copy()
# serie_csv = serie_csv.resample('W').last()
# # Reemplazar el índice por una secuencia numérica comenzando en 1
# serie_csv.index = range(1, len(serie_csv) + 1)
# # Guardar en CSV con columnas Time y Value
# serie_csv.to_csv("VolumTotal_internes_w.csv", header=["Capacity"], index_label="Time")


#%% - Cargar función de mínimo técnico hidráulico
energia_turbinada_mensual = pd.read_excel('generacio_cat.xlsx',decimal=',')     #pd.read_csv('.csv', index_col='Fecha', parse_dates=True).squeeze()
# energia_turbinada_mensual["fecha"] = energia_turbinada_mensual["fecha"].dt.date
energia_turbinada_mensual.set_index("fecha", inplace=True)
energia_turbinada_mensual = energia_turbinada_mensual.Hidráulica

# hydro_level = capacidad['Capacitat actual'].resample('ME').last()
hydro_level = (df_volumen_int.squeeze() + df_volumen_ebre.sum(axis=1)).dropna().resample('ME').last()

data = pd.concat([hydro_level, energia_turbinada_mensual], axis=1)
data.columns = ['Hydro_Capacity', 'Energy_Generated']
data = data.dropna()

# If non-stationary, difference
data_diff = data.diff().dropna()
# data_diff = data.diff(12).dropna()

# Cálculo de la correlación entre las series diferenciadas
corr_value = data_diff['Hydro_Capacity'].corr(data_diff['Energy_Generated'])
print(f"Correlación (Hydro_Capacity vs Energy_Generated): {corr_value:.3f}")

# Granger causality without verbose
print("Granger Causality Tests (lags 1-5):")
for cause, effect in [("Hydro_Capacity","Energy_Generated"), ("Energy_Generated","Hydro_Capacity")]:
    print(f"{cause} -> {effect} p-values:")
    res = grangercausalitytests(data_diff[[cause, effect]], maxlag=5)
    pvals = [res[i+1][0]['ssr_ftest'][1] for i in range(5)]
    print(pvals)
    
    
# Regression elasticity and sensitivity
elasticity = (data_diff['Energy_Generated'].pct_change() / data_diff['Hydro_Capacity'].pct_change()).dropna()
print(f"Elasticidad promedio: {elasticity.mean():.3f}")

# Sensibility
mwh_por_hm3 = (data_diff['Energy_Generated']) / data_diff['Hydro_Capacity']
print(f"Sensibilidad mediana: {mwh_por_hm3.median():.2f} MWh/hm3")

from sklearn.utils import resample

# Datos originales para el cociente
cocientes = (data_diff['Energy_Generated']) / data_diff['Hydro_Capacity']
n_boot = 10000  # número de réplicas bootstrap
medianas_boot = []

for _ in range(n_boot):
    muestra = resample(cocientes, replace=True)
    medianas_boot.append(muestra.median())

error_mediana_cocientes = np.std(medianas_boot)
mediana_estimada = np.median(medianas_boot)  # o cocientes.median()

print(f"Mediana de cocientes: {mediana_estimada:.4f}")
print(f"Error estimado (bootstrap): {error_mediana_cocientes:.4f}")

# plt.hist(mwh_por_hm3[np.isfinite(mwh_por_hm3)], bins=20, range=(-4000, 4000))
# plt.show()


# MWh per hm3 via regression
X = data_diff['Hydro_Capacity'].values.reshape(-1,1)
y = data_diff['Energy_Generated'].values # MWh
lr = LinearRegression(fit_intercept=False).fit(X, y)
y_pred = lr.predict(X)
print(f"Slope (MWh per hm3): {lr.coef_[0]:.2f}")

# Calcular el error estimado de la pendiente (asumiendo error en y = 0.5 para todos los puntos)
# la mitad del último dígito.
suma_x_cuadrado = np.sum(X.flatten()**2)
# Error basado en suposición teórica (σ_y = 0.5)
error_pendiente = np.sqrt((0.5)**2 / suma_x_cuadrado)
print(f"Error estimado de la pendiente (teórico σ=0.5): {error_pendiente:.4f}")

# Error basado en residuos del ajuste
residuos = y - y_pred
n = len(y)
mse = np.sum(residuos**2) / (n - 1)  # sin intercepto → n-1 GL
error_pendiente_residuos = np.sqrt(mse / suma_x_cuadrado)
print(f"Error pendiente (residuos):     {error_pendiente_residuos:.4f}")

# Calcular el coeficiente R-cuadrado (R²)
r2 = r2_score(y, lr.predict(X))

plt.figure(figsize=(10, 6))
plt.scatter(X, y)
plt.plot(X, y_pred, 'r-', linewidth=2)
# plt.xlabel('Reservoir Capacity (hm$^3$)')
plt.xlabel('Diferencies mensuals en les reserves dels embassaments catalans (hm$^3$)')
# plt.ylabel('Monthly Generation (MWh)')
plt.ylabel('Diferencies mensuals en la generació hidràulica a Catalunya (MWh)')

# Añadir texto de R² al gráfico
plt.text(0.05, 0.95, f'R² = {r2:.2f}\ny = {lr.coef_[0]:.2f}x', # + {modelo.intercept_:.2f}',
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

# # Añadir la marca de agua
# plt.text(1, 0.025, '@victorbcn.bsky.social', 
#          fontsize=20, 
#          color='grey', 
#          ha='right', 
#          va='center', 
#          # rotation = 90,
#          alpha=0.5, 
#          transform=plt.gca().transAxes)

plt.show()

# -----------------------------------------
generacion_v2 = pd.read_excel(
    "generacion_v3.xlsx",
    index_col="fecha",
    parse_dates=True
)

# 1) Bins de Hydro_Level
bins = np.arange(0, 101, 10)   # cada 10 %
labels = (bins[:-1] + bins[1:]) / 2
df = generacion_v2[['Hydro_Level','Hidráulica']].dropna()
# df.index = df.index.date
# df['Hydro_Level'] = niveles_esp[df.index]

# df = df[df['Hidráulica'] >= hydro_min]  # sólo nos quedamos con generación >= hydro_min
df['Hidráulica'] = df['Hidráulica']/17095
df['bin'] = pd.cut(df['Hydro_Level'], bins=bins, labels=labels)

# 2) Percentil 1% en cada bin
min_por_bin = df.groupby('bin', observed=False)['Hidráulica'].quantile(0.01).dropna()
x = min_por_bin.index.astype(float)
y = min_por_bin.values

from scipy.interpolate import interp1d


# x,y: tus puntos originales (centros de bin vs percentil)

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

# 3) (Opcional) generar una curva extendida para graficar
x_ext = np.linspace(30, 100, 121)
y_ext = [hydro_min_for_level(xi) * 100 for xi in x_ext]

plt.figure(figsize=(8,4))
plt.plot(x_ext, y_ext, 'C2--',  label='Quadratic Interpolation Function')
plt.scatter(x, y*100/24,   c='C0',  label='Empirical percentile (1%)')
plt.xlim(30,100)
plt.xlabel('Hydro Capacity Level [%]')
plt.ylabel('Minimum Hydro Generation per hour [%]')
plt.legend()
plt.show()

#%% (OPCIONAL) Tests cointegració i GRAFICOS

fig, ax1 = plt.subplots(figsize=(12, 6))

# Color per a cada eix
color_capacity = 'tab:blue'
color_energy = 'tab:red'

# Eix Y primari (esquerra) per a la Capacitat
ax1.set_xlabel('Data')
ax1.set_ylabel('Variació Mensual Volum Embassat (hm³)', color=color_capacity, fontsize=12)
ax1.plot(data_diff.index, data_diff['Hydro_Capacity'], color=color_capacity, linestyle='-', marker='o', markersize=4, label='Variació Volum (hm³)')
ax1.tick_params(axis='y', labelcolor=color_capacity)
ax1.grid(True, linestyle='--', alpha=0.6)

# Crea el segon eix Y que comparteix el mateix eix X
ax2 = ax1.twinx()

# Eix Y secundari (dreta) per a l'Energia
ax2.set_ylabel('Variació Mensual Generació (MWh)', color=color_energy, fontsize=12)
ax2.plot(data_diff.index, data_diff['Energy_Generated'], color=color_energy, linestyle='--', marker='x', markersize=4, label='Variació Generació (MWh)')
ax2.tick_params(axis='y', labelcolor=color_energy)

# Afegir una línia a y=0 per a millor visualització
ax1.axhline(0, color='gray', linewidth=0.8, linestyle=':')

# Títol i llegenda
fig.suptitle('Variacions Mensuals de Volum Embassat i Generació Hidràulica a Catalunya', fontsize=16)
# Demanar les llegendes dels dos eixos i mostrar-les en una única caixa
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

fig.tight_layout(rect=[0, 0, 1, 0.96]) # Ajusta per al títol
plt.show()


from sklearn.preprocessing import StandardScaler

# Estandaritzar les dades (Z-score)
scaler = StandardScaler()
data_diff_scaled = pd.DataFrame(scaler.fit_transform(data_diff), 
                                index=data_diff.index, 
                                columns=data_diff.columns)

# Crear el gràfic
plt.figure(figsize=(12, 6))

plt.plot(data_diff_scaled.index, data_diff_scaled['Hydro_Capacity'], 
         color='tab:blue', linestyle='-', marker='o', markersize=4, 
         label='Variació Volum (Estandarditzat)')

plt.plot(data_diff_scaled.index, data_diff_scaled['Energy_Generated'], 
         color='tab:red', linestyle='--', marker='x', markersize=4, 
         label='Variació Generació (Estandarditzat)')

# Estil del gràfic
plt.title('Diferencies Mensuals Estandarditzades de Volum i Generació (D_02)', fontsize=16)
plt.xlabel('Data')
plt.ylabel('Desviacions Estàndard (Z-score)', fontsize=12)
plt.axhline(0, color='gray', linewidth=0.8, linestyle=':')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Realitzar el test de cointegració de Johansen
# det_order = -1 (sense terme determinista), 0 (constant), 1 (tendència)
# k_ar_diff = número de retards a incloure, generalment 1 o 2 per a dades mensuals
result = coint_johansen(data, det_order=0, k_ar_diff=1)

# Imprimir els resultats de la traça (trace statistic)
print("Resultats del Test de Cointegració de Johansen (Estadístic de la Traça)\n")
print("Hipòtesi Nul·la (r): r=0       r<=1")
print(f"Valor de l'estadístic:  {result.lr1[0]:.2f}    {result.lr1[1]:.2f}")
print(f"Valor crític (95%):   {result.cvt[0, 1]:.2f}    {result.cvt[1, 1]:.2f}\n")

# Interpretació dels resultats
if result.lr1[0] > result.cvt[0, 1]:
    print("Conclusió: Es rebutja la hipòtesi nul·la de r=0 (no cointegració).")
    print("Hi ha evidència d'almenys UNA relació de cointegració entre les sèries.")
    if result.lr1[1] > result.cvt[1, 1]:
        print("A més, es rebutja la hipòtesi nul·la de r<=1, suggerint que ambdues sèries són estacionàries.")
    else:
        print("No es pot rebutjar la hipòtesi nul·la de r<=1, confirmant UNA relació de cointegració.")
else:
    print("Conclusió: No es pot rebutjar la hipòtesi nul·la de r=0.")
    print("No hi ha evidència d'una relació de cointegració entre les sèries.")
#%% - Càrrega de dades horaries peninsulars
#--------------------------------------------------------
#Carga de datos y preparación de la tabla de datos
generacion = pd.read_excel('GeneracionSpain_Horas4.xlsx', index_col='Fecha')
# generacion = generacion[:'2021-11-01']
generacion['Eólica'] = generacion.Eólica0 * generacion.PotenciaEol.iloc[-1]/generacion.PotenciaEol
generacion['Fotovoltaica'] = generacion.FV0 * generacion.PotenciaFV.iloc[-1]/generacion.PotenciaFV
generacion['Cogeneracion'] = generacion.Cogeneración * generacion.PotenciaCog.iloc[-1]/generacion.PotenciaCog

generacion.loc[generacion.Termosolar < 0, 'Termosolar'] = 0
generacion.loc[generacion.Fotovoltaica < 0, 'Fotovoltaica'] = 0
generacion.loc[(generacion.index.month == 10) & ((generacion.index.hour <= 6)|(generacion.index.hour >= 18)), 'Fotovoltaica'] = 0
generacion.loc[(generacion.index.month >= 11) & ((generacion.index.hour <= 6)|(generacion.index.hour >= 19)), 'Fotovoltaica'] = 0
generacion.loc[(generacion.index.month <= 2) & ((generacion.index.hour <= 6)|(generacion.index.hour >= 19)), 'Fotovoltaica'] = 0
generacion.loc[(generacion.index.month == 3) & ((generacion.index.hour <= 5)|(generacion.index.hour >= 19)), 'Fotovoltaica'] = 0
generacion.loc[(generacion.index.month == 4) & ((generacion.index.hour <= 4)|(generacion.index.hour >= 20)), 'Fotovoltaica'] = 0
generacion.loc[(generacion.index.month == 5) & ((generacion.index.hour <= 4)|(generacion.index.hour >= 21)), 'Fotovoltaica'] = 0
generacion.loc[(generacion.index.month == 6) & ((generacion.index.hour <= 4)|(generacion.index.hour >= 21)), 'Fotovoltaica'] = 0
generacion.loc[(generacion.index.month == 7) & ((generacion.index.hour <= 4)|(generacion.index.hour >= 21)), 'Fotovoltaica'] = 0
generacion.loc[(generacion.index.month == 8) & ((generacion.index.hour <= 4)|(generacion.index.hour >= 21)), 'Fotovoltaica'] = 0
generacion.loc[(generacion.index.month == 9) & ((generacion.index.hour <= 5)|(generacion.index.hour >= 20)), 'Fotovoltaica'] = 0

# % Renovables total histórico
generacion['wRen0'] = (generacion.Eólica0 + generacion.FV0 + generacion.Termosolar + generacion.Hidráulica) / generacion.Demanda
# % Renovables no hidráulicas histórico
generacion['wRen1'] = (generacion.Eólica0 + generacion.FV0 + generacion.Termosolar) / generacion.Demanda
# % Renovables no hidráulicas reescalado / ajustado
generacion['wRen2'] = (generacion.Eólica + generacion.Fotovoltaica + generacion.Termosolar) / generacion.Demanda


#%% - Ajust i construcció de dades horaries sintètiques per a Catalunya

generacio = pd.read_excel('generacio_cat.xlsx', decimal=',',index_col='fecha')
generacio.drop(['Fuel + Gas'], axis=1, inplace=True)
generacio.columns = [
    'Cicles', 'Cogeneració', 'Eòlica', 'Total', 'Hidràulica', 'Nuclear', 'AltresRen', 'ResidusNR', 'ResidusR', 'Fotovoltaica', 'Termosolar'
] 

# Ruta al archivo Excel
file_path = 'nuclears.xlsx'

# Cargar cada hoja en un DataFrame separado
Asco1 = pd.read_excel(file_path, sheet_name=0, index_col = 0, parse_dates = True)  # Primera hoja (índice 0)
Asco1.astype(float)
Asco2 = pd.read_excel(file_path, sheet_name=1, index_col = 0, parse_dates = True)  # Segunda hoja (índice 1)
Asco2.astype(float)
Vandellos2 = pd.read_excel(file_path, sheet_name=2, index_col = 0, parse_dates = True)  # Tercera hoja (índice 2)
Vandellos2.astype(float)

# Elimino duplicados
Asco1 = Asco1[~Asco1.index.duplicated(keep='first')]
Asco2 = Asco2[~Asco2.index.duplicated(keep='first')]
Vandellos2 = Vandellos2[~Vandellos2.index.duplicated(keep='first')]

# Unir los DataFrames
nuclears_df = pd.concat([Asco1, Asco2, Vandellos2], axis=1, join='outer')
nuclears_df.columns = ['Asco1', 'Asco2', 'Vandellos2']
# nuclears = nuclears[:-1]
nuclears_df = nuclears_df.interpolate(method='linear')
nuclears_mensual = ((nuclears_df.sum(axis=1)) * 24).resample('ME').sum()
nuclears_base = nuclears_df.resample('h').ffill() * 0.973

nuclears = nuclears_df.sum(axis=1).resample('h').ffill() # Considerem els tres reactors


# # Máscara para todos los meses de octubre, noviembre y diciembre (cualquier año)
# mascara_oct_dic = nuclears.index.month >= 10
# nuclears = nuclears_df[['Asco2','Vandellos2']].sum(axis=1).resample('h').ffill() # Asco I tancat
nuclears = nuclears * 0.973
# corrector_nuclear = (nuclears.resample('ME').sum() / generacio.Nuclear).dropna()
# nuclears = corrector_nuclear.mean() * nuclears

generacio.Nuclear.plot()
nuclears_mensual[:-1].plot()
plt.show()

# -----------------------------------

from load_data_enercat import load_and_process_electricity_demand_data
demanda = load_and_process_electricity_demand_data('DemandaMWh_20250618.csv', freq = 'hourly')
demanda = demanda[demanda > 0.01]
demanda = demanda.interpolate(method='linear')



df_sintetic = pd.concat((demanda['2013-01-01':'2025-01-01'],nuclears['2013-01-01':'2025-01-01']),axis=1)
df_sintetic.columns = ['Demanda','Nuclear']

potencia = pd.read_excel('potencia_cat.xlsx', index_col='Fecha', decimal=',')
potencia.columns = [
    'Cicles', 'Cogeneració', 'Eòlica', 'Hidràulica', 'Nuclear', 'AltresRen', 'Total', 'ResidusNR', 'ResidusR', 'Fotovoltaica', 'Termosolar'
]  

# --------------------------------------
# Asumimos que 'demanda' es una pd.Series con índice datetime y freq='H'
# Si no tiene freq, asigna: demanda = demanda.asfreq('H')
demanda = demanda.squeeze()
demanda = demanda.asfreq('h')
# demanda = demanda[:'2024-12-31']
# 1. Calcular una tendencia suave usando una media móvil de 1 año (365 días)
ventana_anual = 365 * 24  # horas en un año

# Media móvil centrada (requiere pandas >= 1.0)
tendencia = demanda.rolling(window=ventana_anual, center=True, min_periods=1).mean()

# 2. Decidir: ¿modelo aditivo o multiplicativo?
# En demanda eléctrica, suele ser multiplicativo (crecimiento % constante)
demanda_w = demanda / tendencia
# demanda_w[-24*7:].plot()

# 3. Reescalar para que la suma total sea igual a la original
factor = demanda.sum() / demanda_w.sum()
demanda_w = demanda_w * factor
# demanda_w[-24*7:].plot()

#---------------------------
#Reescalado con base en 2024
# Calcular el nivel de referencia
tendencia_2024 = tendencia.loc['2024'].mean()

# Añadir año al índice
dfdem = pd.DataFrame({'demanda': demanda, 'tendencia': tendencia})
dfdem['año'] = dfdem.index.year

# Nivel de tendencia promedio por año
nivel_por_año = dfdem.groupby('año')['tendencia'].mean()

# Factor = nivel_2024 / nivel_año
factor_por_año = tendencia_2024 / nivel_por_año

# Asegurar que 2024 tenga factor = 1
factor_por_año[2024] = 1.0

# Mapear el factor a cada hora
dfdem['factor'] = dfdem['año'].map(factor_por_año)

# Nueva serie: demanda_w = demanda * factor
demanda_w = demanda * dfdem['factor']


# Añadir columnas útiles
dfdem = pd.DataFrame({'demanda': demanda, 'demanda_w': demanda_w})
dfdem['hora'] = dfdem.index.hour
dfdem['mes'] = dfdem.index.month
dfdem['año'] = dfdem.index.year
dfdem['es_laborable'] = ~dfdem.index.weekday.isin([5, 6])

# Función para calcular perfil horario por mes y tipo de día
def calcular_perfiles(df, col_demanda='demanda'):
    perfiles = []
    for (año, mes, laborable), grupo in df.groupby(['año', 'mes', 'es_laborable']):
        perfil = grupo.groupby('hora')[col_demanda].mean()
        # Normalizar por la media diaria del mes para comparar formas
        # perfil_norm = perfil / perfil.mean()
        perfil_norm = perfil
        for hora, valor in perfil_norm.items():
            perfiles.append({
                'año': año,
                'mes': mes,
                'es_laborable': laborable,
                'hora': hora,
                'perfil_norm': valor
            })
    return pd.DataFrame(perfiles)

perfiles_orig = calcular_perfiles(dfdem, 'demanda')
perfiles_estac = calcular_perfiles(dfdem, 'demanda_w')

años_a_mostrar = sorted(perfiles_orig['año'].unique())[-10:]  # últimos 10
perfiles_filtrado = perfiles_orig[perfiles_orig['año'].isin(años_a_mostrar)]

# años_a_mostrar = sorted(perfiles_estac['año'].unique())[-10:]  # últimos 10
# perfiles_filtrado = perfiles_estac[perfiles_estac['año'].isin(años_a_mostrar)]

# Ej: ver cómo evoluciona el perfil en enero (mes=1), laborables
enero_lab = perfiles_orig[(perfiles_orig['mes'] == 5) & (perfiles_orig['es_laborable'])]
enero_lab = perfiles_estac[(perfiles_estac['mes'] == 1) & (perfiles_estac['es_laborable'])]
enero_lab = perfiles_filtrado[(perfiles_filtrado['mes'] == 5) & (perfiles_filtrado['es_laborable'])]

# plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
for año, grupo in enero_lab.groupby('año'):
    plt.plot(grupo['hora'], grupo['perfil_norm'], label=año, alpha=0.7)
plt.title("Evolució del perfil horari normalitzat - Gener (feiners)")
plt.xlabel("Hora del dia")
plt.ylabel("Demanda / promig diari del mes")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------

# autoconsum = pd.read_excel('C:/Users/tirki/Dropbox/Trabajos/Energía/RAC_3.xlsx')
# fotovoltaica = autoconsum[autoconsum['I_TECNOLOGIA'].str.contains('fotovoltaica', case=False, na=False)]
# fotovoltaica_mes = fotovoltaica.groupby(pd.Grouper(key='I_DAT_PEM', freq='ME'))['I_KW_TOT'].sum()
# # fotovoltaica_mes = fotovoltaica.groupby(pd.Grouper(key='I_DAT_PEM', freq='Q'))['I_KW_TOT'].sum()
# fotovoltaica_mes = fotovoltaica_mes[fotovoltaica_mes.index.year >= 2013]
# fotovoltaica_mes = fotovoltaica_mes / 1000
# fotovoltaica_mes_cum = fotovoltaica_mes.cumsum()
# fotovoltaica_mes_cum.to_csv('autoconsum.csv')

fotovoltaica_mes_cum = pd.read_csv('autoconsum.csv', index_col='I_DAT_PEM')
fotovoltaica_mes_cum.index = pd.to_datetime(fotovoltaica_mes_cum.index)
fotovoltaica_mes_cum.columns = ['MW']

# plt.bar(fotovoltaica_mes.index, fotovoltaica_mes, width=30)
# plt.xticks(range(len(dates)), dates, rotation=90)
# plt.ylabel('Potencia instalada en autoconsum [MW]')


potencia['Autoconsum'] = fotovoltaica_mes_cum
# Establir el valor conegut de desembre
potencia.loc['2024-12-31', 'Autoconsum'] = 1381.0 #ICAEN Balanç elèctric de Catalunya
# Forçar NaN als mesos que vols interpolar (juliol-novembre)
potencia.loc['2024-07-31':'2024-11-30', 'Autoconsum'] = np.nan
# Interpolar linealment només aquesta secció
potencia['Autoconsum'] = potencia['Autoconsum'].interpolate(method='linear')
# potencia['Autoconsum'] = potencia['Autoconsum'].ffill()
plt.bar(potencia.Autoconsum.index, potencia.Autoconsum, width=30)
plt.ylabel('Potència acumulada en autoconsum [MW]')
plt.xlabel('')
plt.show()

end_date_range = '2024-12-31'
start_date_range = '2015-06-03'
generacion.loc[generacion.FV0 < 0, 'FV0'] = 0
factorFV = (generacion.FV0 / generacion.PotenciaFV)[start_date_range:end_date_range]
# fotovoltaica_h = potencia.Fotovoltaica.resample('h').ffill()['2015-06-03':end_date_range] * factorFV * 1000
# corrector_FV = (fotovoltaica_h.resample('ME').sum() / generacio.Fotovoltaica).dropna()
# fotovoltaica_h = ((fotovoltaica_h / corrector_FV.resample('h').ffill()).dropna()).round(0)
#------------
# solar_h = (potencia.Fotovoltaica+potencia.Termosolar).resample('h').ffill()['2015-06-03':end_date_range] * factorFV * 1000
# corrector_solar = (solar_h.resample('ME').sum() / (generacio.Fotovoltaica+generacio.Termosolar) ).dropna()
# solar_h = ((solar_h / corrector_solar.resample('h').ffill()).dropna()).round(0)

solar_h_inicial = ((potencia.Fotovoltaica+potencia.Termosolar).resample('h').ffill()[start_date_range:end_date_range] * factorFV * 1000).dropna()
# Càlcul del corrector per cada mes
corrector_mensual = ((generacio.Fotovoltaica+generacio.Termosolar) / solar_h_inicial.resample('ME').sum()).dropna()
# Aplicar el corrector de forma més precisa
solar_h_corregida = solar_h_inicial.copy()
for mes in corrector_mensual.index:
    mask_mes = (solar_h_inicial.index.to_period('M') == mes.to_period('M'))
    solar_h_corregida.loc[mask_mes] *= corrector_mensual.loc[mes]
solar_h = solar_h_corregida.round(0)
solar_h_w = solar_h * ((potencia.Fotovoltaica.iloc[-1]+potencia.Termosolar.iloc[-1])/(potencia.Fotovoltaica+potencia.Termosolar)).resample('h').ffill()

autoconsum_hourly = potencia.Autoconsum.resample('h').interpolate('linear')
demanda_w, _ = extraer_autoconsumo(df_sintetic.Demanda,solar_h_w[start_date_range:end_date_range], autoconsum_hourly, pr=0.75)
demanda_w, _ = insertar_autoconsumo(demanda_w,solar_h_w[start_date_range:end_date_range], 1206.7, pr=0.75)
df_sintetic['Demanda_w'] = demanda_w
df_sintetic = df_sintetic[['Demanda','Demanda_w','Nuclear']]

# generacion.loc[:,'Demanda_w'] = insertar_autoconsumo(
#     generacion, fv_autoconsumo, pr=0.75
# )


wind_cat = pd.read_excel("wind_cat.xlsx", index_col=0, parse_dates=True)['speed'] #m/s
wind_cat = (wind_cat - wind_cat.min()) / (wind_cat.max() - wind_cat.min())

# Crear la serie de potencia horaria
potencia_horaria = potencia.Eòlica.resample('h').ffill()[start_date_range:end_date_range]

# Si tienen diferente longitud, usar reindex sin método de relleno
wind_cat_filtrado = wind_cat[start_date_range:end_date_range]
wind_cat_aligned = wind_cat_filtrado.reindex(potencia_horaria.index)

# En lugar de usar el factor de capacidad peninsular
# factorW = (generacion.Eólica0 / generacion.PotenciaEol)['2015-06-03':]

# Generación inicial
eolica_h_inicial = (potencia_horaria * wind_cat_aligned).dropna()
# eolica_h_inicial = (potencia_horaria * cf).dropna()
# RESTRICCIÓN: No superar potencia instalada
eolica_h_inicial = eolica_h_inicial.clip(upper=potencia_horaria)

# El resto del código permanece igual
corrector_mensual = (generacio.Eòlica / eolica_h_inicial.resample('ME').sum()).dropna()

eolica_h_corregida = eolica_h_inicial.copy()
for mes in corrector_mensual.index:
    mask_mes = (eolica_h_inicial.index.to_period('M') == mes.to_period('M'))
    eolica_h_corregida.loc[mask_mes] *= corrector_mensual.loc[mes]
    eolica_h_corregida.loc[mask_mes] = eolica_h_corregida.loc[mask_mes].clip(upper=potencia_horaria.loc[mask_mes])   
eolica_h = eolica_h_corregida.round(0)
eolica_h_w = eolica_h * (potencia.Eòlica.iloc[-1]/potencia.Eòlica).resample('h').ffill()

# Càlcul més robust del factor de correció
factorCog = (generacion.Cogeneración / generacion.PotenciaCog)[start_date_range:end_date_range]
cogeneracion_h_inicial = (potencia.Cogeneració.resample('h').ffill()[start_date_range:end_date_range] * factorCog).dropna()
# Càlcul del corrector per cada mes
corrector_mensual = (generacio.Cogeneració / cogeneracion_h_inicial.resample('ME').sum()).dropna()

# # Aplicar el corrector de forma més precisa
# cogeneracion_h_corregida = cogeneracion_h_inicial.copy()
# for mes in corrector_mensual.index:
#     mask_mes = (cogeneracion_h_inicial.index.to_period('M') == mes.to_period('M'))
#     cogeneracion_h_corregida.loc[mask_mes] *= corrector_mensual.loc[mes]

# Optimització del bucle for (més ràpid que iterar mes a mes)
# 1. Aseguramos que el corrector tenga un índice de tipo Periodo (Mes)
# Esto elimina la ambigüedad del día (ya no es día 30 o 31, es "Junio 2015")
corrector_mensual.index = corrector_mensual.index.to_period('M')
# 2. Convertimos el índice horario a periodo mensual para hacer el mapeo
period_index = cogeneracion_h_inicial.index.to_period('M')
# 3. Mapeamos: Para cada hora, busca el valor de su mes correspondiente
# Esto es vectorizado y exacto
factor_alineado = corrector_mensual.loc[period_index]
# 4. Asignamos el índice horario original para poder multiplicar
factor_alineado.index = cogeneracion_h_inicial.index
# 5. Calculamos
cogeneracion_h = cogeneracion_h_inicial * factor_alineado   

cogeneracion_h_w = cogeneracion_h * (potencia.Cogeneració.iloc[-1]/potencia.Cogeneració).resample('h').ffill()
# 1. Calcular tendencia suave (media móvil centrada)
tendencia = cogeneracion_h_w.rolling(window=365 * 24, center=True, min_periods=1).mean()
# 2. Definir región de referencia (2024) → factor = 1
mask_ref = (cogeneracion_h_w.index.year >= 2024)
# 3. Calcular nivel de referencia = media de la tendencia en 2022–2024
nivel_ref = tendencia.loc[mask_ref].mean()
# 4. Factor variable: solo aplica fuera de la región de referencia
factor = pd.Series(1.0, index=cogeneracion_h_w.index)  # por defecto 1
factor.loc[~mask_ref] = nivel_ref / tendencia.loc[~mask_ref]  # ajusta el pasado
# 5. Aplicar
cogeneracion_h_w = cogeneracion_h_w * factor
cogeneracion_h_w = cogeneracion_h_w.clip(upper=potencia.Cogeneració.iloc[-1])


df_sintetic = pd.concat(([df_sintetic[start_date_range:end_date_range],cogeneracion_h[:end_date_range],solar_h[:end_date_range],eolica_h[:end_date_range], solar_h_w[:end_date_range], eolica_h_w[:end_date_range], cogeneracion_h_w[:end_date_range]]),axis=1)
df_sintetic.columns = ['Demanda','Demanda_w','Nuclear', 'Cogeneració','Solar','Eòlica','Solar_w','Eòlica_w','Cogen_w']
df_sintetic = df_sintetic.dropna()

# df_sintetic.Eòlica_w.resample('Y').sum()/df_sintetic.Demanda_w.resample('Y').sum()

color_dict = {
    'Nuclear': '#9467bd',
    'Eòlica': '#2ca02c',
    'Solar': '#ff7f0e',
    'Cogeneració': '#8c564b'
}

dia = '2023-12-31'
inicio = pd.to_datetime(dia)
fin = inicio + pd.Timedelta(hours=23)


# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_ylim([0, 6000])

# Graficar el área acumulada del DataFrame
df_sintetic[inicio:fin].iloc[:,2:6].plot.area(ax=ax, stacked=True, alpha=0.5, color=[color_dict.get(col, '#000000') for col in df_sintetic.columns[2:6]])

# Graficar la línea de la Serie
df_sintetic[inicio:fin]['Demanda_w'].plot(ax=ax, color='grey', linewidth=2)
plt.show()

#----------------------------- Sun copernicus (OPCIONAL)
sun_cat = pd.read_excel("sun_cat.xlsx", index_col=0, parse_dates=True) #m/s
sun_cat = (sun_cat - sun_cat.min()) / (sun_cat.max() - sun_cat.min())

# Crear la serie de potencia horaria
potencia_horaria = potencia.Fotovoltaica.resample('h').ffill()[start_date_range:end_date_range] + potencia.Termosolar.resample('h').ffill()[start_date_range:end_date_range]

# cf = cf.reindex(potencia_horaria.index)

sun_cat = sun_cat.sort_index()

sun_cat_filtrado = sun_cat['ssrd'][start_date_range:end_date_range]
# sun_cat_filtrado.index.duplicated().sum()
sun_cat_filtrado = sun_cat_filtrado[~sun_cat_filtrado.index.duplicated(keep='last')]


# Si tienen diferente longitud, usar reindex sin método de relleno
sun_cat_aligned = sun_cat_filtrado.reindex(potencia_horaria.index)

# Generación inicial
solar_h_inicial = (potencia_horaria * sun_cat_aligned).dropna()
# eolica_h_inicial = (potencia_horaria * cf).dropna()
# RESTRICCIÓN: No superar potencia instalada
solar_h_inicial = solar_h_inicial.clip(upper=potencia_horaria)

def fijo_a_monoeje(P_fijo: pd.Series, ganho_rel=0.20):
    """
    Convierte un perfil horario de FV fija a uno monoeje N-S (seguimiento 1 eje).
    
    P_fijo: Serie pandas con energía instantánea (o potencia media) en kW/kWp.
    ganho_rel: ganho anual del monoeje frente a fijo (ej. 0.20 = +20%).
    """
    # Hora relativa al mediodía solar (aprox usando hora local)
    horas_desde_mediodia = (P_fijo.index.hour + P_fijo.index.minute/60) - 12
    
    # Perfil de suavizado: levanta hombros y aplana pico
    factor_forma = 1 + 0.12*(1 - np.exp(-(np.abs(horas_desde_mediodia)/2.5)**1.6)) \
                     - 0.08*np.exp(-(np.abs(horas_desde_mediodia)/1.2)**2)
    
    P_tilted = P_fijo * factor_forma
    
    # Escalado de energía anual al ganho_rel
    energia_fijo = P_fijo.sum()
    energia_tilted = P_tilted.sum()
    escala = (energia_fijo * (1 + ganho_rel)) / energia_tilted
    P_1T = P_tilted * escala
    
    # Forzar cero de noche
    P_1T[P_fijo <= 0] = 0
    return P_1T

def mezcla_perfiles(P_fijo: pd.Series, frac_monoeje=0.5, ganho_rel=0.20):
    P_1T = fijo_a_monoeje(P_fijo, ganho_rel)
    return frac_monoeje * P_1T + (1 - frac_monoeje) * P_fijo

# ==== Ejemplo de uso ====
# Crear perfil monoeje
# solar_h_inicial_monoeje = fijo_a_monoeje(solar_h_inicial, ganho_rel=0.20)

# solar_h_inicial[-24*3:].plot()
# solar_h_inicial_monoeje[-24*3:].plot()

# Crear perfil mezcla 80/20
solar_h_inicial = mezcla_perfiles(solar_h_inicial, frac_monoeje=0.8, ganho_rel=0.20)


# El resto del código permanece igual
corrector_mensual = ((generacio.Fotovoltaica + generacio.Termosolar) / solar_h_inicial.resample('ME').sum()).dropna()

solar_h_corregida = solar_h_inicial.copy()
for mes in corrector_mensual.index:
    mask_mes = (solar_h_inicial.index.to_period('M') == mes.to_period('M'))
    solar_h_corregida.loc[mask_mes] *= corrector_mensual.loc[mes]
    solar_h_corregida.loc[mask_mes] = solar_h_corregida.loc[mask_mes].clip(upper=potencia_horaria.loc[mask_mes])   
solar_h = solar_h_corregida.round(0)
solar_h_w = solar_h * ((potencia.Fotovoltaica.iloc[-1]+potencia.Termosolar.iloc[-1])/(potencia.Fotovoltaica+potencia.Termosolar)).resample('h').ffill()
solar_h_w = solar_h_w.dropna()

df_sintetic.Solar = solar_h
df_sintetic.Solar_w = solar_h_w

# solar_h[-24*10:-24*3].plot()
# solar_h2[-24*10:-24*3].plot()

# solar_h_inicial[-24*4:].plot()
# solar_h_inicial2[-24*4:].plot()



#%% - GRAFICS (OPCIONAL)

# Asumimos que 'potencia' ya está definido como DataFrame

# 1. Extraemos el último valor (por ejemplo, la última fila)
last_date = potencia.index[-1]  # '2024-12-31'
data = potencia.loc[last_date]

data = generacio.resample('YE').sum().loc[last_date]

# 2. Creamos un nuevo diccionario agrupando según tus criterios
agrupat = {
    'Cicles': data['Cicles'],
    'Cogeneració': data['Cogeneració'],
    'Eòlica': data['Eòlica'],
    'Hidràulica': data['Hidràulica'],
    'Nuclear': data['Nuclear'],
    'Solar': data['Fotovoltaica'] + data['Termosolar'],  # Fotovoltaica + Termosolar
    'Altres': data['ResidusNR'] + data['ResidusR'] + data['AltresRen']  # Suma de residus i altres
}

# 3. Eliminar entradas con valor 0 o NaN (opcional)
agrupat = {k: v for k, v in agrupat.items() if pd.notna(v) and v > 0}

# 4. Preparamos datos
labels = list(agrupat.keys())
sizes = list(agrupat.values())

# 1. Obtener los colores de la paleta Set2
set2_colors = sn.color_palette("Set2", 8)  # Set2 tiene 8 colores

# 2. Asignar manualmente cada color de Set2 a una tecnología
# Mejor asignación usando Set2 para que "Solar" tenga el amarillo
color_mapping = {
    'Eòlica': set2_colors[0],        # Verde (ecológico)
    'Hidràulica': set2_colors[2],    # Azul (agua)
    'Altres': set2_colors[6],        # Marrón claro (residuos, orgánico)
    'Cicles': set2_colors[1],        # Naranja (combustión)
    'Cogeneració': set2_colors[3],   # Rosa (alta temperatura)
    'Nuclear': set2_colors[4],       # Gris (neutro, tecnológico)
    'Solar': set2_colors[5],          # ✅ Amarillo brillante (perfecto para solar)
    'Autoconsum': set2_colors[7]
}

# 3. Aplicar al gráfico
labels = list(agrupat.keys())
colors = [color_mapping[label] for label in labels]  # Asignación controlada

# 5. Paleta de colores atractiva (puedes personalizarla)
# Usamos una paleta cálida y variada de seaborn
# colors = sn.color_palette("Set2", len(labels))  # o "Set2", "viridis", "Paired", "husl", "Plasma"

# 6. Crear el gráfico
plt.figure(figsize=(10, 8))
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    textprops={'fontsize': 12, 'weight': 'bold'},
    pctdistance=0.85,  # porcentaje más cerca del centro
    wedgeprops={'linewidth': 2, 'edgecolor': 'white'}  # bordes blancos entre porciones
)

# 7. Estilo "doughnut" (quesito): agujero central
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
plt.gca().add_artist(centre_circle)

# 8. Título
plt.title(f'Potència Instalada - {last_date.strftime("%d/%m/%Y")}', 
          fontsize=16, fontweight='bold', pad=20)
plt.title(f'Electricitat Generada - {last_date.strftime("%d/%m/%Y")}', 
          fontsize=16, fontweight='bold', pad=20)

# 9. Asegurar que sea un círculo
plt.axis('equal')

# 10. Opcional: mejorar el estilo con Seaborn
sn.set_style("whitegrid")
plt.tight_layout()

# 11. Mostrar
plt.show()


# 1. Definir la agrupación de tecnologías (como antes)
def agrupar_tecnologies(df):
    df_grouped = pd.DataFrame()
    df_grouped['Cicles'] = df['Cicles']
    df_grouped['Cogeneració'] = df['Cogeneració']
    df_grouped['Eòlica'] = df['Eòlica']
    df_grouped['Hidràulica'] = df['Hidràulica']
    df_grouped['Nuclear'] = df['Nuclear']
    df_grouped['Solar'] = df['Fotovoltaica'] + df['Termosolar']
    df_grouped['Altres'] = df['ResidusNR'] + df['ResidusR'] + df['AltresRen']
    if (df.columns == 'Autoconsum').sum(): df_grouped['Autoconsum'] = df['Autoconsum']
    return df_grouped

# Aplicar agrupación al histórico completo
potencia_agrupada = agrupar_tecnologies(potencia)
potencia_agrupada = agrupar_tecnologies(generacio.rolling(12).sum())

# 3. Crear gráfico de líneas con leyenda interna
plt.figure(figsize=(10, 8))

# Dibujar una línea por cada tecnología
for col in potencia_agrupada.columns:
    plt.plot(
        potencia_agrupada.index,
        potencia_agrupada[col],
        label=col,
        color=color_mapping[col],
        linewidth=2.5,
        alpha=0.9
    )

# 4. Estilo del gráfico
plt.title('Evolució de la Potència Instalada per Tecnologia', fontsize=18, fontweight='bold', pad=20)
plt.title("Evolució de l'Electricitat Generada anualitzada per Tecnologia", fontsize=18, fontweight='bold', pad=20)

# plt.xlabel('Data', fontsize=14)
plt.ylabel('Potència Instalada (MW)', fontsize=14)

# Leyenda DENTRO del gráfico (mejor visibilidad)
plt.legend(
    loc='upper left',           # Centrado arriba
    fontsize=14,                  # Letra más grande
    ncol=2,                       # 2 columnas para ahorrar espacio vertical
    frameon=True,
    fancybox=True,
    shadow=False,
    borderpad=0.8
)

# Formato de ejes
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Cuadrícula suave
plt.grid(True, alpha=0.3, axis='y')

# Ajuste automático sin margen extra
plt.tight_layout()

# Mostrar
plt.show()
#%% CONTINUA AQUI !
start_date_range = '2016-01-01'
# Resampleas la serie semanal a horaria, rellenando hacia adelante
# hydro_hourly = capacidad['Nivel'].resample('h').ffill()
hydro_hourly = df_pct_total_h

# Reindexar el nivel embalsado sobre el índice horario de 'df_sintetic'
df_sintetic['Hydro_Level'] = 100 * hydro_hourly.reindex(
    df_sintetic.index, 
    method='ffill'
)
# (468+400)/2 = 434
#bombeig = 439.84 + 90
bombeig = 534 # segons PROENCAT (duració 10h)
sensibility = 400


scenario = 2024
if scenario == 2024:
    w_solar = 1; w_wind = 1; w_dem = 1.5; w_cog = 1
    w_wind = 1-(potencia.Eòlica['2024-12-31'] - potencia.Eòlica['2023-12-31'])/2/potencia.Eòlica['2024-12-31']
    w_solar = 1-(potencia.Fotovoltaica['2024-12-31'] - potencia.Fotovoltaica['2023-12-31'])/2/(potencia.Fotovoltaica['2024-12-31']+potencia.Termosolar['2024-12-31'])
    bat = [bombeig, bombeig*10,0.8,0] # els bombeigs son de 10h de duració
    CF_obj = None
    fv_autoconsum = potencia.Autoconsum.iloc[-1]
    fv_autoconsum = (potencia.Autoconsum['2024-12-31'] + potencia.Autoconsum['2023-12-31'])/2
    nuclears = nuclears_df[['Asco1','Asco2','Vandellos2']].sum(axis=1).resample('h').ffill()
elif scenario == 2030:
    # w_solar = 7180.8/(potencia.Fotovoltaica.iloc[-1]+potencia.Termosolar.iloc[-1]) #10  #10 #5
    w_solar = 4971.4/(potencia.Fotovoltaica.iloc[-1]+potencia.Termosolar.iloc[-1])
    w_wind = 6234.2/potencia.Eòlica.iloc[-1] #2 #4 #2
    w_dem = 1.39
    w_cog = 470.2/potencia.Cogeneració.iloc[-1] #0.5
    bat = [2234, 8936, 0.8, 0]
    CF_obj = 0.26 # Factor de capacitat de la eòlica
    fv_autoconsum = 2185.2
    factor_autoconsum = fv_autoconsum/potencia.Autoconsum.iloc[-1]
    nuclears = nuclears_df[['Asco2','Vandellos2']].sum(axis=1).resample('h').ffill() # Asco I tancat

demanda_w, autoconsum_historic = extraer_autoconsumo(df_sintetic.Demanda,df_sintetic.Solar_w[start_date_range:end_date_range], autoconsum_hourly, pr=0.75)
demanda_w = w_dem * demanda_w
demanda_w, autoconsum_estimat = insertar_autoconsumo(demanda_w,df_sintetic.Solar_w[start_date_range:end_date_range], fv_autoconsum, pr=0.75)
df_sintetic['Demanda_w'] = demanda_w

# demanda_w['2024-02-01':'2024-02-07'].plot()
# demanda['2017-02-01':'2017-02-07'].plot()

nuclears = nuclears * 0.973 # factor corrector biaix sistemàtic
df_sintetic['Nuclear'] = nuclears
solar_h_w = solar_h * ((potencia.Fotovoltaica.iloc[-1]+potencia.Termosolar.iloc[-1])/(potencia.Fotovoltaica+potencia.Termosolar)).resample('h').ffill()
df_sintetic['Solar_w'] = solar_h_w.dropna()
# eolica_h_w = eolica_h * (potencia.Eòlica.iloc[-1]/potencia.Eòlica).resample('h').ffill()
eolica_h_w = eolica_h * (potencia.Eòlica.iloc[-1]/potencia.Eòlica).resample('h').ffill()
CF_proxy = eolica_h_w.resample('YE').sum()[1:].mean() / (potencia.Eòlica.iloc[-1] * 8760) #24.3% #8760 h en un año
#df_sintetic.Eòlica.resample('YE').sum()[1:] / (potencia.Eòlica.resample('YE').mean()[1:]*8760) # valor de CF REAL
if CF_obj is not None:
    k = CF_obj/CF_proxy
else:
    k = 1
eolica_h_w = (eolica_h_w * k).clip(upper=potencia.Eòlica.iloc[-1])
df_sintetic['Eòlica_w'] = eolica_h_w.dropna()

cogeneracion_h_w = cogeneracion_h * (potencia.Cogeneració.iloc[-1]/potencia.Cogeneració).resample('h').ffill()
# 1. Calcular tendencia suave (media móvil centrada)
tendencia = cogeneracion_h_w.rolling(window=365 * 24, center=True, min_periods=1).mean()
# 2. Definir región de referencia (2024) → factor = 1
mask_ref = (cogeneracion_h_w.index.year >= 2024)
# 3. Calcular nivel de referencia = media de la tendencia en 2022–2024
nivel_ref = tendencia.loc[mask_ref].mean()
# 4. Factor variable: solo aplica fuera de la región de referencia
factor = pd.Series(1.0, index=cogeneracion_h_w.index)  # por defecto 1
factor.loc[~mask_ref] = nivel_ref / tendencia.loc[~mask_ref]  # ajusta el pasado
# 5. Aplicar
cogeneracion_h_w = cogeneracion_h_w * factor
cogeneracion_h_w = cogeneracion_h_w.clip(upper=potencia.Cogeneració.iloc[-1])
df_sintetic['Cogen_w'] = cogeneracion_h_w.dropna()
# df_sintetic.Cogeneració.resample('YE').sum()[1:] / (potencia.Cogeneració.resample('YE').mean()[1:]*8760)
# df_sintetic.Cogen_w.resample('YE').sum()[1:] / (potencia.Cogeneració.iloc[-1]*8760)

# df_sintetic.loc[:,'gap'] = w_dem * df_sintetic['Demanda_w'] - df_sintetic['Nuclear'] - w_solar * df_sintetic['Solar_w'] - w_wind * df_sintetic['Eòlica_w'] - w_cog * df_sintetic['Cogen_w']
df_sintetic.loc[:,'gap'] = df_sintetic['Demanda_w'] - df_sintetic['Nuclear'] - w_solar * df_sintetic['Solar_w'] - w_wind * df_sintetic['Eòlica_w'] - w_cog * df_sintetic['Cogen_w']

df_sintetic.loc[:,'Solar_w'] = w_solar * df_sintetic['Solar_w']
df_sintetic.loc[:,'Eòlica_w'] = w_wind * df_sintetic['Eòlica_w']
df_sintetic.loc[:,'Cogen_w'] = w_cog * df_sintetic['Cogen_w']


# Sèrie mensual en GWh
serie = cogeneracion_h.resample('ME').sum() / 1000  

plt.figure(figsize=(12, 6))
plt.plot(serie.index, serie, linewidth=2)

# Línies d’esdeveniments
plt.axvline(pd.Timestamp("2019-01-01"), color='gray', linestyle='--', linewidth=1)
plt.text(pd.Timestamp("2019-03-01"), serie.max()*0.95,
         "Màxim tendencial (≈2019)", color='gray')

# Títol i eixos en català
plt.title("Tendència de la producció de cogeneració\n"
          "(normalitzada per potència instal·lada)",
          fontsize=14, fontweight='bold')
plt.ylabel("Generació elèctrica [GWh/mes]", fontsize=12)
plt.xlabel("Any", fontsize=12)

# Suavitzat de la tendència
rolling = serie.rolling(12, center=True).mean()
plt.plot(rolling.index, rolling, linewidth=2.5, linestyle='-', alpha=0.8)

# Estètica
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

df_sintetic = df_sintetic[start_date_range:end_date_range]
#%% - Desagregació de la generació hidráulica
# 1. Parámetros de capacidad relativa
k_int = 0.1  # las internas tienen el 10% de la capacidad de Ebro
k_ebro = 1.0  # normalizamos Ebro a 1

potencia_max_hidraulica_ebro = 1374 #1713 #1557.2
potencia_max_hidraulica_int = 163 #207 #272

level_overflow_pct = 75.0
level_overflow_pct_ebro = level_overflow_pct
level_overflow_pct_int = level_overflow_pct 

sensibility_int = 323
sensibility_ebro = 434

# 2. Series de entrada:
#    - energia_turbinada_mensual: pandas Series con índice mensual
#    - capacidad_internes.Nivel: pandas Series con índice semanal
#    - capacidad_ebre.Nivel: pandas Series con índice semanal

# 3. Resamplear las series semanales de nivel a mensual (último valor disponible de cada mes)
# nivel_int_m = capacidad_internes.Nivel.resample('ME').last()
# nivel_ebro_m = capacidad_ebre.Nivel.resample('ME').last()

# 3. Resamplear las series diarias de nivel a mensual (último valor disponible de cada mes)
nivel_int_m = df_pct_int.resample('ME').last().squeeze()
nivel_ebro_m = df_pct_ebre.resample('ME').last().squeeze()

# 4. Construir DataFrame conjunto con frecuencia mensual
df = pd.DataFrame({
    'Energia': energia_turbinada_mensual,
    'Nivel_int': nivel_int_m,
    'Nivel_ebro': nivel_ebro_m
}).dropna()


# 5. Calcular pesos según capacidad relativa * nivel de embalse
#    Esto refleja potencia instalada y nivel de llenado
# ndf = df.copy()
df['w_int']  = k_int  * df['Nivel_int']
df['w_ebro'] = k_ebro * df['Nivel_ebro']

# 6. Fracciones de distribución de la generación
df['f_int']  = df['w_int']  / (df['w_int'] + df['w_ebro'])
df['f_ebro'] = 1 - df['f_int']

# 7. Desagregar la energía turbinada mensual
df['Energia_int']  = df['Energia'] * df['f_int']
df['Energia_ebro'] = df['Energia'] * df['f_ebro']

# 8. Extraer las series resultantes
energia_turbinada_mensual_internes = df['Energia_int']
energia_turbinada_mensual_ebre      = df['Energia_ebro']

# Verificación: la suma de ambas series debe coincidir con la serie original
# assert (energia_turbinada_mensual_internes + energia_turbinada_mensual_ebre).equals(energia_turbinada_mensual)

# Opcional: visualizar primeras filas
print(df[['Energia', 'Energia_int', 'Energia_ebro']].head())

# Series de entrada ya preparadas:
# - df_sintetic: DataFrame horario con columnas ['Hydro_Level_int', 'Hydro_Level_ebro']
# - energia_turbinada_mensual_internes, energia_turbinada_mensual_ebre: Series mensuales
# - hydro_min_for_level(level_pct): función que da el mínimo relativo para un nivel dado
# - potencia_max_hidraulica_int, potencia_max_hidraulica_ebro: máximos por cuenca
# - sensibility_int, sensibility_ebro: arrays / valores de transformación MWh→hm³
# - level_overflow_pct: umbral de nivel para no almacenar (> valor se descarga siempre)

# Resampleas la serie semanal a horaria, rellenando hacia adelante
# hydro_hourly_int = capacidad_internes['Nivel'].resample('h').ffill()
# hydro_hourly_ebre = capacidad_ebre['Nivel'].resample('h').ffill()
hydro_hourly_int = df_pct_int_h
hydro_hourly_ebre = df_pct_ebre_h

# Reindexar el nivel embalsado sobre el índice horario de 'df_sintetic'
df_sintetic['Hydro_Level_int'] = 100 * hydro_hourly_int.reindex(
    df_sintetic.index, 
    method='ffill'
)
df_sintetic['Hydro_Level_ebro'] = 100 * hydro_hourly_ebre.reindex(
    df_sintetic.index, 
    method='ffill'
)

df_sintetic.loc[:,'gap'] = df_sintetic['Demanda_w'] - df_sintetic['Nuclear'] - df_sintetic['Solar_w'] - df_sintetic['Eòlica_w'] - df_sintetic['Cogen_w']
#%% (OPCIONAL) Cálculo de la generación hidráulica sintética sin acumulación estacional
potencia_max_hidraulica_ebro = 1374 #928
potencia_max_hidraulica_int = 163 #155
potencia_max_hidraulica = potencia_max_hidraulica_int + potencia_max_hidraulica_ebro

energia_turbinada_mensual = pd.read_excel('generacio_cat.xlsx',decimal=',')     #pd.read_csv('.csv', index_col='Fecha', parse_dates=True).squeeze()
# energia_turbinada_mensual["fecha"] = energia_turbinada_mensual["fecha"].dt.date
energia_turbinada_mensual.set_index("fecha", inplace=True)
energia_turbinada_mensual = energia_turbinada_mensual.Hidráulica

# hydro_min = 0.05 * potencia_max_hidraulica
# -----
# hydro_min_h = pd.Series(index=df_pct_total_h.index, dtype=float)
# for i in range(len(df_pct_total_h)):
#     hydro_min_h.iloc[i] = hydro_min_for_level(df_pct_total_h.iloc[i]*100)
# -----
hydro_min_h = df_sintetic.Hydro_Level.apply(hydro_min_for_level)
hydro_sintetic = pd.Series(dtype=float)  # Serie vacía
for month, df_mes in df_sintetic.groupby(pd.Grouper(freq='MS')):
    hydro_min_mes = hydro_min_h[hydro_min_h.index.to_period('M') == month.to_period('M')]
    hydro_min_mes = hydro_min_mes.mean()
    hydro_mes = process_month(month, df_mes, hydro_min_mes, potencia_max_hidraulica)
    hydro_sintetic = pd.concat([hydro_sintetic, hydro_mes])  # Concatena mes a mes
    
hydro_sintetic.sort_index(inplace=True)  # Asegura orden temporal
df_sintetic['Hidràulica'] = hydro_sintetic


sample = df_sintetic[['Demanda_w','Nuclear','Cogen_w','Eòlica_w','Solar_w','Hidràulica']]
sample.columns = ['Demanda','Nuclear','Cogeneració','Eòlica','Solar','Hidràulica']


color_dict = {
    'Nuclear': '#9467bd',
    'Eòlica': '#2ca02c',
    'Solar': '#ff7f0e',
    'Cogeneració': '#8c564b',
    'Hidràulica': '#1f78b4'
}

dia = '2023-12-31'
inicio = pd.to_datetime(dia)
fin = inicio + pd.Timedelta(hours=23)
# fin = inicio + pd.Timedelta(hours=24*7)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_ylim([0, 6000])

# Graficar el área acumulada del DataFrame
sample[inicio:fin].iloc[:,1:].plot.area(
    ax=ax,
    stacked=True,
    alpha=0.5,
    color=[color_dict.get(col, '#000000') for col in sample.columns[1:]]
)


# Graficar la línea de la Serie
sample[inicio:fin]['Demanda'].plot(ax=ax, color='grey', linewidth=2, label='Demanda')
ax.legend(loc='upper left')
plt.show()

# sample.Hidràulica['2023-11-30':'2023-12-01']
#%% Cálculo de generación hidráulica sintética con acumulación estacional

# Resampleas la serie semanal a horaria, rellenando hacia adelante
# hydro_hourly_int = capacidad_internes['Nivel'].resample('h').ffill()
# hydro_hourly_ebre = capacidad_ebre['Nivel'].resample('h').ffill()
hydro_hourly_int = df_pct_int_h
hydro_hourly_ebre = df_pct_ebre_h

# Reindexar el nivel embalsado sobre el índice horario de 'df_sintetic'
df_sintetic['Hydro_Level_int'] = 100 * hydro_hourly_int.reindex(
    df_sintetic.index, 
    method='ffill'
)
df_sintetic['Hydro_Level_ebro'] = 100 * hydro_hourly_ebre.reindex(
    df_sintetic.index, 
    method='ffill'
)


df_sintetic.loc[:,'gap'] = df_sintetic['Demanda_w'] - df_sintetic['Nuclear'] - df_sintetic['Solar_w'] - df_sintetic['Eòlica_w'] - df_sintetic['Cogen_w']


potencia_max_hidraulica_ebro = 1374 #1713
potencia_max_hidraulica_int = 163 #155 #207

sensibility_int = 323
sensibility_ebro = 434

#umbrales de level overflow (%)
level_overflow_pct_ebro = 75.0 
level_overflow_pct_int = 75.0 


# --- 2. Llamada a la función --- (295ms) -> (288ms) -> 278ms
df_sintetic = calcular_generacion_hidraulica(
    df_sintetic=df_sintetic,
    energia_turbinada_mensual_internes=energia_turbinada_mensual_internes,
    energia_turbinada_mensual_ebre=energia_turbinada_mensual_ebre,
    potencia_max_int=potencia_max_hidraulica_int,
    potencia_max_ebro=potencia_max_hidraulica_ebro,
    sensibility_int=sensibility_int,
    sensibility_ebro=sensibility_ebro,
    max_capacity_int= max_capacity_int,
    max_capacity_ebro= max_capacity_int,
    level_overflow_pct_int=level_overflow_pct_int,
    level_overflow_pct_ebro=level_overflow_pct_ebro,
    max_salto_pct_mensual=10,
    puntos_optimizacion=0
)

# --- 3. Resultado ---
# df_hidraulica tendrá columnas:
#   - 'hydro_int': generación horaria cuenca interna
#   - 'hydro_ebro': generación horaria cuenca Ebro
#   - 'Hidráulica': suma de ambas


#%%
print(df_sintetic.Hidràulica.head())

print(df_sintetic.Hydro_Level_ebro.min())
print(df_sintetic.Hydro_Level_ebro.max())
print(df_sintetic.Hydro_Level_int.min())
print(df_sintetic.Hydro_Level_int.max())

(df_sintetic.hydro_int.resample('ME').sum() - energia_turbinada_mensual_internes)['2015-06-01':].plot()
(df_sintetic.hydro_ebro.resample('ME').sum() - energia_turbinada_mensual_ebre)['2015-06-01':].plot()
plt.show()

df_sintetic.Hydro_Level_int.plot()
df_sintetic.Hydro_Level_ebro.plot()
plt.show()

#%%
%%time
# Demanda no cubierta por las renovables no hidráulicas reescaladas y la hidráulica estimada
df_sintetic['gap0'] = df_sintetic['Demanda_w'] - df_sintetic['Solar_w'] - df_sintetic['Eòlica_w'] - df_sintetic['Hidràulica']
# Demanda no cubierta por las fuentes inflexibles reescaladas más la hidráulica estimada
df_sintetic['gap'] = df_sintetic['Demanda_w'] - df_sintetic['Nuclear'] - df_sintetic['Solar_w'] - df_sintetic['Eòlica_w'] - df_sintetic['Cogen_w'] - df_sintetic['Hidràulica']

# bat = [500, 2000, 0.8, 0] #bombeos existentes
# bat = [2234, 8936, 0.8, 0]
capacity, power = battery(bat,df_sintetic['gap'])
df_sintetic['gap'] = df_sintetic['gap'] - power
df_sintetic['gap0'] = df_sintetic['gap0'] - power

df_sintetic['Bateries'] = pd.Series(power, index=df_sintetic.index) #descàrrega
df_sintetic.loc[df_sintetic['Bateries'] < 0,'Bateries'] = 0
df_sintetic['Càrrega'] = pd.Series(power * (-1), index=df_sintetic.index)
df_sintetic.loc[df_sintetic['Càrrega'] < 0,'Càrrega'] = 0
df_sintetic['Càrrega'] = df_sintetic['Càrrega'] + df_sintetic['Demanda_w'] #* w_dem
df_sintetic['SOC'] = pd.Series(capacity, index=df_sintetic.index)

df_sintetic['Gas+Imports'] = df_sintetic['Demanda_w'] - df_sintetic['Nuclear'] - df_sintetic['Solar_w'] - df_sintetic['Eòlica_w'] - df_sintetic['Cogen_w'] - df_sintetic['Bateries'] - df_sintetic['Hidràulica']
df_sintetic.loc[df_sintetic['Gas+Imports'] < 0, 'Gas+Imports'] = 0

# autoconsum_estimat = df_sintetic['Demanda'] - df_sintetic['Demanda_w'] + autoconsum_historic
# demanda_sense_autoconsum, autoconsum_historic = extraer_autoconsumo(df_sintetic.Demanda,solar_h_w['2015-06-03':end_date_range], autoconsum_hourly, pr=0.75)
# demanda_w = insertar_autoconsumo(demanda_sense_autocon,solar_h_w['2015-06-03':end_date_range], fv_autoconsum, pr=0.75)

# --------------------------------

# sample = df_sintetic[['Demanda','Nuclear','Cogeneració','Eòlica','Solar','Hidràulica']]
sample = df_sintetic[['Demanda_w','Nuclear','Cogen_w','Eòlica_w','Solar_w','Hidràulica', 'Bateries','Gas+Imports', 'Càrrega', 'gap0']]
sample.columns = ['Demanda','Nuclear','Cogeneració','Eòlica','Solar','Hidràulica', 'Bateries','Gas+Imports', 'Càrrega', 'gap0']

start_date_range = sample.index[0]
start_date_range = '2016-01-01'
end_date_range = '2024-12-31'
sample = sample[start_date_range:end_date_range]
total_autoconsum = autoconsum_estimat[start_date_range:end_date_range]

# Define coberturas en porcentaje
total_demand = sample['Demanda'].sum()

sample = sample.copy()
sample.loc[:,'Total'] = sample['Gas+Imports'] + sample['Cogeneració'] + sample['Nuclear'] + sample['Solar'] + sample['Eòlica'] + sample['Hidràulica']
sample.loc[:,'Total2'] = sample['Gas+Imports'] + sample['Cogeneració'] + sample['Nuclear'] + sample['Solar'] + sample['Eòlica'] + sample['Hidràulica'] + total_autoconsum

# sample.Solar.sum()/sample.Total2.sum()
# total_autoconsum.sum()/sample.Total2.sum()
# total_autoconsum.resample('YE').sum()
# total_autoconsum['2024-01-01':end_date_range].sum()/1000
# df_sintetic.Solar['2024-01-01':end_date_range].sum()/1000

n_years = round(len(sample)/24/365,1)

metrics = {
    'Wind':    sample['Eòlica'].sum()        *100/ total_demand,
    'Solar':    sample['Solar'].sum()        *100/ total_demand,
    'Autoconsum': total_autoconsum.sum()     *100/ total_demand,
    'Hydro':         sample['Hidràulica'].sum()                      *100/ total_demand,
    'Nuclear':       sample['Nuclear'].sum()                    *100/ total_demand,
    'Cogeneració':        sample['Cogeneració'].sum()*100/ total_demand,
    'Cicles + Import.':        sample['Gas+Imports'].sum()*100/ total_demand,        
    'Bateries':       sample['Bateries'].sum()                    *100/ total_demand,    
    'Fossil+Imports':        (sample['Gas+Imports'] + sample['Cogeneració']).sum()*100/ total_demand,
    # 'Low-carbon':    (sample[['Eòlica','Solar','Hidràulica','Nuclear']].sum(axis=1).sum())*100/total_demand,
    'Renewables': sample[['Eòlica','Solar','Hidràulica']].sum().sum()*100/ total_demand,
    # Ren.-coverage A: las renovables se recortan (curtailment) para proteger la generación inflexible.
    'Ren.-coverage': 100 - (sample['Gas+Imports']+sample['Nuclear']+sample['Cogeneració']).sum()*100/total_demand,
    # Ren.-coverage B: nuclear y cogeneración se adaptan al mix renovable
    'Ren.cov-B': round((1-sum(sample.gap0[sample.gap0 > 0]) / sum(sample.Demanda))*100,1),
    # 'Curtailments': round((1-sum(sample.gap0[sample.gap0 > 0]) / sum(w_dem * sample.Demanda))*100,1) - (100 - (sample['Gas+Imports']+sample['Nuclear']+sample['Cogeneració']).sum()*100/total_demand),
    'Clean-coverage': 100 - (sample['Gas+Imports']+sample['Cogeneració']).sum()*100/total_demand,
    'Surpluses':   ((sample['Total'] + sample['Bateries'] - sample['Càrrega']).sum())*100/ total_demand

}

for name, val in metrics.items():
    print(f"{name:15s}: {val:.2f}%")
    
print('')

metrics = {
    'Wind':    sample['Eòlica'].sum()        /1000/n_years,
    'Solar':    sample['Solar'].sum()        /1000/n_years,
    'Autoconsum': total_autoconsum.sum()     /1000/n_years,
    'Hydro':         sample['Hidràulica'].sum()                      /1000/n_years,
    'Nuclear':       sample['Nuclear'].sum()                /1000/n_years    ,#*100/ total_demand,
    'Cogeneració':        sample['Cogeneració'].sum() /1000/n_years,
    'Cicles + Import.':        sample['Gas+Imports'].sum() /1000/n_years,
    'Bateries':       sample['Bateries'].sum()                    /1000/n_years,
    # 'Fossil+Imports':        (sample['Gas+Imports'] + sample['Cogeneració']).sum() /1000/n_years,
    # # 'Low-carbon':    (sample[['Eòlica','Solar','Hidràulica','Nuclear']].sum(axis=1).sum())*100/total_demand,
    # 'Renewables': sample[['Eòlica','Solar','Hidràulica']].sum().sum() /1000/n_years,
    # # Ren.-coverage A: las renovables se recortan (curtailment) para proteger la generación inflexible.
    # 'Ren.-coverage': 100 - (sample['Gas+Imports']+sample['Nuclear']+sample['Cogeneració']).sum()*100/total_demand,
    # # Ren.-coverage B: nuclear y cogeneración se adaptan al mix renovable
    # 'Ren.cov-B': round((1-sum(sample.gap0[sample.gap0 > 0]) / sum(w_dem * sample.Demanda))*100,1),
    # # 'Curtailments': round((1-sum(sample.gap0[sample.gap0 > 0]) / sum(w_dem * sample.Demanda))*100,1) - (100 - (sample['Gas+Imports']+sample['Nuclear']+sample['Cogeneració']).sum()*100/total_demand),
    # 'Clean-coverage': 100 - (sample['Gas+Imports']+sample['Cogeneració']).sum()*100/total_demand,
    'Surpluses':   ((sample['Total'] + sample['Bateries'] - sample['Càrrega']).sum()) /1000/n_years,

}

for name, val in metrics.items():
    print(f"{name:15s}: {val:.1f} GWh")


dem2024 = demanda.resample('YE').sum().iloc[-2]

gen = generacio['2024-01-01':]

print


# sample['Eòlica'].sum() / sample['Total'].sum()
# sample['Nuclear'].sum() / sample['Total'].sum()

# df_sintetic.Demanda.resample('ME').sum().plot()
# df_sintetic.Demanda_w.resample('ME').sum().plot()

# total_production = df_sintetic['Gas+Imports'] + df_sintetic['Cogeneració'] + df_sintetic['Nuclear'] + df_sintetic['Solar'] + df_sintetic['Eòlica'] + df_sintetic['Hidràulica']
# total_production = df_sintetic['Gas+Imports'] + df_sintetic['Cogen_w'] + df_sintetic['Nuclear'] + df_sintetic['Solar_w'] + df_sintetic['Eòlica_w'] + df_sintetic['Hidràulica']
#%%
sample = df_sintetic[['Demanda_w','Nuclear','Cogen_w','Eòlica_w','Solar_w', 'Hidràulica', 'Bateries', 'Gas+Imports', 'Càrrega']]
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
dia = '2020-05-01'
inicio = pd.to_datetime(dia)
fin = inicio + pd.Timedelta(hours=23)
fin = inicio + pd.Timedelta(hours=24*7)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_ylim([0, 6000])

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

# (df_sintetic.hydro_ebro.resample('ME').sum() - energia_turbinada_mensual_ebre)['2015-06-01':].tail(24)

# surpluses = sample['Total'] + sample['Bateries'] - sample['Càrrega']

#%% - 465 ± 22.1 ms
%%time
# --------------------------------------------------------------------------------------
# NOTA: Asegúrate de que las funciones auxiliares como 'calcular_generacion_hidraulica_final',
# 'reducir_generacion_numpy', 'reajustar_por_overload_numpy', '_objetivo_optimizador',
# y 'battery' estén definidas y disponibles en el scope de este script.
# --------------------------------------------------------------------------------------

def generar_escenario_sintetico(
    # --- DATOS DE ENTRADA (DataFrames y Series) ---
    df_demanda: pd.Series,
    df_nucleares_base: pd.DataFrame, # DataFrame con las 3 nucleares en columnas
    df_cogeneracion: pd.Series,
    df_solar: pd.Series,
    df_eolica: pd.Series,
    df_autoconsum: pd.Series,
    df_potencia_historica: pd.DataFrame, # DataFrame con la potencia histórica de cada tecnología
    df_capacidad_internes: pd.DataFrame,
    df_capacidad_ebre: pd.DataFrame,
    energia_turbinada_mensual_internes: pd.Series,
    energia_turbinada_mensual_ebre: pd.Series,
        
    # --- PARÁMETROS DE CONFIGURACIÓN DEL ESCENARIO ---
    nucleares_activas: list = [True, True, True],
    pesos: dict = {'solar': 1, 'wind': 1, 'dem': 1, 'cog': 1, 'auto': 1},
    baterias_config: list = [500, 2000, 0.8, 0],
    max_salto_hidro_pct: float = 5.0,
    optimizador_hidro: str = 'rapido', # 'rapido' o 'robusto'
       
    # --- PARÁMETROS FÍSICOS DEL MODELO ---
    potencia_max_hidro: dict = {'ebro': potencia_max_hidraulica_ebro, 'int': potencia_max_hidraulica_int},
    sensibilidad_hidro: dict = {'ebro': sensibility_ebro, 'int': sensibility_int},
    capacidad_max_hidro: dict = {'ebro': max_capacity_ebro, 'int': max_capacity_int}, #{'ebro': 2341, 'int': 677},
    umbral_overflow_pct: dict = {'ebro': 75.0, 'int': 75.0},
    
    # --- RANGO DE FECHAS ---
    # start_date: str = '2015-06-03',
    start_date: str = '2016-01-01',
    end_date: str = '2024-12-31'
) -> tuple[pd.DataFrame, dict]:
    """
    Genera un escenario energético sintético completo a partir de un conjunto de parámetros.

    Esta función encapsula todo el proceso:
    1. Construcción del DataFrame base según la configuración nuclear y de pesos.
    2. Simulación de la generación hidráulica con lógica de suavizado.
    3. Simulación del almacenamiento con baterías/bombeo.
    4. Cálculo de la generación térmica residual (Gas+Imports).
    5. Cálculo de un conjunto de métricas clave del escenario.

    :param nucleares_activas: Lista booleana [Asco1, Asco2, Vandellos2] para activar/desactivar centrales.
    :param pesos: Diccionario con los factores de reescalado para solar, eólica, demanda y cogeneración.
    :param baterias_config: Lista de configuración para la simulación de baterías [capacidad, potencia, eficiencia, soc_inicial].
    :param max_salto_hidro_pct: Límite de variación mensual del nivel de los embalses para suavizado.
    :param optimizador_hidro: 'rapido' para usar minimize_scalar, 'robusto' para usar búsqueda manual.
    :param datos_de_entrada: Todos los DataFrames y Series necesarios.
    :param parametros_fisicos: Parámetros numéricos del modelo.
    :param fechas: Fechas de inicio y fin de la simulación.
    
    :return: Una tupla conteniendo:
             - df_escenario (pd.DataFrame): El DataFrame horario completo del escenario.
             - metricas (dict): Un diccionario con las métricas de rendimiento calculadas.
    """

    # --- 1. CONFIGURACIÓN NUCLEAR ---
    nombres_nucleares = ['Asco1', 'Asco2', 'Vandellos2']
    nucleares_a_usar = [nombre for nombre, activo in zip(nombres_nucleares, nucleares_activas) if activo]
    
    if nucleares_a_usar:
        df_nuclear_total = df_nucleares_base[nucleares_a_usar].sum(axis=1).resample('h').ffill()
    else:
        df_nuclear_total = pd.Series(0, index=df_demanda.index).resample('h').ffill()

    # --- 2. CONSTRUCCIÓN Y REESCALADO DEL DATAFRAME INICIAL ---
    df_sintetic = pd.concat([
        df_demanda[start_date:end_date],
        df_nuclear_total[start_date:end_date]
    ], axis=1)
    df_sintetic.columns = ['Demanda', 'Nuclear']

   
    # Reescalado de renovables y cogeneración basado en la potencia final deseada
    solar_reescalada = df_solar * (df_potencia_historica[['Fotovoltaica', 'Termosolar']].sum(axis=1)[end_date] / df_potencia_historica[['Fotovoltaica', 'Termosolar']].sum(axis=1)).resample('h').ffill()
    eolica_reescalada = df_eolica * (df_potencia_historica['Eòlica'][end_date] / df_potencia_historica['Eòlica']).resample('h').ffill()
    cogen_reescalada = df_cogeneracion * (df_potencia_historica['Cogeneració'][end_date] / df_potencia_historica['Cogeneració']).resample('h').ffill()
    # 1. Calcular tendencia suave (media móvil centrada)
    tendencia = cogen_reescalada.rolling(window=365 * 24, center=True, min_periods=1).mean()
    # 2. Definir región de referencia (2024) → factor = 1
    mask_ref = (cogen_reescalada.index.year >= 2024)
    # 3. Calcular nivel de referencia = media de la tendencia en 2022–2024
    nivel_ref = tendencia.loc[mask_ref].mean()
    # 4. Factor variable: solo aplica fuera de la región de referencia
    factor = pd.Series(1.0, index=cogen_reescalada.index)  # por defecto 1
    factor.loc[~mask_ref] = nivel_ref / tendencia.loc[~mask_ref]  # ajusta el pasado
    # 5. Aplicar
    cogen_reescalada = cogen_reescalada * factor
    cogen_reescalada = cogen_reescalada.clip(upper=potencia.Cogeneració[end_date])

    df_sintetic['Solar_w'] = pesos['solar'] * solar_reescalada
    df_sintetic['Eòlica_w'] = pesos['wind'] * eolica_reescalada
    df_sintetic['Cogen_w'] = pesos['cog'] * cogen_reescalada
    
    df_sintetic = df_sintetic.dropna()

    # Recalculo de la demanda considerando el nuevo autoconsumo
    df_sintetic.Demanda, autoconsum_historic = extraer_autoconsumo(df_sintetic.Demanda,df_sintetic.Solar_w, df_autoconsum[start_date:end_date], pr=0.75)
    df_sintetic.Demanda = pesos['dem'] * df_sintetic.Demanda
    df_sintetic.Demanda, autoconsum_estimat = insertar_autoconsumo(df_sintetic.Demanda,df_sintetic.Solar_w, pesos['auto'] * potencia.Autoconsum[end_date], pr=0.75)


    # --- 3. CÁLCULO DEL GAP INICIAL Y NIVELES HIDRO ---
    df_sintetic['gap'] = (df_sintetic['Demanda'] - 
                          df_sintetic['Nuclear'] - 
                          df_sintetic['Solar_w'] - 
                          df_sintetic['Eòlica_w'] - 
                          df_sintetic['Cogen_w'])

    # hydro_hourly_int = df_capacidad_internes['Nivel'].resample('h').ffill()
    # hydro_hourly_ebre = df_capacidad_ebre['Nivel'].resample('h').ffill()
    hydro_hourly_int = df_capacidad_internes
    hydro_hourly_ebre = df_capacidad_ebre
    df_sintetic['Hydro_Level_int'] = 100 * hydro_hourly_int.reindex(df_sintetic.index, method='ffill')
    df_sintetic['Hydro_Level_ebro'] = 100 * hydro_hourly_ebre.reindex(df_sintetic.index, method='ffill')
    
    df_sintetic = df_sintetic.dropna()
    
    # --- 4. SIMULACIÓN DE LA GENERACIÓN HIDRÁULICA ---
    puntos_opt = 300 if optimizador_hidro == 'robusto' else 0
    df_sintetic = calcular_generacion_hidraulica( # Usamos la función final que creamos
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
        puntos_optimizacion=puntos_opt,
        hydro_min_for_level = hydro_min_for_level
    )
    df_sintetic.rename(columns={'Hidràulica': 'Hidráulica'}, inplace=True)

    df_sintetic.hydro_int = np.clip(df_sintetic.hydro_int,0,potencia_max_hidro['int'])
    df_sintetic.hydro_ebro = np.clip(df_sintetic.hydro_ebro,0,potencia_max_hidro['ebro'])
    df_sintetic.Hidráulica = df_sintetic.hydro_int + df_sintetic.hydro_ebro

    # --- 5. SIMULACIÓN DE BATERÍAS/BOMBEO ---
    # Recalculamos el gap residual después de la hidráulica sin considerar las fuentes no renovables inflexibles
    df_sintetic['gap0'] = (df_sintetic['Demanda'] - 
                          df_sintetic['Solar_w'] - 
                          df_sintetic['Eòlica_w'] - 
                          df_sintetic['Hidráulica'])
    
    # Recalculamos el gap residual después de la hidráulica
    df_sintetic['gap'] = (df_sintetic['Demanda'] - 
                          df_sintetic['Nuclear'] - 
                          df_sintetic['Solar_w'] - 
                          df_sintetic['Eòlica_w'] - 
                          df_sintetic['Cogen_w'] - 
                          df_sintetic['Hidráulica'])
      
    capacity, power = battery(baterias_config, df_sintetic['gap'])
    df_sintetic['gap'] = df_sintetic['gap'] - power
    df_sintetic['gap0'] = df_sintetic['gap0'] - power
    
    df_sintetic['Bateries'] = pd.Series(power, index=df_sintetic.index) #descàrrega
    df_sintetic.loc[df_sintetic['Bateries'] < 0,'Bateries'] = 0
    df_sintetic['Càrrega'] = pd.Series(power * (-1), index=df_sintetic.index)
    df_sintetic.loc[df_sintetic['Càrrega'] < 0,'Càrrega'] = 0
    df_sintetic['Càrrega'] = df_sintetic['Càrrega'] + df_sintetic['Demanda']
    df_sintetic['SOC'] = pd.Series(capacity, index=df_sintetic.index)
    
    # --- 6. CÁLCULO DE LA GENERACIÓN RESIDUAL (TÉRMICA + IMPORTS) ---
    df_sintetic['Gas+Imports'] = (df_sintetic['Demanda'] -
                                  (df_sintetic['Nuclear'] + df_sintetic['Solar_w'] + 
                                   df_sintetic['Eòlica_w'] + df_sintetic['Cogen_w'] + 
                                   df_sintetic['Hidráulica'] + df_sintetic['Bateries']))
    df_sintetic['Gas+Imports'] = df_sintetic['Gas+Imports'].clip(lower=0)
    
    # return df_sintetic, capacity, power
    
    # --- 7. CÁLCULO DE MÉTRICAS FINALES ---
    # Preparamos el dataframe para las métricas, usando nombres consistentes
    sample = df_sintetic.rename(columns={
        'Cogen_w': 'Cogeneració', 'Eòlica_w': 'Eòlica', 'Solar_w': 'Solar'
    })[['Demanda', 'Nuclear', 'Cogeneració', 'Eòlica', 'Solar', 'Hidráulica', 'Bateries', 'Gas+Imports', 'Càrrega', 'gap0']]
    
    total_demand = sample['Demanda'].sum()
    # Evitar division por cero si no hay demanda
    if total_demand < 1e-6:
        return df_sintetic, {}

    # Total de producción que cubre la demanda + carga de baterías
    total_supply = sample['Càrrega'] + sample['Demanda']
    
    sample.loc[:,'Total'] = sample['Gas+Imports'] + sample['Cogeneració'] + sample['Nuclear'] + sample['Solar'] + sample['Eòlica'] + sample['Hidráulica']


    metrics = {
        'Wind %': sample['Eòlica'].sum() * 100 / total_demand,
        'Solar %': sample['Solar'].sum() * 100 / total_demand,
        'Autoconsum': autoconsum_estimat.sum()     *100/ total_demand,        
        'Hydro %': sample['Hidráulica'].sum() * 100 / total_demand,
        'Nuclear %': sample['Nuclear'].sum() * 100 / total_demand,
        'Cogeneració':        sample['Cogeneració'].sum()*100/ total_demand,
        'Cicles + Import.':        sample['Gas+Imports'].sum()*100/ total_demand,        
        'Batteries %': sample['Bateries'].sum() * 100 / total_demand,
        'Fossil+Imports %': (sample['Gas+Imports'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Low-carbon %': (sample[['Eòlica', 'Solar', 'Hidráulica', 'Nuclear']].sum().sum()) * 100 / total_demand,
        'Renewables %': sample[['Eòlica', 'Solar', 'Hidráulica']].sum().sum() * 100 / total_demand,
        'Ren.-coverage': 100 - (sample['Gas+Imports']+sample['Nuclear']+sample['Cogeneració']).sum()*100/total_demand,
        'Ren.cov-B': round((1-sum(sample.gap0[sample.gap0 > 0]) / sum(sample.Demanda))*100,1),
        'Clean-coverage %': 100 - (sample['Gas+Imports'] + sample['Cogeneració']).sum() * 100 / total_demand,
        'Surpluses':   ((sample['Total'] + sample['Bateries'] - sample['Càrrega']).sum())*100/ total_demand
    }

    # n_years = 9
    # metrics = {
    #     'Wind':    sample['Eòlica'].sum()        /1000/n_years,
    #     'Solar':    sample['Solar'].sum()        /1000/n_years,
    #     'Autoconsum': autoconsum_estimat.sum()     /1000/n_years,
    #     'Hydro':         sample['Hidráulica'].sum()                      /1000/n_years,
    #     'Nuclear':       sample['Nuclear'].sum()                /1000/n_years    ,#*100/ total_demand,
    #     'Cogeneració':        sample['Cogeneració'].sum() /1000/n_years,
    #     'Cicles + Import.':        sample['Gas+Imports'].sum() /1000/n_years,
    #     'Bateries':       sample['Bateries'].sum()                    /1000/n_years,
    #     'Surpluses':   ((sample['Total'] + sample['Bateries'] - sample['Càrrega']).sum()) /1000/n_years,
    
    # }

    df_sintetic.rename(columns={'Hidráulica': 'Hidràulica'}, inplace=True)

    return df_sintetic, {k: round(v, 2) for k, v in metrics.items()}

# 2024
bat = [534, 5340, 0.8, 0]
w_wind = 0.9893
w_solar = 0.9369
w_auto = fv_autoconsum / potencia.Autoconsum.iloc[-1]

escenario = {
    'nucleares_activas': [True,True,True],
    'pesos': {'solar': w_solar, 'wind': w_wind, 'dem': 1, 'cog': 1, 'auto': w_auto},
    'baterias_config': bat,
    'max_salto_hidro_pct': 5, #10.0,
    'optimizador_hidro': 'rapido'
}

# Generar escenario sintético
results, energy_metrics = generar_escenario_sintetico(
    # Parámetros del escenario
    **escenario,
    
    # Datos base
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
    
    # Parámetros físicos
    potencia_max_hidro={'ebro': potencia_max_hidraulica_ebro, 'int': potencia_max_hidraulica_int},
    sensibilidad_hidro={'ebro': sensibility_ebro, 'int': sensibility_int},  # Corregido
    capacidad_max_hidro={'ebro': max_capacity_ebro, 'int': max_capacity_int},
    umbral_overflow_pct={'ebro': 75.0, 'int': 75.0}
)

# results.Hydro_Level_int.plot()
# old_level_int.plot()
# plt.show()

# results.Hydro_Level_ebro.plot()
# old_level_ebro.plot()
# plt.show()

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

# ------------------------------

# Cargamos datos de desalación histórica
dessalacio = pd.read_csv('C:/Users/tirki/Dropbox/Trabajos/Energía/dessalacio_20250723.csv')
# dessalacio = pd.read_excel('C:/Users/tirki/Dropbox/Trabajos/Energía/dessalinitzacio_20251027.xlsx')
dessalacio = dessalacio.groupby('Dia').sum()['Volum (hm3)']
dessalacio.index = pd.to_datetime(dessalacio.index, dayfirst=True)
dessalacio_diaria = dessalacio.resample('D').sum()['2015-06-03':'2024-12-31']
dessalacio_mensual = dessalacio.resample('ME').sum()

plt.figure(figsize=(10, 6))
dessalacio_diaria.rolling(7).median().plot()
# dessalacio_mensual.plot()
plt.show()
#%%
# =============================================================================
# 1. CARGAR Y PREPARAR EL PROXY
# =============================================================================
#https://analisi.transparenciacatalunya.cat/Medi-Ambient/Estacions-de-regeneraci-d-aigua-p-bliques-de-Catal/5bep-wiuf/data_preview
#https://analisi.transparenciacatalunya.cat/Medi-Ambient/Aigua-regenerada-del-Prat-de-Llobregat/3prd-nrah/data_preview <-- MIRAR ESTE!
# # Cargamos datos de regeneración histórica
# regeneracio = pd.read_csv('C:/Users/tirki/Dropbox/Trabajos/Energía/regeneracio_20250820.csv')
# regeneracio["Cabal canal (m³/s)"] = pd.to_numeric(regeneracio["Cabal canal (m³/s)"].str.replace(",", "."), errors="coerce")
# regeneracio = regeneracio.groupby('Dia').sum()['Cabal canal (m³/s)']
# regeneracio.index = pd.to_datetime(regeneracio.index, dayfirst=True)
# regeneracio.name = 'Volum'
# regeneracio_diaria = regeneracio.resample('D').sum()['2015-06-03':'2024-12-31']


# Datos del proxy
df_proxy = pd.read_csv("regeneracion_diaria.csv", parse_dates=["Dia"])
df_proxy = df_proxy.set_index("Dia").sort_index()
df_proxy.columns = ["Volum"]
# Calcular mediana móvil (usamos 60 días como compromiso)
df_proxy["proxy"] = df_proxy["Volum"].rolling(60, center=True, min_periods=1).median()
# Rellenar NaN en los extremos con el valor más cercano
df_proxy["proxy"] = df_proxy["proxy"].ffill().bfill()

# =============================================================================
# 2. DATOS DE ENTRADA Y CONFIGURACIÓN
# =============================================================================
# Volúmenes anuales objetivo (hm3/año)
years = np.arange(2016, 2025)
volumes = np.array([0.01, 0.03, 0.06, 8.1, 9.2, 32.9, 48.3, 54.6, 39.9])
serie_anual = pd.Series(volumes, index=years)
# Eje temporal diario completo
dates = pd.date_range("2016-01-01", "2024-12-31", freq="D")
n_days = len(dates)

# =============================================================================
# 3. CREAR PROXY NORMALIZADO PARA TODO EL PERIODO
# =============================================================================

# Crear serie del proxy alineada con todas las fechas
proxy_full = pd.Series(0.0, index=dates, name="proxy")

# Copiar valores del proxy donde existen
for date in df_proxy.index:
    if date in proxy_full.index:
        proxy_full.loc[date] = df_proxy.loc[date, "proxy"]

# Normalizar el proxy año a año para que sume el volumen objetivo
proxy_normalizado = np.zeros(n_days)

for year, total in serie_anual.items():
    idx = np.where(dates.year == year)[0]
    proxy_year = proxy_full.iloc[idx].values
    
    if year < 2019:
        # Antes de 2019: todo cero
        proxy_normalizado[idx] = 0
    elif proxy_year.sum() > 0:
        # Normalizar para que sume el total anual
        proxy_normalizado[idx] = proxy_year * (total / proxy_year.sum())
    else:
        # Si el proxy es cero, distribuir uniformemente
        proxy_normalizado[idx] = total / len(idx)

print("Verificación del proxy normalizado (sumas anuales):")
for year in years:
    idx = np.where(dates.year == year)[0]
    print(f"  {year}: {proxy_normalizado[idx].sum():.2f} (objetivo: {serie_anual[year]:.2f})")
    
# =============================================================================
# 4. OPTIMIZACIÓN CON PROXY COMO FORMA OBJETIVO
# =============================================================================

# Variable de decisión
x = cp.Variable(n_days, nonneg=True)

# Índices desde 2019
start_date = "2019-01-01"
mask_post = dates >= start_date
idx_post = np.where(mask_post)[0]

# Primera derivada (pendiente) desde 2019
dx = x[idx_post[1:]] - x[idx_post[:-1]]

# Segunda derivada (curvatura) desde 2019
d2x = x[idx_post[2:]] - 2 * x[idx_post[1:-1]] + x[idx_post[:-2]]

# Pesos (ajustables)
lambda_slope = 5000.0  # Penaliza cambios bruscos (muy alto)
lambda_curv = 100.0    # Penaliza curvatura (alto)
lambda_proxy = 100.0 #10 # Penaliza desviación del proxy (bajo)

# Función objetivo: suavidad + seguimiento del proxy
objective = cp.Minimize(
    lambda_slope * cp.sum_squares(dx)
    + lambda_curv * cp.sum_squares(d2x)
    + lambda_proxy * cp.sum_squares(x[idx_post] - proxy_normalizado[idx_post])
)

# Restricciones
constraints = []

# Pre-2019: todo a cero o casi cero
for year in years[years < 2019]:
    idx = np.where(dates.year == year)[0]
    constraints.append(cp.sum(x[idx]) <= 0.1)

# 2019 en adelante: volumen anual exacto
for year in years[years >= 2019]:
    idx = np.where(dates.year == year)[0]
    constraints.append(cp.sum(x[idx]) == serie_anual[year])

# Resolver
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CLARABEL, verbose=True)
print("\nEstado:", problem.status)
print(f"Valor objetivo: {problem.value:.6f}")

# =============================================================================
# 5. RESULTADO
# =============================================================================

regeneracio_diaria = pd.Series(x.value, index=dates, name="Regeneracio_hm3_dia")

# Verificar volúmenes anuales
print("\nVerificación de volúmenes anuales:")
for year in years:
    calc = regeneracio_diaria[regeneracio_diaria.index.year == year].sum()
    orig = serie_anual[year]
    print(f"  {year}: calculado={calc:.2f}, original={orig:.2f}")
    
# =============================================================================
# 6. GRAFICAR COMPARACIÓN
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(14, 6))  # Apaisado: ancho > alto

mask_zoom = dates >= "2019-01-01"
ax.plot(dates[mask_zoom], regeneracio_diaria.values[mask_zoom], 
        linewidth=1.5, label="Resultat optimitzat")
ax.plot(dates[mask_zoom], proxy_normalizado[mask_zoom], 
        linewidth=1, alpha=0.7, linestyle="--", label="Proxy normalitzat")
ax.set_ylabel("hm³/dia")
ax.set_title("Regeneració diària desde 2019 – optimizació guiada per proxy", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Encontrar el primer pico de producción
data = regeneracio_diaria[:'2020-01-01'].values
first_peak_idx = np.argmax(data)  # Índice del primer máximo
first_peak_date = dates[first_peak_idx]
first_peak_value = data[first_peak_idx]

# Añadir flecha y etiqueta
ax.annotate('Primera prova demostrativa',
            xy=(first_peak_date, first_peak_value),
            xytext=(first_peak_date, first_peak_value * 1.2),  # Ligeramente arriba del pico
            arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
            ha='center',
            fontsize=12,
            fontweight='bold')

plt.tight_layout()
plt.show()

#%%
# print("\nGráfico guardado en regeneracion_con_proxy.png")    


"""
“La ERA del Prat opera en régimen esencialmente continuo. La variabilidad temporal del caudal regenerado vertido al río responde a criterios de gestión del sistema (especialmente en episodios de sequía) y no a un patrón estacional intrínseco de la instalación.”

“La ERA opera en continuo y los cambios de volumen anual se materializan mediante ajustes progresivos de consigna, no como escalones instantáneos.”

La serie diaria se ha reconstruido mediante interpolación continua de los volúmenes anuales y posterior normalización por año, asumiendo una evolución progresiva de la capacidad operativa de la ERA y evitando discontinuidades artificiales.

Dado que los datos históricos no permiten caracterizar con precisión el periodo de puesta en marcha de la instalación, se ha permitido un volumen residual muy reducido en el año previo al inicio de la operación regular, con el fin de evitar discontinuidades artificiales en la reconstrucción diaria.
"""

# 1 Estima el ahorro debido a las restricciones históricas
consumo_base_diario_hm3 = 2.05
# ahorro, niveles = remove_restrictions(100*capacidad_internes.Nivel['2015-06-01':'2024-12-31'],
#                                       consumo_base_diario_hm3)
# ahorro, niveles = remove_restrictions(100*df_pct_int['2015-06-01':'2024-12-31'].squeeze(),
#                                       consumo_base_diario_hm3)
ahorro, niveles = remove_restrictions_seasonal(100*df_pct_int['2015-06-01':'2024-12-31'].squeeze(),
                                      consumo_base_diario_hm3)
# ahorro, niveles = remove_restrictions_seasonal(100*df_pct_int['2016-01-01':'2024-12-31'].squeeze(),
#                                       consumo_base_diario_hm3)

ahorro_pct = 100 * ahorro.resample('h').interpolate(method='linear') / max_capacity_int
ahorro_pct = ahorro_pct['2015-06-03':]
# (100*capacidad_internes.Nivel['2015-06-03':'2024-12-31']) - niveles
# (100*capacidad_internes.Nivel['2015-06-03':'2024-12-31'] - 100*ahorro/max_capacity_int).plot()
# niveles.plot()


# 2 Extrae la desalación histórica y calcula los niveles sin desalacion (sd)
nivel_sd = remove_real_desal(
    base_level=df_sintetic['Hydro_Level_int'],
    dessalacio_diaria=dessalacio_diaria,
    max_capacity_int=max_capacity_int
)

nivel_sd = remove_real_desal(
    base_level=nivel_sd,
    dessalacio_diaria=regeneracio_diaria,
    max_capacity_int=max_capacity_int
)

# 1.5 Nivel sin desalación histórica ni restricciones de consumo
# sin intervenciones (si)
nivel_si = nivel_sd - ahorro_pct
nivel_si = nivel_si.dropna()


dessalacio_diaria.plot()
regeneracio_diaria.plot()

#%%
niveles = nivel_si.resample('d').last()

# Configuració d'estil
sn.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

# Crear la figura
fig, ax = plt.subplots(figsize=(12, 6))

# 1. Plot de la sèrie Històrica
ax.plot(df_sintetic.index, df_sintetic['Hydro_Level_int'], 
        color='#1f77b4', linewidth=2, 
        label='Nivell Històric Registrat')

# 2. Plot de la sèrie sense Restriccions
ax.plot(niveles.index, niveles, 
        color='#d62728', linewidth=2, linestyle='--', 
        # label='Nivell Contrafactual (Sense Restriccions)')
        label='Nivell Contrafactual (Sense Intervencions)')

# 3. Llindars de Sequera (Línies i Etiquetes)
llindars = {
    'Prealerta': 60,
    'Alerta': 40,
    'Excepcionalitat': 25,
    'Emergència': 16
}

colors_sequera = ['#FFD700', '#FFA500', '#FF4500', '#8B0000'] # Groc, Taronja, Vermell, Vermell fosc

# Dibuixar línies de llindars
x_min = df_sintetic.index.min()
for (nom, valor), color in zip(llindars.items(), colors_sequera):
    ax.axhline(valor, color=color, linewidth=1, linestyle=':', alpha=0.8)
    ax.text(x_min, valor + 1, f' {nom} ({valor}%)', 
            color=color, verticalalignment='bottom', fontsize=9, fontweight='bold')

# Límit Físic (0%)
ax.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.8)
ax.text(x_min, 2, ' Límit Físic (0%)', verticalalignment='bottom', fontsize=10, color='black')

# 4. Zona de Dèficit Teòric
ax.fill_between(niveles.index, niveles, 0, 
                where=(niveles < 0), 
                color='red', alpha=0.1, 
                label='Dèficit Hídric Teòric')

# 5. Formatat
ax.set_ylabel('Nivell de Reserves (%)', fontsize=12)
ax.set_xlim(df_sintetic.index.min(), df_sintetic.index.max())
ax.set_ylim(bottom=min(niveles.min(), -10), top=100) # Ajustar límit inferior per veure el dèficit

ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)
# ax.set_title("Impacte de l'Estalvi per Restriccions en les Reserves (2016-2024)", fontsize=14, pad=15)
ax.set_title("Impacte de totes les Intervencions en les Reserves (2016-2024)", fontsize=14, pad=15)


plt.tight_layout()
plt.show()


#%%
import matplotlib.dates as mdates


# --- 1. PREPARACIÓ DE DADES (Simulació per a l'exemple) ---
# Substitueix 'dessalacio_diaria' per la teva sèrie real
# dates = pd.date_range(start='2020-01-01', end='2024-05-01', freq='D')
# valors_simulats = np.linspace(0.5, 2.5, len(dates)) + np.random.normal(0, 0.2, len(dates))
# dessalacio_diaria = pd.Series(valors_simulats, index=dates)

# --- 2. DEFINICIÓ DELS PERÍODES DE SEQUERA (EXEMPLE TER-LLOBREGAT) ---
# Has d'ajustar aquestes dates amb les dates oficials d'entrada en vigor
periodes_sequera = [
    # Sequera de 2016-2017 (Va fregar l'Alerta)
    ('2016-09-04', '2017-01-29', 'Prealerta', '#FFD700'),
    # Sequera de 2017-2018 (Va arribar a Alerta/Excepcionalitat)
    ('2017-09-17', '2018-04-01', 'Prealerta', '#FFD700'),    
    # Episodi breu de tardor 2019 (Glòria)
    ('2019-09-08', '2019-12-08', 'Prealerta', '#FFD700'),    
    # Inici del descens continuat
    ('2022-01-09', '2022-08-21', 'Prealerta', '#FFD700'),
    # Fase central d'Alerta (elimina el soroll de setembre 2022)
    ('2022-08-22', '2023-04-30', 'Alerta', '#FFA500'),
    # Fase crítica (elimina oscil·lacions maig 2023)
    ('2023-05-01', '2024-01-28', 'Excepcionalitat', '#FF4500'),
    # Pic de l'Emergència
    ('2024-01-29', '2024-05-05', 'Emergència', '#8B0000'),
    # Recuperació ràpida (pluges primavera 2024)
    ('2024-05-06', '2024-05-19', 'Excepcionalitat', '#FF4500'),
    ('2024-05-20', '2024-06-16', 'Alerta', '#FFA500'),
    # Estat actual i projecció
    ('2024-06-17', '2025-03-20', 'Prealerta', '#FFD700')
]

# --- 3. CREACIÓ DEL GRÀFIC ---
sn.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(12, 6))

# A) Dibuixar les franges verticals (fons)
# Ho fem abans de la línia per a que quedin darrere
labels_afegits = set() # Per evitar duplicar llegenda

for inici, fi, estat, color in periodes_sequera:
    # Convertir strings a datetime
    start_date = pd.to_datetime(inici)
    end_date = pd.to_datetime(fi)
    
    # Només afegim l'etiqueta a la llegenda una vegada per estat
    label = estat if estat not in labels_afegits else None
    labels_afegits.add(estat)
    
    ax.axvspan(start_date, end_date, color=color, alpha=0.2, label=label)
    
    # Opcional: Afegir text a la part superior de la franja
    mid_point = start_date + (end_date - start_date) / 2
    # ax.text(mid_point, ax.get_ylim()[1], estat[0], ha='center', va='bottom', fontsize=8, color=color)

# B) Dibuixar la sèrie de dessalinització
# Fem servir una mitjana mòbil per suavitzar el gràfic si les dades diàries tenen molt soroll
# Si vols la dada crua, canvia 'dessalacio_suavitzada' per 'dessalacio_diaria'
dessalacio_suavitzada = dessalacio_diaria.rolling(window=7, center=True).mean()

ax.plot(dessalacio_suavitzada.index, dessalacio_suavitzada, 
        color='#1f77b4', linewidth=1.5, 
        label='Producció Dessalinització (Mitjana 7 dies)')

# C) Formatat
ax.set_ylabel('Producció Diària (hm³/dia)', fontsize=12)
ax.set_xlabel('Any', fontsize=12)
ax.set_title('Evolució de la Dessalinització i Estats de Sequera', fontsize=14, pad=15)

# Format de dates a l'eix X
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12)) # Cada 6 mesos
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=0)

# Llegenda: Col·loquem la llegenda fora o a un lloc que no molesti
# Utilitzem 'handles' per assegurar l'ordre correcte si cal
ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10, title="Estat Sequera")

# Ajustar marges
plt.tight_layout()
plt.show()

# seasonal_capacity = seasonal_decompose(capacidad['Capacitat actual'].resample('ME').last()['2015-06-01':], model='additive', period=12).seasonal
# monthly_seasonal = seasonal_capacity.groupby(seasonal_capacity.index.month).mean()
# seasonal_amplitude = monthly_seasonal.max() - monthly_seasonal.min()
#%%

# 788 ms ± 12.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# # 1. Convierte potencia renovable a instalar en pesos
# solar = 12000
# wind = 5000
# bat = [1100, 1100*4, 0.8, 0]

# peso_solar = solar / (potencia.Fotovoltaica.iloc[-1] - potencia.Termosolar.iloc[-1])
# peso_wind = wind / potencia.Eòlica.iloc[-1]

# # Niveles actuales -----------
# peso_solar = 1
# peso_wind = 1
# bat = [500,500*4,0.8,0]

# # 2. Define un escenario específico que quieras probar
# escenario = {
#     'nucleares_activas': [True, True, True],
#     'pesos': {'solar': peso_solar, 'wind': peso_wind, 'dem': 1, 'cog': 1},
#     # 'pesos': {'solar': 10, 'wind': 4, 'dem': 1, 'cog': 0.5},
#     'baterias_config': bat,
#     'max_salto_hidro_pct': 5.0,
#     'optimizador_hidro': 'rapido'
# }

# # 2.2 Define los costes del escenario
# coste_Solar = (solar - potencia.Fotovoltaica.iloc[-1] - potencia.Termosolar.iloc[-1]) * 775000
# coste_Wind = (wind - potencia.Eòlica.iloc[-1]) * 1400000
# coste_Bat = (bat[1] - 2000) * 325000

# # 2.3 Define las variables del escenario de desalacion y gestión hidráulica
# min_run_hours = 4
# max_desalation = 30 #MW #30
# w_desal = max_desalation / 30 #30 MW es el valor actual instalat
# midpoint_estimation = find_midpoint_for_target_factor(75,0.5 / w_desal)
# # midpoint_estimation = 75
# # midpoint_estimation = find_midpoint_for_target_factor(80,0.27 / w_desal)
# overflow_threshold_pct = 95


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
    min_run_hours = 4,
    max_desalation = 30,
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
    level_final, desal_final, savings_final, extra_hydro_final, thermal_reduced_final = simulate_full_water_management(
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
    desal_final_hm3 = desal_final * 0.01 / 30
    
    # Calcular excedentes descontando desalación
    surpluses_desal = surpluses - desal_final
    
    # Calcular amplitud estacional
    seasonal_capacity = seasonal_decompose(level_final.resample('ME').last()['2015-06-01':], model='additive', period=12).seasonal
    monthly_seasonal = seasonal_capacity.groupby(seasonal_capacity.index.month).mean()
    seasonal_amplitude = monthly_seasonal.max() - monthly_seasonal.min()

    # Recalcular los niveles de los embalses y de Gas+Imports
    results['hydro_int'] += extra_hydro_final
    results['Hidráulica'] = results.hydro_ebro + results.hydro_int
    results['Gas+Imports'] -= thermal_reduced_final
    results['Hydro_Level_int'] = level_final
   
    
    # Métricas hídricas (corregido para usar df_niveles_int)
    hydro_metrics = {
        # 'Restricciones históricas (días)': df_niveles_int[(100*df_niveles_int.Nivel) < 40]['2015-06-01':'2024-12-31'].count().iloc[0]*7,
        'Restricciones históricas (días)': df_niveles_int[(100*df_niveles_int) < 40]['2015-06-01':'2024-12-31'].count()*7,
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
        'surpluses_afterdesal': surpluses_desal.sum(),
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
%%time

results = procesar_escenario(
    df_demanda = demanda,
    df_nuclear = nuclears_base,
    df_cogeneracion = cogeneracion_h,      
    df_solar = solar_h,
    df_eolica = eolica_h,
    df_autoconsum = autoconsum_hourly,
    df_potencia = potencia,
    df_niveles_int = df_pct_int_h.squeeze(), #capacidad_internes,
    df_niveles_ebro = df_pct_ebre_h.squeeze(), #capacidad_ebre,
    df_energia_turbinada_mensual_internes = energia_turbinada_mensual_internes,
    df_energia_turbinada_mensual_ebre = energia_turbinada_mensual_ebre,
    df_nivel_si = nivel_si.dropna(), #niveles sin intervención, ni desalación ni restricciones
    max_capacity_int = max_capacity_int,
    max_capacity_ebro = max_capacity_ebro,
    potencia_max_int = potencia_max_hidraulica_int,
    potencia_max_ebro = potencia_max_hidraulica_ebro,
    consumo_base_diario_hm3 = consumo_base_diario_hm3,
    sensibility_int = sensibility_int,
    sensibility_ebro = sensibility_ebro,
    nucleares_activas = [False,True,True],    
    potencia_cogeneracion = 470.2, #None,
    # variables de decisión (factores)
    potencia_solar = 7180.8, #6000, #12685.72,
    potencia_eolica = 6234.2, #9000, #8922.85,
    potencia_baterias = 2234, #1500, #866.69,
    min_run_hours = 4, #4,
    max_desalation = 120, #163.09,
    midpoint_estimation = 60, #50.11, #54 parámetro del sigmoide de desalación
    overflow_threshold_pct = 64, #68.78, #95
    seasonal_phase_months=2,
    seasonal_desal_amplitude=0.9
    )

# int(results['level_final'].min())
# int(results['level_final'].max())

# results['energy_data']['Gas+Imports'].sum()
# # df_results.iloc[21602].astype(int)
# df_results.iloc[13151].astype(int)
# df_results_factibles.iloc[19589].astype(int)
# print(selection.iloc[10].map('{:.1f}'.format))
# print(df_results_factibles.iloc[19589].map('{:.1f}'.format))
# print(df_tests.iloc[1].map('{:.1f}'.format))

#%%
# Script para comparar resultados paralelizados vs manuales
def test_discrepancies(df_parallel_results):
    """
    Ejecuta manualmente los mismos escenarios del DataFrame paralelizado
    y compara los resultados para detectar discrepancias.
    
    Args:
        df_parallel_results: DataFrame con los resultados del código paralelizado
    """
    
    # Lista para almacenar resultados manuales
    manual_results = []
    discrepancies = []
    
    print(f"Testando {len(df_parallel_results)} escenarios manualmente...")
    print("="*60)
    
    for idx, row in df_parallel_results.iterrows():
        print(f"Ejecutando escenario {idx+1}/{len(df_parallel_results)}...")
        
        try:
            # Ejecutar manualmente con los mismos parámetros
            result_manual = procesar_escenario(
                df_demanda=demanda,
                df_nuclear=nuclears_base,
                df_cogeneracion=cogeneracion_h,
                df_solar=solar_h,
                df_eolica=eolica_h,
                df_potencia=potencia,
                df_niveles_int=capacidad_internes,
                df_niveles_ebro=capacidad_ebre,
                df_energia_turbinada_mensual_internes=energia_turbinada_mensual_internes,
                df_energia_turbinada_mensual_ebre=energia_turbinada_mensual_ebre,
                df_nivel_si=nivel_si,
                max_capacity_int=max_capacity_int,
                max_capacity_ebro=max_capacity_ebro,
                potencia_max_int=potencia_max_hidraulica_int,
                potencia_max_ebro=potencia_max_hidraulica_ebro,
                consumo_base_diario_hm3=consumo_base_diario_hm3,
                sensibility_int=sensibility_int,
                sensibility_ebro=sensibility_ebro,
                nucleares_activas=[True, True, True],
                potencia_solar=row['potencia_solar'],
                potencia_eolica=row['potencia_eolica'],
                potencia_cogeneracion=None,
                potencia_baterias=row['potencia_baterias'],
                min_run_hours=row['min_run_hours'],
                max_desalation=row['max_desalation'],
                midpoint_estimation=row['midpoint_estimation'],
                overflow_threshold_pct=row['overflow_threshold_pct'],
                seasonal_phase_months=row['seasonal_phase_corrector'],
                seasonal_desal_amplitude=row['seasonal_amplitude_corrector']
            )
            
            # Extraer métricas clave
            min_level_manual = result_manual["level_final"].min()
            max_level_manual = result_manual["level_final"].max()
            gas_imports_manual = result_manual['energy_data']['Gas+Imports'].sum()
            
            # Almacenar resultado manual
            manual_results.append({
                'scenario_id': idx,
                'min_level_manual': min_level_manual,
                'max_level_manual': max_level_manual,
                'gas_imports_manual': gas_imports_manual
            })
            
            # Comparar con resultados paralelizados
            min_level_parallel = row['min_level']
            max_level_parallel = row['max_level']
            gas_imports_parallel = row['gas_imports']
            
            # Calcular diferencias (tolerancia para errores de precisión numérica)
            tolerance = 1e-6
            diff_min = abs(min_level_manual - min_level_parallel)
            diff_max = abs(max_level_manual - max_level_parallel)
            diff_gas = abs(gas_imports_manual - gas_imports_parallel)
            
            # Detectar discrepancias significativas
            has_discrepancy = (diff_min > tolerance or 
                             diff_max > tolerance or 
                             diff_gas > tolerance)
            
            if has_discrepancy:
                discrepancy_info = {
                    'scenario_id': idx,
                    'params': {
                        'potencia_solar': row['potencia_solar'],
                        'potencia_eolica': row['potencia_eolica'],
                        'potencia_baterias': row['potencia_baterias'],
                        'min_run_hours': row['min_run_hours'],
                        'max_desalation': row['max_desalation'],
                        'midpoint_estimation': row['midpoint_estimation'],
                        'overflow_threshold_pct': row['overflow_threshold_pct'],
                        'seasonal_phase_months': row['seasonal_phase_corrector'],
                        'seasonal_desal_amplitude': row['seasonal_amplitude_corrector']
                    },
                    'differences': {
                        'min_level': {'parallel': min_level_parallel, 'manual': min_level_manual, 'diff': diff_min},
                        'max_level': {'parallel': max_level_parallel, 'manual': max_level_manual, 'diff': diff_max},
                        'gas_imports': {'parallel': gas_imports_parallel, 'manual': gas_imports_manual, 'diff': diff_gas}
                    }
                }
                discrepancies.append(discrepancy_info)
                
                print(f"  ⚠️  DISCREPANCIA DETECTADA en escenario {idx}")
                print(f"      Min Level: {min_level_parallel:.6f} vs {min_level_manual:.6f} (diff: {diff_min:.6f})")
                print(f"      Max Level: {max_level_parallel:.6f} vs {max_level_manual:.6f} (diff: {diff_max:.6f})")
                print(f"      Gas+Imports: {gas_imports_parallel:.6f} vs {gas_imports_manual:.6f} (diff: {diff_gas:.6f})")
            else:
                print(f"  ✅ Escenario {idx} - Resultados coinciden")
                
        except Exception as e:
            print(f"  ❌ Error en escenario {idx}: {str(e)}")
            continue
    
    # Resumen de resultados
    print("\n" + "="*60)
    print("RESUMEN DE COMPARACIÓN:")
    print(f"Total escenarios testados: {len(df_parallel_results)}")
    print(f"Escenarios con discrepancias: {len(discrepancies)}")
    print(f"Escenarios coincidentes: {len(df_parallel_results) - len(discrepancies)}")
    
    if discrepancies:
        print("\n🔍 ANÁLISIS DE DISCREPANCIAS:")
        for disc in discrepancies:
            print(f"\nEscenario {disc['scenario_id']}:")
            print(f"  Parámetros: {disc['params']}")
            for metric, values in disc['differences'].items():
                print(f"  {metric}: Paralelo={values['parallel']:.6f}, Manual={values['manual']:.6f}, Diff={values['diff']:.6f}")
    else:
        print("✅ ¡Todos los escenarios coinciden perfectamente!")
    
    # Crear DataFrame de comparación
    df_comparison = pd.DataFrame(manual_results)
    df_comparison = df_comparison.merge(
        df_parallel_results[['min_level', 'max_level', 'gas_imports']].reset_index().rename(columns={'index': 'scenario_id'}),
        on='scenario_id',
        suffixes=('_manual', '_parallel')
    )
    
    return df_comparison, discrepancies

# Ejecutar el test (asume que df_results es tu DataFrame de resultados paralelizados)
# df_comparison, discrepancies_found = test_discrepancies(df_results)
# Ejecutar el test de discrepancias
df_comparison, discrepancies_found = test_discrepancies(df_tests)

# Ver la comparación completa
print("\nComparación detallada:")
print(df_comparison)

# Si hay discrepancias, examinar en detalle
if discrepancies_found:
    print("\nPrimer caso con discrepancia:")
    print(discrepancies_found[0])
#%%
(capacidad_internes.Nivel*100)['2015-06-01':'2024-12-31'].plot()
results['level_final'].plot()
plt.show()

print('Días con restricciones de consumo de agua')
print('-----------------------------------------')
print('Datos históricos:',capacidad_internes[(100*capacidad_internes.Nivel) < 40]['2015-06-01':'2024-12-31'].count().iloc[0]*7)
# print('Escenario + desalación histórica:',int(results['energy_data'][results['energy_data'].Hydro_Level_int < 40].count().iloc[0]/24))
print('Escenario + desalación de excedentes:',int(results['level_final'][results['level_final'] < 40].count()/24))
print('-----------------------------------------')
print('Dessalació histórica (hm3):',int(dessalacio_diaria.sum()))
print('Dessalació simulada (hm3):',int(results['desal_final_hm3'].sum()))
print('Factor de capacidad (%):', results['capacity_factor'])  #int(100 * results['desal_final'].mean() / max_desalation))
print('-----------------------------------------')
print('Restricciones históricas (hm3):', int(max_capacity_int * ahorro_pct.iloc[-1] / 100))
print('Restricciones en escenario (hm3):', int(results['savings_final'].sum()))

dessalacio_diaria.plot()
results['desal_final_hm3'].resample('D').sum().plot()
plt.show()

surpluses = results['energy_data'].gap
surpluses[surpluses > 0] = 0
surpluses *= -1

#%%

# sample = df_sintetic[['Demanda','Nuclear','Cogeneració','Eòlica','Solar','Hidràulica']]
sample = results['energy_data'][['Demanda','Nuclear','Cogen_w','Eòlica_w','Solar_w','Hidràulica', 'Bateries','Gas+Imports', 'Càrrega', 'gap0']]
sample.columns = ['Demanda','Nuclear','Cogeneració','Eòlica','Solar','Hidràulica', 'Bateries','Gas+Imports', 'Càrrega', 'gap0']

# sample = sample['2024-01-01':]

# Define coberturas en porcentaje
total_demand = sample['Demanda'].sum()
sample = sample.copy()
sample.loc[:,'Total'] = sample['Gas+Imports'] + sample['Cogeneració'] + sample['Nuclear'] + sample['Solar'] + sample['Eòlica'] + sample['Hidràulica']


metrics = {
    'Wind':    sample['Eòlica'].sum()        *100/ total_demand,
    'Solar':    sample['Solar'].sum()        *100/ total_demand,
    'Hydro':         sample['Hidràulica'].sum()                      *100/ total_demand,
    'Nuclear':       sample['Nuclear'].sum()                    *100/ total_demand,
    'Bateries':       sample['Bateries'].sum()                    *100/ total_demand,
    'Fossil+Imports':        (sample['Gas+Imports'] + sample['Cogeneració']).sum()*100/ total_demand,
    'Low-carbon':    (sample[['Eòlica','Solar','Hidràulica','Nuclear']].sum(axis=1).sum())*100/total_demand,
    'Renewables': sample[['Eòlica','Solar','Hidràulica']].sum().sum()*100/ total_demand,
    # Ren.-coverage A: las renovables se recortan (curtailment) para proteger la generación inflexible.
    'Ren.-coverage': 100 - (sample['Gas+Imports']+sample['Nuclear']+sample['Cogeneració']).sum()*100/total_demand,
    # Ren.-coverage B: nuclear y cogeneración se adaptan al mix renovable
    'Ren.cov-B': round((1-sum(sample.gap0[sample.gap0 > 0]) / sum(w_dem * sample.Demanda))*100,1),
    'Clean-coverage': 100 - (sample['Gas+Imports']+sample['Cogeneració']).sum()*100/total_demand,
    'Surpluses':   ((sample['Total'] + sample['Bateries'] - sample['Càrrega']).sum())*100/ total_demand

}

for name, val in metrics.items():
    print(f"{name:15s}: {val:.1f}%")

#%%
color_dict = {
    'Nuclear': '#9467bd',
    'Eòlica': '#2ca02c',
    'Solar': '#ff7f0e',
    'Cogeneració': '#8c564b',
    'Hidràulica': '#1f78b4',
    'Bateries': '#d62728',  # rojo intenso, consistente con el estilo de la paleta
    'Gas+Imports': '#7f7f7f'
}

dia = '2023-12-31'
# dia = '2023-04-01'
# dia = '2023-07-01'
# dia = '2020-04-01'
# dia = '2020-05-01'
inicio = pd.to_datetime(dia)
fin = inicio + pd.Timedelta(hours=23)
fin = inicio + pd.Timedelta(hours=24*7)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(12, 6))
# ax.set_ylim([0, 6000])

# Graficar el área acumulada del DataFrame
sample[inicio:fin].iloc[:,1:-3].plot.area(
    ax=ax,
    stacked=True,
    alpha=0.5,
    color=[color_dict.get(col, '#000000') for col in sample.columns[1:]]
)


# Graficar la línea de la Serie
sample[inicio:fin]['Càrrega'].plot(ax=ax, color='black', style='--', linewidth=2)
sample[inicio:fin]['Demanda'].plot(ax=ax, color='grey', linewidth=2, label='Demanda')


ax.legend(loc='upper left')
plt.show()

(df_sintetic.hydro_ebro.resample('ME').sum() - energia_turbinada_mensual_ebre)['2015-06-01':].tail(24)

surpluses = sample['Total'] + sample['Bateries'] - sample['Càrrega']


#%%
%%time
for i in range(5000,15000,5000):
    for j in range(3000,9000,2000):
        results = procesar_escenario(
            df_demanda = demanda,
            df_nuclear = nuclears_base,
            df_cogeneracion = cogeneracion_h,      
            df_solar = solar_h,
            df_eolica = eolica_h,
            df_potencia = potencia,
            df_niveles_int = capacidad_internes,
            df_niveles_ebro = capacidad_ebre,
            df_energia_turbinada_mensual_internes = energia_turbinada_mensual_internes,
            df_energia_turbinada_mensual_ebre = energia_turbinada_mensual_ebre,
            df_nivel_si = nivel_si, #niveles sin intervención, ni desalación ni restricciones
            max_capacity_int = max_capacity_int,
            max_capacity_ebro = max_capacity_ebro,
            potencia_max_int = potencia_max_hidraulica_int,
            potencia_max_ebro = potencia_max_hidraulica_ebro,
            consumo_base_diario_hm3 = consumo_base_diario_hm3,
            sensibility_int = sensibility_int,
            sensibility_ebro = sensibility_ebro,
            
            nucleares_activas = [True,True,True],    
            potencia_solar = i,
            potencia_eolica = j,
            potencia_cogeneracion = None,
            potencia_baterias = 1100,
            min_run_hours = 4,
            max_desalation = 30,
            midpoint_estimation = 75, # parámetro del sigmoide de desalación
            overflow_threshold_pct = 95 ,   # umbral a partir del cual se turbina más
            seasonal_phase_months= 6.0,
            seasonal_desal_amplitude=0.3
            )
        print(i,j,results['level_final'].min(), results['level_final'].max())
        
#%%
# Rutina paralelizada
%%time

# Generamos todas las combinaciones (i, j)
param_grid = [(i, j) for i in range(5000, 15000, 5000)
                        for j in range(3000, 13000, 2000)]

def run_case(i, j):
    results = procesar_escenario(
        df_demanda = demanda,
        df_nuclear = nuclears_base,
        df_cogeneracion = cogeneracion_h,      
        df_solar = solar_h,
        df_eolica = eolica_h,
        df_potencia = potencia,
        df_niveles_int = capacidad_internes,
        df_niveles_ebro = capacidad_ebre,
        df_energia_turbinada_mensual_internes = energia_turbinada_mensual_internes,
        df_energia_turbinada_mensual_ebre = energia_turbinada_mensual_ebre,
        df_nivel_si = nivel_si,
        max_capacity_int = max_capacity_int,
        max_capacity_ebro = max_capacity_ebro,
        potencia_max_int = potencia_max_hidraulica_int,
        potencia_max_ebro = potencia_max_hidraulica_ebro,
        consumo_base_diario_hm3 = consumo_base_diario_hm3,
        sensibility_int = sensibility_int,
        sensibility_ebro = sensibility_ebro,
        nucleares_activas = [True, True, True],    
        potencia_solar = i,
        potencia_eolica = j,
        potencia_cogeneracion = None,
        potencia_baterias = 1100,
        min_run_hours = 4,
        max_desalation = 30,
        midpoint_estimation = 75,
        overflow_threshold_pct = 95
    )
    return (i, j, results['level_final'].min(), results['level_final'].max())

# Ejecutar en paralelo (usa todos los cores con n_jobs=-1)
results = Parallel(n_jobs=-1)(delayed(run_case)(i, j) for i, j in param_grid)

# Mostrar resultados
for i, j, mn, mx in results:
    print(i, j, mn, mx)

# 10.000

#%%
%%time
import itertools
from joblib import Parallel, delayed

# === 1. Define tu espacio de búsqueda ===
# (ejemplo con 7 variables, sustituye por tus rangos reales)
potencia_solar = np.arange(1000,21000,1000)
potencia_eolica = np.arange(2000,11000,1000)
potencia_baterias = np.arange(500,3500,500)
min_run_hours = np.arange(3, 8, 1)
max_desalation = np.arange(30, 130, 10)
midpoint_estimation = np.arange(50,91,1)
overflow_threshold_pct = np.arange(60,99,1)
seasonal_phase_months = np.arange(0, 12, 1)
seasonal_amplitude = np.arange(0,1.1,0.1)

potencia_solar = np.arange(1000,20000,100)
potencia_eolica = np.arange(2000,10000,100)
potencia_baterias = np.arange(500,5000,100)
min_run_hours = np.arange(3, 8, 1)
max_desalation = np.arange(60, 160, 1)
midpoint_estimation = np.arange(10,91,1)
overflow_threshold_pct = np.arange(40,99,1)
seasonal_phase_months = np.arange(0, 11.9, 0.1)
seasonal_amplitude = np.arange(0,1.1,0.1)

# combinaciones -> si son muchas puedes muestrear 10k al azar en lugar de usar itertools.product
# param_grid = list(itertools.product(potencia_solar, potencia_eolica, potencia_baterias,min_run_hours, max_desalation, midpoint_estimation, overflow_threshold_pct))

# Si tienes más combinaciones que 10.000:
# import random; param_grid = random.sample(param_grid, 10000)

# === 2. Genera escenarios aleatorios ===
n_samples = 50000  # número de escenarios deseado
rng = np.random.default_rng(42)  # semilla reproducible

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
        rng.choice(seasonal_amplitude)
    )
    for _ in range(n_samples)
]

# === 3. Define la función de evaluación ===
def run_case(p_solar, p_eolica, p_baterias, run_h, desal, midpoint, overflow, phase, amplitude):
    results = procesar_escenario(
        df_demanda=demanda,
        df_nuclear=nuclears_base,
        df_cogeneracion=cogeneracion_h,
        df_solar=solar_h,
        df_eolica=eolica_h,
        df_potencia=potencia,
        df_niveles_int=capacidad_internes,
        df_niveles_ebro=capacidad_ebre,
        df_energia_turbinada_mensual_internes=energia_turbinada_mensual_internes,
        df_energia_turbinada_mensual_ebre=energia_turbinada_mensual_ebre,
        df_nivel_si=nivel_si,
        max_capacity_int=max_capacity_int,
        max_capacity_ebro=max_capacity_ebro,
        potencia_max_int=potencia_max_hidraulica_int,
        potencia_max_ebro=potencia_max_hidraulica_ebro,
        consumo_base_diario_hm3=consumo_base_diario_hm3,
        sensibility_int=sensibility_int,
        sensibility_ebro=sensibility_ebro,
        nucleares_activas=[True, True, True],
        potencia_solar=p_solar,
        potencia_eolica=p_eolica,
        potencia_cogeneracion=None,
        potencia_baterias=p_baterias,
        min_run_hours=run_h,
        max_desalation=desal,
        midpoint_estimation=midpoint,
        overflow_threshold_pct=overflow,
        seasonal_phase_months= phase,
        seasonal_desal_amplitude = amplitude
    )

    return {
        # === Variables de entrada ===
        "potencia_solar": p_solar,
        "potencia_eolica": p_eolica,
        "potencia_baterias": p_baterias,
        "min_run_hours": run_h,
        "max_desalation": desal,
        "midpoint_estimation": midpoint,
        "overflow_threshold_pct": overflow,
        "seasonal_phase_corrector": phase,
        "seasonal_amplitude_corrector": amplitude,

        # === Métricas de salida (ejemplo, añade más) ===
        "min_level": results["level_final"].min(),
        "max_level": results["level_final"].max(),
        "mean_level": results['level_final'].mean(),
        "desal_hm3": results['desal_final_hm3'].sum(),
        "restriction_days": results['hydro_metrics']['Restricciones escenario (días)'],
        "restriction_savings": results['savings_final'].sum(),
        "desal_cf": results['capacity_factor'],
        "surpluses": results['surpluses_afterdesal'],
        "gas_imports": results['energy_data']['Gas+Imports'].sum(),
        "instal_costs": results['costes']['total'],
        "seasonal_amplitude": results['hydro_metrics']['Variación estacional (%)']
    }

# === 4. Ejecuta en paralelo ===
if __name__ == "__main__":
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_case)(*params) for params in scenarios_params
    )


    # === 5. Guardar en DataFrame y exportar ===
    # df_tests = pd.DataFrame(results)
    df_results2 = pd.DataFrame(results)
    df_results2.to_csv("gridsearch_results2.csv", index=False)   # fácil de cargar después
    df_results2.to_parquet("gridsearch_results2.parquet")        # más compacto y rápido

    # print("Resultados guardados en gridsearch_results.csv/parquet")


df_results3 = pd.read_parquet("gridsearch_results2.parquet")

# df_results[df_results['max_level'] <= 100].min_level.max()
df_results_factibles2 = df_results2[df_results2['max_level'] <= 100]
# df_results[df_results['max_level'] <= 100].iloc[df_results[df_results['max_level'] <= 100].min_level.argmax()].astype(int)

# df_results_factibles[df_results_factibles['min_level'] > 30]

# df_tests.min_level

# 100 -> 36s
# 500 -> 3m
# 1000 -> 360s = 6m
# 10000 -> 3600s = 1h
# 50000 -> 5h | 6h 40m
# 100000 -> 10h
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
                label=f'Todas las soluciones factibles ({len(df)})', s=50)
    
    # Soluciones de Pareto
    plt.scatter(pareto_df[obj1], pareto_df[obj2], color='red', 
                label=f'Frente de Pareto ({len(pareto_df)})', s=100, edgecolor='black')
    
    plt.xlabel(f'{obj1} {"(minimizar)" if dir1=="min" else "(maximizar)"}')
    plt.ylabel(f'{obj2} {"(minimizar)" if dir2=="min" else "(maximizar)"}')
    plt.title('Frente de Pareto - Soluciones Factibles No Dominadas')
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
        # "costo_vs_seguridad": {
        #     "objectives": ["instal_costs", "gas_imports", "min_level", "surpluses", "mean_level", "restriction_savings"],
        #     "directions": ["min", "min", "max", "min", "max","min"],
        #     "description": "Minimizar costes y emisiones y maximizar seguridad hídrica"
        # },
        "seguridad_excedentes_emisiones": {
            "objectives": ["min_level", "gas_imports", "surpluses"],
            "directions": ["max", "min", "min"],
            "description": "Minimizar excedentes y emisiones y maximizar seguridad hídrica"
        },
        
        
        # Configuración 2: Minimizar días de restricción y costes
        "seguridad_vs_desalacion": {
            "objectives": ["min_level", "max_desalation"],
            "directions": ["max", "min"],
            "description": "Minimizar potencia de desalación y maximizar seguridad hídrica"
        },
        
        # # Configuración 3: Tri-objetivo básico
        # "tri_objetivo_basico": {
        #     "objectives": ["instal_costs", "restriction_days", "gas_imports"],
        #     "directions": ["min", "min", "min"],
        #     "description": "Minimizar costes, restricciones e importaciones"
        # },
        
        # # Configuración 4: Maximizar desalación vs minimizar costes
        # "desal_vs_costo": {
        #     "objectives": ["desal_cf", "instal_costs"],
        #     "directions": ["max", "min"],
        #     "description": "Maximizar factor de capacidad desalación vs minimizar costes"
        # }
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

# ===================================================================
# EJEMPLO DE USO
# ===================================================================

# Extraer múltiples frontales de Pareto
pareto_results = extract_pareto_energy_scenarios(df_results_factibles2)

# Acceder a una configuración específica
# pareto_costo_seguridad = pareto_results["costo_vs_seguridad"]["pareto_df"]
pareto_triobjetivo = pareto_results["seguridad_excedentes_emisiones"]["pareto_df"]
pareto_config2 = pareto_results['seguridad_vs_desalacion']['pareto_df']

# Visualizar (para 2 objetivos)
plot_pareto_front_2d(
    df_results_factibles, 
    pareto_results["costo_vs_seguridad"]["pareto_df"],
    "instal_costs", "gas_imports", 
    "min", "min"
)

# Visualizar (para 2 objetivos)
plot_pareto_front_2d(
    df_results, 
    pareto_results["costo_vs_seguridad"]["pareto_df"],
    "instal_costs", "min_level", 
    "min", "max"
)

# # Uso:
# import matplotlib
# matplotlib.use('Qt5Agg') 
    
plot_pareto_3d_plotly(
    df_results_factibles, 
    pareto_results["costo_vs_seguridad"]["pareto_df"],
    "instal_costs", "gas_imports", "restriction_savings",
    "min", "min", "max"
)

# Visualizar (para 2 objetivos)
plot_pareto_front_2d(
    df_results_factibles2, 
    pareto_results["seguridad_excedentes_emisiones"]["pareto_df"],
    "gas_imports", "min_level",
    "min", "max"
)

plot_pareto_front_2d(
    df_results_factibles2, 
    pareto_results["seguridad_excedentes_emisiones"]["pareto_df"],
    "surpluses", "min_level",
    "min", "max"
)

plot_pareto_front_2d(
    df_results_factibles2, 
    pareto_results["seguridad_vs_desalacion"]["pareto_df"],
    "max_desalation", "min_level",
    "min", "max"
)



plot_pareto_3d_plotly(
    df_results_factibles2, 
    pareto_results["seguridad_excedentes_emisiones"]["pareto_df"],
    "min_level", "surpluses", "gas_imports",
    "max", "min", "min"
)
# matplotlib.use('inline')  # O la que tengas por defecto


# Ver las mejores soluciones
print("Top 5 soluciones del frente de Pareto (costo vs seguridad):")
print(pareto_costo_seguridad[["potencia_solar", "potencia_eolica", "instal_costs", "min_level", "gas_imports", "surpluses", "desal_cf"]].head())

selection = pareto_triobjetivo[pareto_triobjetivo.min_level > 35]
selection = selection[selection.surpluses < 1.5e8]
selection[selection.gas_imports < 3e6]
#%%

sample = results['energy_data'][['Demanda','Nuclear','Cogen_w','Eòlica_w','Solar_w','Hidràulica', 'Bateries','Gas+Imports', 'Càrrega', 'gap0']]
sample = pd.concat((sample,results['desal_final']), axis=1)
sample.columns = ['Demanda','Nuclear','Cogeneració','Eòlica','Solar','Hidràulica', 'Bateries','Gas+Imports', 'Càrrega', 'gap0', 'Dessalació']


color_dict = {
    'Nuclear': '#9467bd',
    'Eòlica': '#2ca02c',
    'Solar': '#ff7f0e',
    'Cogeneració': '#8c564b',
    'Hidràulica': '#1f78b4',
    'Bateries': '#d62728',  # rojo intenso, consistente con el estilo de la paleta
    'Gas+Imports': '#7f7f7f'
}

dia = '2023-12-31'
# dia = '2023-04-01'
# dia = '2023-07-01'
# dia = '2020-04-01'
# dia = '2020-05-01'
inicio = pd.to_datetime(dia)
fin = inicio + pd.Timedelta(hours=23)
fin = inicio + pd.Timedelta(hours=24*7)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(12, 6))
# ax.set_ylim([0, 6000])

# Graficar el área acumulada del DataFrame
sample[inicio:fin].iloc[:,1:-3].plot.area(
    ax=ax,
    stacked=True,
    alpha=0.5,
    color=[color_dict.get(col, '#000000') for col in sample.columns[1:]]
)


# Graficar la línea de la Serie
sample[inicio:fin]['Càrrega'].plot(ax=ax, color='black', style='--', linewidth=2)
sample[inicio:fin]['Demanda'].plot(ax=ax, color='grey', linewidth=2, label='Demanda')
# (sample['Dessalació']+sample['Càrrega'])[inicio:fin].plot(ax=ax, color='grey', linewidth=2, style='--', label='Dessalació')

ax.legend(loc='upper left')
plt.show()


#%%

# 1. Creamos una serie booleana
mask = surpluses > 0

# 2. Detectamos los “bloques” de cambio
bloques = (mask != mask.shift(1)).cumsum()

# 3. Calculamos la duración de cada bloque de excedentes
duraciones = (
    mask
    .groupby(bloques)
    .apply(lambda x: len(x) if x.iloc[0] else 0)
)
duraciones = duraciones[duraciones > 0]

# 4. Media y mediana
media = duraciones.mean()
mediana = duraciones.median()
q1 = duraciones.quantile(0.25)
q3 = duraciones.quantile(0.75)
q90 = duraciones.quantile(0.90)
q95 = duraciones.quantile(0.95)
q99 = duraciones.quantile(0.99)
iqr = q3 - q1
maximo = duraciones.max()
# prob_sup = (duraciones > q3).mean()

print(f"Media de horas consecutivas con excedentes: {media:.2f}")
print(f"Mediana de horas consecutivas con excedentes: {mediana:.0f}")
print(f"Máximo de horas consecutivas con excedentes: {maximo:.0f}")
print(f"Q1: {q1:.0f}, Q3: {q3:.0f}, IQR: {iqr:.0f}")
print(f"Percentil 90%: {q90:.0f}")
print(f"Percentil 95%: {q90:.0f}")
print(f"Percentil 99%: {q90:.0f}")



int(df_resultado_A[df_resultado_A.Hydro_Level_int < 25].count().iloc[0]/24)
int(df_resultado_A[df_resultado_A.Hydro_Level_int < 16].count().iloc[0]/24)


#%%

# df_resultado_A.Hydro_Level_int.plot()
# level_syn.plot()
# sr.plot()
# sr2.plot()
# plt.show()

level_syn2, desal_syn2, ahorro_syn = simulate_full_water_management(
    surpluses=surpluses,
    level_base=nivel_si,
    max_capacity_int=max_capacity_int,
    consumo_base_diario_hm3=2
)

desal_syn2_hm3 = desal_syn2 * 0.01 / 30

print('Días con restricciones de consumo de agua')
print('-----------------------------------------')
print('Datos históricos:',capacidad_internes[(100*capacidad_internes.Nivel) < 40]['2015-06-01':'2024-12-31'].count().iloc[0]*7)
print('Escenario + desalación histórica:',int(df_resultado_A[df_resultado_A.Hydro_Level_int < 40].count().iloc[0]/24))
print('Escenario + desalación de excedentes:',int(level_syn2[level_syn2 < 40].count()/24))
print('-----------------------------------------')
print('Dessalació histórica (hm3):',int(dessalacio_diaria.sum()))
print('Dessalació simulada (hm3):',int(desal_syn2_hm3.sum()))
print('Factor de capacidad (%):', int(100 * desal_syn2.mean() / max_desalation))
print('-----------------------------------------')
print('Restricciones históricas (hm3):', int(max_capacity_int * ahorro_pct.iloc[-1] / 100))
print('Restricciones en escenario (hm3):', int(ahorro_syn.sum()))

# df_resultado_A.Hydro_Level_int.plot()
# level_syn2.plot()
# plt.show()

# desal_syn2_hm3.resample('D').sum().plot()
# dessalacio_diaria.plot()
# plt.show()


# # df_resultado_A.Hydro_Level_int['2020-01-01':'2020-06-01'].resample('ME').max()
# # level_syn2['2020-01-01':'2020-06-01'].resample('ME').max()
# # results_dict['level']['2020-01-01':'2020-06-01'].resample('ME').max()
# df_resultado_A['hydro_int'].resample('ME').sum()['2020-01-01':'2020-06-01']

# df_resultado_A['Gas+Imports']['2020-01-01':'2020-06-01'].plot()
# df_resultado_A['hydro_int']['2020-01-01':'2020-06-01'].plot()


#%%

# Acceder a resultados
print(f"Status: {result['status']}")
print(f"Nivel máximo final: {result['statistics']['final_max_level_pct']:.2f}%")
print(f"Energía redistribuida: {result['statistics']['total_hydro_extra_mwh']:.1f} MWh")

# Series finales
niveles_finales = result['levels']
generacion_termica_final = result['thermal_generation'] 
generacion_hidro_final = result['hydro_generation']

df_resultado_A.Hydro_Level_int.plot(label='original')
levelsyn75.plot(label='desalation 75')
niveles_finales.plot(label='extrahydro 95')
plt.legend()
plt.show()

#%%

level_final, desal_final, savings_final, extra_hydro_final, thermal_reduced_final = simulate_full_water_management(
    surpluses=surpluses,
    level_base=nivel_si,
    thermal_generation=df_resultado_A['Gas+Imports'],
    base_hydro_generation=df_resultado_A['hydro_int'],
    max_capacity_int=max_capacity_int,
    consumo_base_diario_hm3=2,
    # ... otros parámetros existentes ...
    max_hydro_capacity_mw=potencia_max_int,
    overflow_threshold_pct=95.0,  # Activar al 95%
    sensitivity_mwh_per_percent= max_capacity_int * 0.01 * sensibility_int  # Tu factor de conversión
)


desal_final_hm3 = desal_final * 0.01 / 30

print('Días con restricciones de consumo de agua')
print('-----------------------------------------')
print('Datos históricos:',capacidad_internes[(100*capacidad_internes.Nivel) < 40]['2015-06-01':'2024-12-31'].count().iloc[0]*7)
print('Escenario + desalación histórica:',int(df_resultado_A[df_resultado_A.Hydro_Level_int < 40].count().iloc[0]/24))
print('Escenario + desalación de excedentes:',int(level_final[level_final < 40].count()/24))
print('-----------------------------------------')
print('Dessalació histórica (hm3):',int(dessalacio_diaria.sum()))
print('Dessalació simulada (hm3):',int(desal_final_hm3.sum()))
print('Factor de capacidad (%):', int(100 * desal_final.mean() / max_desalation))
print('-----------------------------------------')
print('Restricciones históricas (hm3):', int(max_capacity_int * ahorro_pct.iloc[-1] / 100))
print('Restricciones en escenario (hm3):', int(savings_final.sum()))

# Analizar resultados
results = analyze_water_management_results(
    level_final, desal_final, savings_final, 
    extra_hydro_final, thermal_reduced_final
)
print(f"Status: {results['feasibility']['status']}")
print(f"Nivel máximo: {results['summary']['final_max_level_pct']:.2f}%")



desal_final_hm3.resample('D').sum().plot()
dessalacio_diaria.plot()

# df_resultado_A.Hydro_Level_int.plot(label='original')
# levelsyn75.plot(label='desalation 75')
# niveles_finales.plot(label='extrahydro 95')
# level_final.plot(label='allinone')
# plt.legend()
# plt.show()

#%%


# Construcción de df_sintetic
df_sintetic = pd.concat((demanda['2013-01-01':'2025-01-01'],nuclears['2013-01-01':'2025-01-01']),axis=1)
df_sintetic.columns = ['Demanda','Nuclear']

# bombeig = 439.84 + 90
# sensibility = 400

w_solar = 1 #10  #10 #5
w_wind = 1 #4 #2
w_dem = 1
w_cog = 1 #0.5

solar_h_w = solar_h * ((potencia.Fotovoltaica.iloc[-1]+potencia.Termosolar.iloc[-1])/(potencia.Fotovoltaica+potencia.Termosolar)).resample('h').ffill()
df_sintetic['Solar_w'] = solar_h_w.dropna()
eolica_h_w = eolica_h * (potencia.Eòlica.iloc[-1]/potencia.Eòlica).resample('h').ffill()
df_sintetic['Eòlica_w'] = eolica_h_w.dropna()
cogeneracion_h_w = cogeneracion_h * (potencia.Cogeneració.iloc[-1]/potencia.Cogeneració).resample('h').ffill()
df_sintetic['Cogen_w'] = cogeneracion_h_w.dropna()

df_sintetic.loc[:,'gap'] = w_dem * df_sintetic['Demanda'] - df_sintetic['Nuclear'] - w_solar * df_sintetic['Solar_w'] - w_wind * df_sintetic['Eòlica_w'] - w_cog * df_sintetic['Cogen_w']

df_sintetic.loc[:,'Solar_w'] = w_solar * df_sintetic['Solar_w']
df_sintetic.loc[:,'Eòlica_w'] = w_wind * df_sintetic['Eòlica_w']
df_sintetic.loc[:,'Cogen_w'] = w_cog * df_sintetic['Cogen_w']

df_sintetic = df_sintetic.dropna()

# Resampleas la serie semanal a horaria, rellenando hacia adelante
hydro_hourly = capacidad['Nivel'].resample('h').ffill()

# Reindexar el nivel embalsado sobre el índice horario de 'df_sintetic'
df_sintetic.loc[:,'Hydro_Level'] = 100 * hydro_hourly.reindex(
    df_sintetic.index, 
    method='ffill'
)


# 1. Parámetros de capacidad relativa
k_int = 0.1  # las internas tienen el 10% de la capacidad de Ebro
k_ebro = 1.0  # normalizamos Ebro a 1

potencia_max_hidraulica_ebro = 1713 #1557.2
potencia_max_hidraulica_int = 207 #272

# Umbrales de sobrecapacidad por encima de los cuales está prohibido acumular reserva
level_overflow_pct = 75.0
level_overflow_pct_ebro = level_overflow_pct
level_overflow_pct_int = level_overflow_pct 

sensibility = 400
sensibility_int = sensibility 
sensibility_ebro = sensibility

# 2. Series de entrada:
#    - energia_turbinada_mensual: pandas Series con índice mensual
#    - capacidad_internes.Nivel: pandas Series con índice semanal
#    - capacidad_ebre.Nivel: pandas Series con índice semanal

# 3. Resamplear las series semanales de nivel a mensual (último valor disponible de cada mes)
nivel_int_m = capacidad_internes.Nivel.resample('ME').last()
nivel_ebro_m = capacidad_ebre.Nivel.resample('ME').last()

# 4. Construir DataFrame conjunto con frecuencia mensual
df = pd.DataFrame({
    'Energia': energia_turbinada_mensual,
    'Nivel_int': nivel_int_m,
    'Nivel_ebro': nivel_ebro_m
}).dropna()


# 5. Calcular pesos según capacidad relativa * nivel de embalse
#    Esto refleja potencia instalada y nivel de llenado
# ndf = df.copy()
df['w_int']  = k_int  * df['Nivel_int']
df['w_ebro'] = k_ebro * df['Nivel_ebro']

# 6. Fracciones de distribución de la generación
df['f_int']  = df['w_int']  / (df['w_int'] + df['w_ebro'])
df['f_ebro'] = 1 - df['f_int']

# 7. Desagregar la energía turbinada mensual
df['Energia_int']  = df['Energia'] * df['f_int']
df['Energia_ebro'] = df['Energia'] * df['f_ebro']

# 8. Extraer las series resultantes
energia_turbinada_mensual_internes = df['Energia_int']
energia_turbinada_mensual_ebre      = df['Energia_ebro']

# Resampleas la serie semanal a horaria, rellenando hacia adelante
hydro_hourly_int = capacidad_internes['Nivel'].resample('h').ffill()
hydro_hourly_ebre = capacidad_ebre['Nivel'].resample('h').ffill()

# Reindexar el nivel embalsado sobre el índice horario de 'df_sintetic'
df_sintetic.loc[:,'Hydro_Level_int'] = 100 * hydro_hourly_int.reindex(
    df_sintetic.index, 
    method='ffill'
)
df_sintetic.loc[:,'Hydro_Level_ebro'] = 100 * hydro_hourly_ebre.reindex(
    df_sintetic.index, 
    method='ffill'
)

# 103 ms ± 2.05 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

#%%


# --- 2. Llamada a la función ---
df_results = calcular_generacion_hidraulica(
    df_sintetic=df_sintetic,
    energia_turbinada_mensual_internes=energia_turbinada_mensual_internes,
    energia_turbinada_mensual_ebre=energia_turbinada_mensual_ebre,
    potencia_max_int=potencia_max_hidraulica_int,
    potencia_max_ebro=potencia_max_hidraulica_ebro,
    sensibility_int=sensibility_int,
    sensibility_ebro=sensibility_ebro,
    max_capacity_int= max_capacity_int,
    max_capacity_ebro= max_capacity_int,
    level_overflow_pct_int=level_overflow_pct_int,
    level_overflow_pct_ebro=level_overflow_pct_ebro,
    max_salto_pct_mensual=5,
    puntos_optimizacion=0
)


# 276 ms ± 1.43 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
#%%
# # Resampleas la serie semanal a horaria, rellenando hacia adelante
# hydro_hourly_int = capacidad_internes['Nivel'].resample('h').ffill()
# hydro_hourly_ebre = capacidad_ebre['Nivel'].resample('h').ffill()

# # Reindexar el nivel embalsado sobre el índice horario de 'df_sintetic'
# df_sintetic['Hydro_Level_int'] = 100 * hydro_hourly_int.reindex(
#     df_sintetic.index, 
#     method='ffill'
# )
# df_sintetic['Hydro_Level_ebro'] = 100 * hydro_hourly_ebre.reindex(
#     df_sintetic.index, 
#     method='ffill'
# )

# df_sintetic.loc[:,'gap'] = w_dem * df_sintetic['Demanda'] - df_sintetic['Nuclear'] - df_sintetic['Solar_w'] - df_sintetic['Eòlica_w'] - df_sintetic['Cogen_w']


# potencia_max_hidraulica_ebro = 1713
# potencia_max_hidraulica_int = 207

# sensibility = 400
# sensibility_int = sensibility 
# sensibility_ebro = sensibility

# #umbrales de level overflow (%)
# level_overflow_pct_ebro = 75.0 
# level_overflow_pct_int = 75.0 

# # --- 2. Llamada a la función ---
# df_sintetic = calcular_generacion_hidraulica(
#     df_sintetic=df_sintetic,
#     energia_turbinada_mensual_internes=energia_turbinada_mensual_internes,
#     energia_turbinada_mensual_ebre=energia_turbinada_mensual_ebre,
#     potencia_max_int=potencia_max_hidraulica_int,
#     potencia_max_ebro=potencia_max_hidraulica_ebro,
#     sensibility_int=sensibility_int,
#     sensibility_ebro=sensibility_ebro,
#     max_capacity_int= max_capacity_int,
#     max_capacity_ebro= max_capacity_int,
#     level_overflow_pct_int=level_overflow_pct_int,
#     level_overflow_pct_ebro=level_overflow_pct_ebro,
#     max_salto_pct_mensual=5,
#     puntos_optimizacion=0
# )

#%%

# 1. Convierte potencia renovable a instalar en pesos
solar = 12000
wind = 5000
bat = [1100, 1100*4, 0.8, 0]

peso_solar = solar / (potencia.Fotovoltaica.iloc[-1] - potencia.Termosolar.iloc[-1])
peso_wind = wind / potencia.Eòlica.iloc[-1]

# peso_solar = 1
# peso_wind = 1
# bat = [500,500*4,0.8,0]

# 2. Define un escenario específico que quieras probar
escenario = {
    'nucleares_activas': [False, True, True],
    'pesos': {'solar': peso_solar, 'wind': peso_wind, 'dem': 1, 'cog': 1},
    # 'pesos': {'solar': 10, 'wind': 4, 'dem': 1, 'cog': 0.5},
    'baterias_config': bat,
    'max_salto_hidro_pct': 5.0,
    'optimizador_hidro': 'rapido'
}

# 2.2 Parámetros de desalación
potencia_desalacion = 30 #30 MW es el valor actual instalado (VALOR CONOCIDO)
min_run_hours = 4
max_desalation = 60 #MW #30
w_desal = max_desalation / potencia_desalacion 
midpoint_estimation = find_midpoint_for_target_factor(75,0.5 / w_desal)
# midpoint_estimation = 75
# midpoint_estimation = find_midpoint_for_target_factor(80,0.27 / w_desal)

# def procesar_escenario(
        
#         )
#     # 2.3 Define los costes del escenario
#     depreciation = [1,1,1,1] #solar, wind, batteries, desalation
    
    
#     coste_Solar = depreciation[0] * (solar - potencia.Fotovoltaica.iloc[-1] - potencia.Termosolar.iloc[-1]) * 775000
#     coste_Wind = depreciation[1]* (wind - potencia.Eòlica.iloc[-1]) * 1400000
#     coste_Bat = depreciation[2] * (bat[1] - 2000) * 325000
#     coste_Desal = depreciation[3] * (max_desalation - potencia_desalacion) * 15_000_000
#     coste_Total = coste_Solar + coste_Wind + coste_Bat + coste_Desal
    
#     # 3. Llama a la función maestra
#     results, metrics = generar_escenario_sintetico(
#         # Pasa los parámetros del escenario
#         **escenario,
        
#         # Pasa los datos base
#         df_demanda=demanda,
#         df_nucleares_base=nuclears_base, # Suponiendo que tienes un df con las 3 por separado
#         df_cogeneracion=cogeneracion_h,
#         df_solar=solar_h,
#         df_eolica=eolica_h,
#         df_potencia_historica=potencia,
#         df_capacidad_internes=capacidad_internes,
#         df_capacidad_ebre=capacidad_ebre,
#         energia_turbinada_mensual_internes=energia_turbinada_mensual_internes,
#         energia_turbinada_mensual_ebre=energia_turbinada_mensual_ebre,
        
#         # Pasa los parámetros físicos
#         potencia_max_hidro={'ebro': 1713, 'int': 207},
#         sensibilidad_hidro=400,
#         capacidad_max_hidro={'ebro': max_capacity_ebro, 'int': max_capacity_int}, # Debes tener estas variables definidas
#         umbral_overflow_pct={'ebro': 75.0, 'int': 75.0}
#     )
    
#     print(metrics)
    
#     results.hydro_int = np.clip(results.hydro_int,0,potencia_max_hidraulica_int)
#     results.hydro_ebro = np.clip(results.hydro_ebro,0,potencia_max_hidraulica_ebro)
    
#     # surpluses = df_resultado_A['Gas+Imports'] + df_resultado_A['Cogen_w'] + df_resultado_A['Solar_w'] + df_resultado_A['Eòlica_w'] + df_resultado_A['Nuclear'] + df_resultado_A['Hidráulica'] + df_resultado_A['Bateries'] - df_resultado_A['Càrrega']
#     surpluses = results.gap
#     surpluses[surpluses > 0] = 0
#     surpluses *= -1
    
    
#     # 1) Nivel “sin real”
#     sr = remove_real_desal(
#         base_level=results['Hydro_Level_int'],
#         dessalacio_diaria=dessalacio_diaria,
#         max_capacity_int=max_capacity_int
#     )
    
#     ahorro, niveles = remove_restrictions(100*capacidad_internes.Nivel['2015-06-01':'2024-12-31'])
#     ahorro = 100 * ahorro.resample('h').interpolate(method='linear') / max_capacity_int
#     ahorro = ahorro['2015-06-03':]
    
#     # 1.5 Nivel sin desalación histórica ni restricciones de consumo
#     sr2 = sr - ahorro
    
    
#     level_final, desal_final, savings_final, extra_hydro_final, thermal_reduced_final = simulate_full_water_management(
#         surpluses=surpluses,
#         level_base=sr2,
#         thermal_generation=results['Gas+Imports'],
#         base_hydro_generation=results['hydro_int'],
#         max_capacity_int=max_capacity_int,
#         consumo_base_diario_hm3=2,
#         # ... otros parámetros existentes ...
#         max_hydro_capacity_mw=potencia_max_int,
#         overflow_threshold_pct=95.0,  # Activar al 95%
#         sensitivity_mwh_per_percent= max_capacity_int * 0.01 * sensibility_int  # Tu factor de conversión
#     )
    
    
#     desal_final_hm3 = desal_final * 0.01 / 30
    
#     print('Días con restricciones de consumo de agua')
#     print('-----------------------------------------')
#     print('Datos históricos:',capacidad_internes[(100*capacidad_internes.Nivel) < 40]['2015-06-01':'2024-12-31'].count().iloc[0]*7)
#     print('Escenario + desalación histórica:',int(results[results.Hydro_Level_int < 40].count().iloc[0]/24))
#     print('Escenario + desalación de excedentes:',int(level_final[level_final < 40].count()/24))
#     print('-----------------------------------------')
#     print('Dessalació histórica (hm3):',int(dessalacio_diaria.sum()))
#     print('Dessalació simulada (hm3):',int(desal_final_hm3.sum()))
#     print('Factor de capacidad (%):', int(100 * desal_final.mean() / max_desalation))
#     print('Máximo drowdown (%):', round(min(level_final)))
#     print('-----------------------------------------')
#     print('Restricciones históricas (hm3):', int(max_capacity_int * ahorro.iloc[-1] / 100))
#     print('Restricciones en escenario (hm3):', int(savings_final.sum()))
    
    
#     results.hydro_int += extra_hydro_final
#     results.Hidráulica = results.hydro_ebre + results.hydro_int
#     results['Gas+Imports'] -= thermal_reduced_final
#     results['Hydro_Level_int'] = level_final

# volver a calcular métricas tras estos cambios.

# 606 ms ± 9.92 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 1.23 s ± 20.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
#%%


def analyze_drought_phases(
    reservoir_levels: pd.Series,
    umbrales_sequia: dict = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Analiza los días en cada fase de sequía según los niveles de embalse.
    
    Args:
        reservoir_levels: Serie temporal con niveles de embalse (%)
        umbrales_sequia: Diccionario con umbrales de cada fase
        verbose: Si mostrar visualización y estadísticas por pantalla
        
    Returns:
        np.ndarray: Array 1D con días en cada fase [Normalidad, Prealerta, Alerta, 
                   Excepcionalidad, Emergencia I, Emergencia II, Emergencia III]
    """
    
    # Umbrales por defecto
    if umbrales_sequia is None:
        umbrales_sequia = {
            'Emergencia III': 5.4,
            'Emergencia II': 10.95, 
            'Emergencia I': 16.3,
            'Excepcionalidad': 25.0,
            'Alerta': 40.0,
            'Prealerta': 60.0
        }
    
    # Función para clasificar cada nivel
    def get_fase_sequia(nivel_pct):
        if nivel_pct < umbrales_sequia['Emergencia III']: 
            return 'Emergencia III'
        elif nivel_pct < umbrales_sequia['Emergencia II']: 
            return 'Emergencia II'
        elif nivel_pct < umbrales_sequia['Emergencia I']: 
            return 'Emergencia I'
        elif nivel_pct < umbrales_sequia['Excepcionalidad']: 
            return 'Excepcionalidad'
        elif nivel_pct < umbrales_sequia['Alerta']: 
            return 'Alerta'
        elif nivel_pct < umbrales_sequia['Prealerta']: 
            return 'Prealerta'
        else: 
            return 'Normalitat'
    
    # Clasificar cada observación
    fases = reservoir_levels.apply(get_fase_sequia)
    
    # Convertir a días (asumiendo datos horarios)
    hours_per_day = 24
    if len(reservoir_levels) > 1:
        # Detectar frecuencia automáticamente
        time_diff = reservoir_levels.index[1] - reservoir_levels.index[0]
        if hasattr(time_diff, 'total_seconds'):
            hours_per_observation = time_diff.total_seconds() / 3600
        else:
            hours_per_observation = 1  # Asumir horario si no se puede detectar
    else:
        hours_per_observation = 1
    
    observations_per_day = hours_per_day / hours_per_observation
    
    # Contar observaciones por fase
    fase_counts = fases.value_counts()
    
    # Ordenar las fases de menos a más severa
    fases_ordenadas = [
        'Normalitat', 'Prealerta', 'Alerta', 'Excepcionalidad',
        'Emergencia I', 'Emergencia II', 'Emergencia III'
    ]
    
    # Crear array de salida con días por fase
    dias_por_fase = np.zeros(7)
    for i, fase in enumerate(fases_ordenadas):
        if fase in fase_counts:
            dias_por_fase[i] = fase_counts[fase] / observations_per_day
    
    # Mostrar resultados si verbose=True
    if verbose:
        print("=" * 60)
        print("ANÁLISIS DE FASES DE SEQUÍA")
        print("=" * 60)
        
        total_dias = dias_por_fase.sum()
        
        print(f"\nPeríodo analizado: {total_dias:.1f} días")
        print(f"Desde: {reservoir_levels.index[0].strftime('%Y-%m-%d')}")
        print(f"Hasta: {reservoir_levels.index[-1].strftime('%Y-%m-%d')}")
        print(f"Nivel promedio: {reservoir_levels.mean():.1f}%")
        print(f"Nivel mínimo: {reservoir_levels.min():.1f}%")
        print(f"Nivel máximo: {reservoir_levels.max():.1f}%")
        
        print("\n" + "-" * 60)
        print("DÍAS POR FASE DE SEQUÍA:")
        print("-" * 60)
        
        for i, fase in enumerate(fases_ordenadas):
            dias = dias_por_fase[i]
            porcentaje = (dias / total_dias) * 100 if total_dias > 0 else 0
            
            # Colores para la visualización
            if fase == 'Normalitat':
                color_icon = "🟢"
            elif fase == 'Prealerta':
                color_icon = "🟡"
            elif fase == 'Alerta':
                color_icon = "🟠"
            elif fase == 'Excepcionalidad':
                color_icon = "🔶"
            elif 'Emergencia' in fase:
                color_icon = "🔴"
            
            print(f"{color_icon} {fase:15s}: {dias:8.1f} días ({porcentaje:5.1f}%)")
        
        print("-" * 60)
        
        # Estadísticas adicionales
        dias_criticos = dias_por_fase[4:].sum()  # Emergencias I, II, III
        dias_problematicos = dias_por_fase[2:].sum()  # Alerta en adelante
        
        print(f"\nRESUMEN:")
        print(f"• Días en situación normal/prealerta: {dias_por_fase[:2].sum():.1f} días ({(dias_por_fase[:2].sum()/total_dias)*100:.1f}%)")
        print(f"• Días con restricciones (alerta+): {dias_problematicos:.1f} días ({(dias_problematicos/total_dias)*100:.1f}%)")
        print(f"• Días en emergencia: {dias_criticos:.1f} días ({(dias_criticos/total_dias)*100:.1f}%)")
        
        # Crear visualización
        _plot_drought_analysis(dias_por_fase, fases_ordenadas, umbrales_sequia, reservoir_levels)
    
    return dias_por_fase


def _plot_drought_analysis(dias_por_fase, fases_ordenadas, umbrales_sequia, reservoir_levels):
    """Función auxiliar para crear las visualizaciones."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Días por fase (barras)
    colores = ['#2E8B57', '#FFD700', '#FF8C00', '#FF6347', '#DC143C', '#8B0000', '#4B0000']
    
    bars = ax1.bar(range(len(fases_ordenadas)), dias_por_fase, color=colores, alpha=0.8)
    ax1.set_xlabel('Fase de Sequía')
    ax1.set_ylabel('Días')
    ax1.set_title('Distribución de Días por Fase de Sequía')
    ax1.set_xticks(range(len(fases_ordenadas)))
    ax1.set_xticklabels([f.replace(' ', '\n') for f in fases_ordenadas], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    for bar, dias in zip(bars, dias_por_fase):
        if dias > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dias_por_fase)*0.01,
                    f'{dias:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 2: Serie temporal con bandas de color
    ax2.plot(reservoir_levels.index, reservoir_levels.values, 'b-', linewidth=1, alpha=0.7)
    
    # Añadir bandas horizontales para cada umbral
    y_max = max(reservoir_levels.max(), 100)
    
    # Banda de normalidad (arriba)
    ax2.axhspan(umbrales_sequia['Prealerta'], y_max, alpha=0.15, color='green', label='Normalitat')
    
    # Prealerta
    ax2.axhspan(umbrales_sequia['Alerta'], umbrales_sequia['Prealerta'], alpha=0.15, color='gold', label='Prealerta')
    
    # Alerta
    ax2.axhspan(umbrales_sequia['Excepcionalidad'], umbrales_sequia['Alerta'], alpha=0.15, color='orange', label='Alerta')
    
    # Excepcionalidad
    ax2.axhspan(umbrales_sequia['Emergencia I'], umbrales_sequia['Excepcionalidad'], alpha=0.15, color='coral', label='Excepcionalidad')
    
    # Emergencias
    ax2.axhspan(umbrales_sequia['Emergencia II'], umbrales_sequia['Emergencia I'], alpha=0.15, color='red', label='Emergencia I')
    ax2.axhspan(umbrales_sequia['Emergencia III'], umbrales_sequia['Emergencia II'], alpha=0.15, color='darkred', label='Emergencia II')
    ax2.axhspan(0, umbrales_sequia['Emergencia III'], alpha=0.15, color='maroon', label='Emergencia III')
    
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Nivel del Embalse (%)')
    ax2.set_title('Evolución Temporal del Nivel del Embalse')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Gráficos generados: Distribución por fases y evolución temporal")


# Función auxiliar para análisis rápido
def quick_drought_summary(reservoir_levels: pd.Series, umbrales_sequia: dict = None) -> dict:
    """
    Versión simplificada que retorna solo un diccionario con estadísticas básicas.
    """
    dias_array = analyze_drought_phases(reservoir_levels, umbrales_sequia, verbose=False)
    
    fases_ordenadas = [
        'Normalitat', 'Prealerta', 'Alerta', 'Excepcionalidad',
        'Emergencia I', 'Emergencia II', 'Emergencia III'
    ]
    
    total_dias = dias_array.sum()
    
    return {
        'dias_por_fase': dict(zip(fases_ordenadas, dias_array)),
        'total_dias': total_dias,
        'dias_criticos': dias_array[4:].sum(),  # Emergencias
        'dias_normales': dias_array[:2].sum(),  # Normal + Prealerta
        'porcentaje_critico': (dias_array[4:].sum() / total_dias * 100) if total_dias > 0 else 0,
        'nivel_promedio': reservoir_levels.mean(),
        'nivel_minimo': reservoir_levels.min(),
        'nivel_maximo': reservoir_levels.max()
    }

# Análisis completo con visualización
dias = analyze_drought_phases(level_final, verbose=True)
print(f"Días en emergencia total: {dias[4:].sum():.1f}")

# Solo obtener el array sin visualización
dias = analyze_drought_phases(level_final, verbose=False)

# Resumen rápido en diccionario
summary = quick_drought_summary(level_final)
print(f"Porcentaje crítico: {summary['porcentaje_critico']:.1f}%")

#%%
%%time
import time

from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')

# Opcional: Si tienes DEAP instalado
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("DEAP no disponible. Usando scipy.optimize")

# # Opcional: Si tienes pygmo instalado
# try:
#     import pygmo as pg
#     PYGMO_AVAILABLE = True
# except ImportError:
#     PYGMO_AVAILABLE = False

class OptimizadorEnergetico:
    """
    Clase para optimizar escenarios energéticos usando diferentes metaheurísticas
    """
    
    def __init__(self, datos_base, funcion_procesar_escenario):
        """
        Inicializa el optimizador
        
        Args:
            datos_base: Diccionario con todos los DataFrames y parámetros fijos
            funcion_procesar_escenario: Tu función procesar_escenario
        """
        self.datos_base = datos_base
        self.procesar_escenario = funcion_procesar_escenario
        self.mejor_resultado = None
        self.historial_evaluaciones = []
        
    def definir_variables_decision(self):
        """
        Define los rangos de las variables de decisión
        """
        self.variables = {
            'potencia_solar': {
                'tipo': 'continuous',
                'min': 408.64,    # Valor actual
                'max': 20000       # Máximo deseado
            },
            'potencia_eolica': {
                'tipo': 'continuous', 
                'min': 1406.15,    # Valor actual
                'max': 10000        # Máximo deseado
            },
            'potencia_baterias': {
                'tipo': 'continuous',
                'min': 500,        # Actual
                'max': 5000        # Máximo
            },
            'max_desalation': {
                'tipo': 'continuous',
                'min': 30,         # Actual
                'max': 200         # Máximo
            },
            'overflow_threshold_pct': {
                'tipo': 'continuous',
                'min': 60,
                'max': 98
            },
            'min_run_hours': {
                'tipo': 'integer',
                'min': 1,
                'max': 10
            },
            'midpoint_estimation': {
                'tipo': 'continuous',
                'min': 10,
                'max': 100
            },
            'seasonal_phase_months': {
                'tipo': 'continuous',
                'min': 0,
                'max': 11.99
            },
            'seasonal_desal_amplitude': {
                'tipo': 'continuous',
                'min': 0,
                'max': 1
            }
        }
        
    def decodificar_individuo(self, x):
        """
        Convierte el vector de optimización en parámetros para la función
        """
        params = {}
        idx = 0
        
        # Variables continuas
        params['potencia_solar'] = x[idx]
        idx += 1
        params['potencia_eolica'] = x[idx]
        idx += 1
        params['potencia_baterias'] = x[idx]
        idx += 1
        params['max_desalation'] = x[idx]
        idx += 1
        params['overflow_threshold_pct'] = x[idx]
        idx += 1
        
        # Variable entera
        params['min_run_hours'] = int(round(x[idx]))
        idx += 1
        
        # Variable continua adicional
        params['midpoint_estimation'] = x[idx]
        idx += 1
        params['seasonal_phase_months'] = x[idx]
        idx += 1
        params['seasonal_desal_amplitude'] = x[idx]

        
        return params
        
    def funcion_objetivo(self, x):
        """
        Función objetivo a minimizar/maximizar con restricción de desbordamiento
        """
        try:
            # Decodificar parámetros
            params = self.decodificar_individuo(x)
            
            # Ejecutar simulación
            resultado = self.procesar_escenario(
                **self.datos_base,
                **params
            )
            
            # RESTRICCIÓN CRÍTICA: Verificar que no hay desbordamiento
            level_final = resultado['level_final']
            if (level_final > 100).any():
                # Penalizar severamente escenarios con desbordamiento
                return 1e6  # Valor muy alto para minimización
            
            # Definir objetivos (puedes modificar según tus necesidades)
            coste_total = resultado['costes']['total']
            emisiones = resultado['energy_data']['Gas+Imports']
            seasonal_amplitude = resultado['hydro_metrics']['Variación estacional (%)']
            restricciones_dias = resultado['hydro_metrics']['Restricciones escenario (días)']
            minimo_llenado = resultado['hydro_metrics']['Llenado mínimo (%)']
            excedentes_post_desal = resultado.get('surpluses_afterdesal', 0)  # Tu variable 2

            
            # Función objetivo multiobjetivo ponderada
            # Minimizar costes, minimizar amplitud estacional, minimizar restricciones
            # Maximizar mínimo llenado
            
            objetivo = (
                0.25 * (coste_total / 1e9) +  # Normalizar costes
                # 0.3 * (seasonal_amplitude / 100) +  # Normalizar amplitud
                # 0.2 * (restricciones_dias / 365) +  # Normalizar días
                0.5 * (1 - minimo_llenado / 100) +   # Penalizar bajo llenado
                0.25 * abs(excedentes_post_desal / 1e6)  # Tu variable 2 (normalizada)
            )
            
            # Guardar historial
            self.historial_evaluaciones.append({
                'parametros': params.copy(),
                'resultado': resultado,
                'objetivo': objetivo
            })
            
            return objetivo
            
        except Exception as e:
            print(f"Error en evaluación: {e}")
            return 1e6  # Penalizar configuraciones inválidas
    
    def optimizar_scipy_differential_evolution(self, maxiter=50, popsize=15):
        """
        Optimización usando Differential Evolution de scipy
        """
        print("Iniciando optimización con Differential Evolution...")
        
        # Definir bounds
        bounds = [
            # (0, len(self.variables['nucleares_activas']['opciones'])-1),  # nucleares (categórico)
            (self.variables['potencia_solar']['min'], self.variables['potencia_solar']['max']),
            (self.variables['potencia_eolica']['min'], self.variables['potencia_eolica']['max']),
            (self.variables['potencia_baterias']['min'], self.variables['potencia_baterias']['max']),
            (self.variables['max_desalation']['min'], self.variables['max_desalation']['max']),
            (self.variables['overflow_threshold_pct']['min'], self.variables['overflow_threshold_pct']['max']),
            (self.variables['min_run_hours']['min'], self.variables['min_run_hours']['max']),
            (self.variables['midpoint_estimation']['min'], self.variables['midpoint_estimation']['max']),
            (self.variables['seasonal_phase_months']['min'], self.variables['seasonal_phase_months']['max']),
            (self.variables['seasonal_desal_amplitude']['min'], self.variables['seasonal_desal_amplitude']['max'])
        ]
        
        def callback(xk, convergence):
            """Callback para mostrar progreso"""
            if len(self.historial_evaluaciones) % 10 == 0 and len(self.historial_evaluaciones) > 0:
                objetivos_validos = [h['objetivo'] for h in self.historial_evaluaciones if h['objetivo'] < 1e5]
                if objetivos_validos:
                    mejor_obj = min(objetivos_validos)
                    print(f"Evaluación {len(self.historial_evaluaciones)}: Mejor objetivo = {mejor_obj:.4f}")
                else:
                    print(f"Evaluación {len(self.historial_evaluaciones)}: Todas las evaluaciones han fallado")
        
        # Ejecutar optimización
        resultado = differential_evolution(
            self.funcion_objetivo,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=42,
            callback=callback,
            disp=True
        )
        
        # Procesar resultado
        mejor_params = self.decodificar_individuo(resultado.x)
        
        print(f"\\nOptimización completada!")
        print(f"Mejor objetivo: {resultado.fun:.4f}")
        print("Mejores parámetros:")
        for k, v in mejor_params.items():
            print(f"  {k}: {v}")
            
        return resultado, mejor_params
    
    def optimizar_con_nsga2_deap(self, ngen=30, pop_size=50):
        """
        Optimización multiobjetivo usando NSGA-II de DEAP
        """
        if not DEAP_AVAILABLE:
            print("DEAP no está disponible. Instálalo con: pip install deap")
            return None
            
        # Configurar DEAP
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, +1.0, -1.0, -1.0))  # 3 objetivos a minimizar
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        # Generadores para cada variable
        # toolbox.register("nuclear", np.random.randint, 0, len(self.variables['nucleares_activas']['opciones']))
        toolbox.register("solar", np.random.uniform, 
                        self.variables['potencia_solar']['min'], 
                        self.variables['potencia_solar']['max'])
        toolbox.register("eolica", np.random.uniform,
                        self.variables['potencia_eolica']['min'],
                        self.variables['potencia_eolica']['max'])
        toolbox.register("baterias", np.random.uniform,
                        self.variables['potencia_baterias']['min'],
                        self.variables['potencia_baterias']['max'])
        toolbox.register("desal", np.random.uniform,
                        self.variables['max_desalation']['min'],
                        self.variables['max_desalation']['max'])
        toolbox.register("overflow", np.random.uniform,
                        self.variables['overflow_threshold_pct']['min'],
                        self.variables['overflow_threshold_pct']['max'])
        toolbox.register("min_hours", np.random.randint,
                        self.variables['min_run_hours']['min'],
                        self.variables['min_run_hours']['max'] + 1)
        toolbox.register("midpoint", np.random.uniform,
                        self.variables['midpoint_estimation']['min'],
                        self.variables['midpoint_estimation']['max'])
        toolbox.register("phase", np.random.uniform,
                        self.variables['seasonal_phase_months']['min'],
                        self.variables['seasonal_phase_months']['max'])
        toolbox.register("amplitude", np.random.uniform,
                        self.variables['seasonal_desal_amplitude']['min'],
                        self.variables['seasonal_desal_amplitude']['max'])        
        
        # Crear individuos
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.solar, toolbox.eolica, 
                         toolbox.baterias, toolbox.desal, toolbox.overflow,
                         toolbox.min_hours, toolbox.midpoint, 
                         toolbox.phase, toolbox.amplitude), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluar_multiobjetivo(individual):
            """Función de evaluación multiobjetivo con restricción de desbordamiento"""
            params = self.decodificar_individuo(individual)
            resultado = self.procesar_escenario(**self.datos_base, **params)
            
            # Verificar restricción de desbordamiento
            level_final = resultado['level_final']
            if (level_final > 100).any():
                # Devolver valores muy altos para penalizar
                return 1e10, -1e6, 1e10, 1e10
            
            # Tres objetivos separados
            emisiones = resultado['energy_data']['Gas+Imports'].sum()/1000
            # restricciones_dias = resultado['hydro_metrics']['Restricciones escenario (días)'] / 365
            # restricciones_hm3 = resultado['savings_final'].sum()
            llenado_minimo = resultado['level_final'].min()
            # llenado_medio = 100 - resultado['level_final'].mean()
            # factor_capacidad = 100 - resultado['capacity_factor']
            # amplitud = resultado['hydro_metrics']['Variación estacional (%)'] / 100
            excedentes_post_desal = resultado.get('surpluses_afterdesal', 0) /1000  # Tu variable 2
            costes = resultado['costes']['total'] / 1e6


            
            
            return emisiones, llenado_minimo, excedentes_post_desal, costes
        
        toolbox.register("evaluate", evaluar_multiobjetivo)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        # toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        # Después de toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        def checkBounds(min_bounds, max_bounds):
            def decorator(func):
                def wrapper(*args, **kargs):
                    offspring = func(*args, **kargs)
                    for child in offspring:
                        for i in range(len(child)):
                            if child[i] < min_bounds[i]:
                                child[i] = min_bounds[i]
                            elif child[i] > max_bounds[i]:
                                child[i] = max_bounds[i]
                    return offspring
                return wrapper
            return decorator
        
        min_bounds = [self.variables['potencia_solar']['min'], self.variables['potencia_eolica']['min'], 
                      self.variables['potencia_baterias']['min'], self.variables['max_desalation']['min'],
                      self.variables['overflow_threshold_pct']['min'], self.variables['min_run_hours']['min'],
                      self.variables['midpoint_estimation']['min'], self.variables['seasonal_phase_months']['min'],
                      self.variables['seasonal_desal_amplitude']['min']]
        
        max_bounds = [self.variables['potencia_solar']['max'], self.variables['potencia_eolica']['max'],
                      self.variables['potencia_baterias']['max'], self.variables['max_desalation']['max'],
                      self.variables['overflow_threshold_pct']['max'], self.variables['min_run_hours']['max'],
                      self.variables['midpoint_estimation']['max'], self.variables['seasonal_phase_months']['max'],
                      self.variables['seasonal_desal_amplitude']['max']]
        
        toolbox.decorate("mate", checkBounds(min_bounds, max_bounds))
        toolbox.decorate("mutate", checkBounds(min_bounds, max_bounds))        
        # toolbox.register("mutate", tools.mutPolynomialBounded, low=[0, self.variables['potencia_solar']['min'], 
        #           self.variables['potencia_eolica']['min'], self.variables['potencia_baterias']['min'],
        #           self.variables['max_desalation']['min'], self.variables['overflow_threshold_pct']['min'],
        #           self.variables['min_run_hours']['min'], self.variables['midpoint_estimation']['min'],
        #           self.variables['seasonal_phase_months']['min'], self.variables['seasonal_desal_amplitude']['min']], 
        #           up=[0, self.variables['potencia_solar']['max'], self.variables['potencia_eolica']['max'],
        #           self.variables['potencia_baterias']['max'], self.variables['max_desalation']['max'],
        #           self.variables['overflow_threshold_pct']['max'], self.variables['min_run_hours']['max'],
        #           self.variables['midpoint_estimation']['max'], self.variables['seasonal_phase_months']['max'],
        #           self.variables['seasonal_desal_amplitude']['max']], eta=20.0, indpb=0.1)
        toolbox.register("select", tools.selNSGA2)
        
        # NUEVO: Paralelización con joblib
        def parallel_evaluate(func, individuals):
            return Parallel(n_jobs=-1)(delayed(func)(ind) for ind in individuals)
        toolbox.register("map", parallel_evaluate)
        
        # Después de toolbox.register("select", tools.selNSGA2)
        np.random.seed(42)
        
        # Ejecutar algoritmo
        pop = toolbox.population(n=pop_size)
        hof = tools.ParetoFront()
        
        pop, logbook = algorithms.eaMuPlusLambda(
            pop, toolbox, mu=pop_size, lambda_=pop_size,
            cxpb=0.7, mutpb=0.3, ngen=ngen,
            halloffame=hof, verbose=True
        )
        
        return pop, hof, logbook

def configurar_optimizacion(procesar_escenario_func, datos_reales, potencia_cogeneracion_fija=943.503):
    """
    Función helper para configurar rápidamente la optimización
    
    Args:
        procesar_escenario_func: Tu función procesar_escenario
        datos_reales: Diccionario con todos los DataFrames y parámetros reales
        potencia_cogeneracion_fija: Valor fijo para cogeneración
    """
    # Añadir cogeneración fija a los datos
    datos_reales['potencia_cogeneracion'] = potencia_cogeneracion_fija
    
    # Añadir nucleares fijas a los datos
    datos_reales['nucleares_activas'] = [True, True, True]
    
    # Crear optimizador
    optimizador = OptimizadorEnergetico(datos_reales, procesar_escenario_func)
    optimizador.definir_variables_decision()
    
    return optimizador

# Ejemplo de uso
if __name__ == "__main__":
    t0 = time.time()
    
    # IMPORTANTE: Configurar con tus datos reales
    datos_reales = {
        'df_demanda': demanda,  # Cambia por tus variables reales
        'df_nuclear': nuclears_base,
        'df_cogeneracion': cogeneracion_h,
        'df_solar': solar_h,
        'df_eolica': eolica_h,
        'df_potencia': potencia,  # ESTE ES CRÍTICO - no puede ser None
        'df_niveles_int': capacidad_internes,
        'df_niveles_ebro': capacidad_ebre,
        'df_energia_turbinada_mensual_internes': energia_turbinada_mensual_internes,
        'df_energia_turbinada_mensual_ebre': energia_turbinada_mensual_ebre,
        'df_nivel_si': nivel_si,
        'max_capacity_int': max_capacity_int,
        'max_capacity_ebro': max_capacity_ebro,
        'potencia_max_int': potencia_max_hidraulica_int,
        'potencia_max_ebro': potencia_max_hidraulica_ebro,
        'consumo_base_diario_hm3': consumo_base_diario_hm3,
        'sensibility_int': sensibility_int,
        'sensibility_ebro': sensibility_ebro,
    }
    
    # Configurar optimización con datos reales
    optimizador = configurar_optimizacion(procesar_escenario, datos_reales, potencia_cogeneracion_fija=943.503)
    
    # # Opción 1: Differential Evolution (recomendado para empezar)
    # resultado_de, mejores_params = optimizador.optimizar_scipy_differential_evolution(
    #     maxiter=1,  # Pocas iteraciones para prueba
    #     popsize=8   # Población pequeña para prueba
    # )
    
      
    # Opción 2: NSGA-II multiobjetivo (si tienes DEAP)
    if DEAP_AVAILABLE:
        pop, hof, logbook = optimizador.optimizar_con_nsga2_deap(ngen=10, pop_size=20)

    print("Tiempo de ejecución:", time.time() - t0, "s")

# # 1. Configurar con tus datos reales (REEMPLAZA None por tus variables)
# datos_reales = {
#     'df_demanda': demanda,  # Tu DataFrame de demanda
#     'df_nuclear': nuclears_base,  # Tu DataFrame nuclear
#     'df_cogeneracion': cogeneracion_h,
#     'df_solar': solar_h,
#     'df_eolica': eolica_h,
#     'df_potencia': potencia,  # CRÍTICO: Este es el que causaba el error
#     'df_niveles_int': capacidad_internes,
#     'df_niveles_ebro': capacidad_ebre,
#     'df_energia_turbinada_mensual_internes': energia_turbinada_mensual_internes,
#     'df_energia_turbinada_mensual_ebre': energia_turbinada_mensual_ebre,
#     'df_nivel_si': nivel_si,
#     # Parámetros numéricos
#     'max_capacity_int': max_capacity_int,
#     'max_capacity_ebro': max_capacity_ebro,
#     'potencia_max_int': potencia_max_int,
#     'potencia_max_ebro': potencia_max_ebro,
#     'consumo_base_diario_hm3': consumo_base_diario_hm3,
#     'sensibility_int': sensibility_int,
#     'sensibility_ebro': sensibility_ebro,
# }

# # 2. Crear optimizador
# optimizador = configurar_optimizacion(procesar_escenario, datos_reales, potencia_cogeneracion_fija=943.503)

# # 3. Ejecutar optimización
# resultado_de, mejores_params = optimizador.optimizar_scipy_differential_evolution(
#     maxiter=10,  # Empezar con pocas iteraciones
#     popsize=8,    # Población pequeña para test
#     workers=-1
# )


# Mejores parámetros:
#   nucleares_activas: [True, True, True]
#   potencia_solar: 12685.723575688524
#   potencia_eolica: 8922.8584337011
#   potencia_baterias: 866.6949906323123
#   max_desalation: 163.09138321302126
#   overflow_threshold_pct: 68.78460058374851
#   min_run_hours: 4
#   midpoint_estimation: 50.115756338716295


# Mejores parámetros:
#   nucleares_activas: [True, True, True]
#   potencia_solar: 12519.378625240748
#   potencia_eolica: 7266.693158045117
#   potencia_baterias: 694.2818110865524
#   max_desalation: 171.9731736152927
#   overflow_threshold_pct: 71.45965007511684
#   min_run_hours: 3
#   midpoint_estimation: 50.77640267964148

# #-----
#     potencia_solar = 2139.99, #12685,
#     potencia_eolica = 2475.71, #8922,
#     potencia_baterias = 1308.68, #866.69,
#     min_run_hours = 4, #4,
#     max_desalation = 101.43, #90,
#     midpoint_estimation = 81.64, #54 # parámetro del sigmoide de desalación
#     overflow_threshold_pct = 83.51 #90, #95

#%%
def extraer_resultados_nsga2(optimizador, hof):
    """Extrae y procesa los resultados del NSGA-II"""
    
    # Convertir frente de Pareto a DataFrame
    pareto_solutions = []
    for ind in hof:
        params = optimizador.decodificar_individuo(ind)
        objetivos = ind.fitness.values
        
        pareto_solutions.append({
            **params,
            'objetivo_1_emisiones': objetivos[0],
            'objetivo_2_llenado_min': objetivos[1], 
            'objetivo_3_excedentes': objetivos[2]
        })
    
    df_pareto = pd.DataFrame(pareto_solutions)
    
    print(f"Frente de Pareto: {len(df_pareto)} soluciones")
    print("\nMejores 5 soluciones:")
    print(df_pareto.head())
    
    return df_pareto

# Después de ejecutar NSGA-II:
df_pareto = extraer_resultados_nsga2(optimizador, hof)

# # Filtrar solo soluciones factibles (sin penalizaciones)
# df_factibles = df_pareto[
#     (df_pareto['objetivo_1_emisiones'] < 1e5) &
#     (df_pareto['objetivo_2_llenado_min'] > -1e5) &
#     (df_pareto['objetivo_3_excedentes'] < 1e5)
# ]
# print(f"Soluciones factibles: {len(df_factibles)} de {len(df_pareto)}")

# Plotting 2D para cada par
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Par 1: Emisiones vs Llenado
axes[0].scatter(df_pareto['objetivo_1_emisiones'], df_pareto['objetivo_2_llenado_min'], color='red')
axes[0].set_xlabel('Emisiones (minimizar)')
axes[0].set_ylabel('Llenado Min (maximizar)')
axes[0].set_title('Frente Pareto: Emisiones vs Llenado')

# Par 2: Emisiones vs Excedentes  
axes[1].scatter(df_pareto['objetivo_1_emisiones'], df_pareto['objetivo_3_excedentes'], color='blue')
axes[1].set_xlabel('Emisiones (minimizar)')
axes[1].set_ylabel('Excedentes (minimizar)')
axes[1].set_title('Frente Pareto: Emisiones vs Excedentes')

# Par 3: Llenado vs Excedentes
axes[2].scatter(df_pareto['objetivo_2_llenado_min'], df_pareto['objetivo_3_excedentes'], color='green')
axes[2].set_xlabel('Llenado Min (maximizar)')
axes[2].set_ylabel('Excedentes (minimizar)')
axes[2].set_title('Frente Pareto: Llenado vs Excedentes')

plt.tight_layout()
plt.show()

# Plotting 3D con Plotly
plot_pareto_3d_plotly(
    df_pareto, df_pareto,  # Mismo df para ambos (solo pareto)
    'objetivo_1_emisiones', 'objetivo_2_llenado_min', 'objetivo_3_excedentes',
    'min', 'max', 'min'
)

#%%
def extraer_todas_las_soluciones(optimizador):
    """Extrae todas las soluciones evaluadas durante la optimización"""
    
    todas_soluciones = []
    
    for eval_data in optimizador.historial_evaluaciones:
        # Variables de entrada
        params = eval_data['parametros']
        
        # Variables de salida del resultado completo
        resultado = eval_data['resultado']
        
        solucion = {
            # Variables de entrada
            'potencia_solar': params['potencia_solar'],
            'potencia_eolica': params['potencia_eolica'], 
            'potencia_baterias': params['potencia_baterias'],
            'max_desalation': params['max_desalation'],
            'overflow_threshold_pct': params['overflow_threshold_pct'],
            'min_run_hours': params['min_run_hours'],
            'midpoint_estimation': params['midpoint_estimation'],
            'seasonal_phase_months': params['seasonal_phase_months'],
            'seasonal_desal_amplitude': params['seasonal_desal_amplitude'],
            
            # Variables de salida (objetivos y métricas)
            'emisiones_gas_imports': resultado['energy_data']['Gas+Imports'].sum(),
            'llenado_minimo': resultado['level_final'].min(),
            'llenado_promedio': resultado['level_final'].mean(),
            'excedentes_post_desal': resultado.get('surpluses_afterdesal', 0),
            'restricciones_dias': resultado['hydro_metrics']['Restricciones escenario (días)'],
            'factor_capacidad_desal': resultado['capacity_factor'],
            'coste_total': resultado['costes']['total'],
            'variacion_estacional': resultado['hydro_metrics']['Variación estacional (%)'],
            'objetivo_combinado': eval_data['objetivo']
        }
        
        todas_soluciones.append(solucion)
    
    df_todas = pd.DataFrame(todas_soluciones)
    print(f"Total soluciones evaluadas: {len(df_todas)}")
    
    return df_todas

# Uso después de la optimización:
df_todas_soluciones = extraer_todas_las_soluciones(optimizador)
print(df_todas_soluciones.describe())


#%%

from matplotlib.patches import Patch

# Tus datos de ejemplo
data = [
    ['D_00', 'Demanda i generació elèctrica per font a Catalunya', 'Mensual', 2011, 'REE'],
    ['D_01', 'Demanda elèctrica a Catalunya', 'Horària', 2005, 'Gencat'],
    ['D_02', 'Indicadors energètics de Catalunya', 'Mensual', 2005, 'Gencat'],
    ['D_03', 'Generació nuclear, segons reactor', 'Diària', 2005, 'CSN'],
    ['D_04', 'Potencia elèctrica instal·lada per font', 'Mensual', 2015, 'REE'],
    ['D_05', 'Autoconsum instal·lat', 'Mensual', 2018, 'Gencat (RAC)'],
    ['D_06', 'Generació elèctrica xarxa peninsular', 'Horària', 2015, 'REE (ESIOS)'],
    ['D_07', 'Balanç elèctric de Catalunya', 'Anual', 1988, 'Gencat (ICAEN)'],
    ['D_08', 'Capacitat dels embassaments peninsulars', 'Setmanal', 1988, 'MITECO'],
    ['D_09', 'Energia teòrica màxima produïble als embassaments peninsulars', 'Setmanal', 1988, 'MITECO'],
    ['D_10', 'Capacitat dels embassaments a la conca de l\'Ebre', 'Diària', 1997, 'SAIH Ebre'],
    ['D_11', 'Capacitat dels embassaments a les conques internes', 'Diària', 2000, 'Gencat (ACA)'],
    ['D_12', 'Volum de dessalació', 'Diària', 2013, 'Gencat (ATL)'],
    ['D_13', 'Volum de regeneració', 'Diària', 2019, 'Gencat'],
    ['D_14', 'Dades climàtiques de Copernicus', 'Horària',1988, 'UE']
]

# Crear DataFrame
df = pd.DataFrame(data, columns=['Codi', 'Tipus', 'Granularitat', 'Inici', 'Font'])
df['Inici'] = df['Inici'].astype(int)

# Configurar granularidades y colores
granularitats = ['Horària', 'Diària', 'Setmanal', 'Mensual', 'Anual']

# Paleta suave de Seaborn
paleta_seaborn = sn.color_palette("pastel", 5)
paleta = {
    'Horària': paleta_seaborn[0],  # Azul
    'Diària': paleta_seaborn[1],   # Rojo/Naranja
    'Setmanal': paleta_seaborn[2], # Verde
    'Mensual': paleta_seaborn[3],   # Rosa
    'Anual': paleta_seaborn[4]
}

# Configurar años (2025 como referencia)
any_actual = 2026
anys = range(df['Inici'].min(), any_actual + 1)


# Ejemplo: D_05 termina en 2023
fi_especial = {'D_02': 2024.75}  # Código → año de fin

# Crear figura
plt.figure(figsize=(14, 8))

# Dibujar barras para cada dataset
for i, (idx, row) in enumerate(df[::-1].iterrows()):
    any_fi = fi_especial.get(row['Codi'], any_actual)
    # Calcular longitud de la barra (desde año inicio hasta 2024)
    longitut = any_fi - row['Inici']

    if row['Codi'] == 'D_06': plt.barh(y=i, width=2024 - 2011 + 1, left=2007, height=0.7, color=paleta['Mensual'])
    if row['Codi'] == 'D_06': plt.barh(y=i, width=2024 - 2011 + 1, left=2011, height=0.7, color=paleta['Diària'])

    
    # Dibujar barra horizontal
    plt.barh(
        y=i,
        width=longitut,
        left=row['Inici'], #- 0.5,  # Ajuste para centrar en año
        height=0.7,
        color=paleta[row['Granularitat']],
        edgecolor='white',
        linewidth=0.5
    )
    
    x_pos = row['Inici'] - 0.5 + 0.7
    
    # Ajuste específico para los dos casos problemáticos
    if row['Codi'] in ['D_14', 'D_07','D_08', 'D_09']:  # Cambia por los códigos afectados
        x_pos += 7.0  # Desplaza 1.5 unidades a la derecha
        
    if row['Codi'] == 'D_06': x_pos -= 8#4
    
    plt.text(
        x_pos,
        i, 
        row['Tipus'],
        # f"{row['Tipus']} — {row['Font']}",         
        ha='left', 
        va='center',
        fontweight='bold',
        fontsize=13
    )
    
# Configurar ejes
plt.yticks(range(len(df)), df[::-1]['Codi'])
plt.xticks(range(df['Inici'].min(), any_actual + 1, 2), rotation=0)
# plt.xlim(df['Inici'].min() - 1, any_actual + 1)
plt.xlim(1995,any_actual+1)

# Etiquetas y título
# plt.xlabel('Any', fontsize=12, labelpad=10)
plt.ylabel('Codi del dataset', fontsize=12, labelpad=10)
plt.title('Disponibilitat històrica dels datasets', 
          fontsize=16, fontweight='bold', pad=20)

# Leyenda personalizada
legend_elements = [
    Patch(facecolor=paleta[g], edgecolor='white', label=g) 
    for g in granularitats
]
plt.legend(
    handles=legend_elements,
    title='Granularitat',
    loc='upper left',
    # bbox_to_anchor=(0, 0.90),  # Desplaza ligeramente hacia abajo
    fontsize=12,
    title_fontsize=13,
    frameon=True,
    fancybox=True
)

# Indicar que la barra viene de antes
plt.text(
    1995, i-13.5, #i+0.5, 
    '← Inici 1940', 
    ha='center', va='center', 
    color='gray', fontsize=12
)

# Indicar que la barra viene de antes
plt.text(
    1995, i-8.5, #i-4.5, 
    '← Inici 1988', 
    ha='center', va='center', 
    color='gray', fontsize=12
)
plt.text(
    1995, i-7.5, #i-4.5, 
    '← Inici 1988', 
    ha='center', va='center', 
    color='gray', fontsize=12
)
plt.text(
    1995, i-6.5, #i-4.5, 
    '← Inici 1990', 
    ha='center', va='center', 
    color='gray', fontsize=12
)


# Estilo
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.gca().set_facecolor('#f9f9f9')

# Añadir línea vertical para 2015
# plt.axvline(x=any_actual, color='red', linestyle='--', alpha=0.7, label='2024')
# plt.axvline(x=2005, color='red', linestyle='--', alpha=0.7, label='2005')
# plt.axvline(x=2013, color='red', linestyle='--', alpha=0.7, label='2013')
plt.axvline(x=2015, color='red', linestyle='--', alpha=0.7, label='2015')
# plt.text(any_actual + 0.2, len(df)-1.5, '2024', color='red', fontweight='bold')
plt.text(2015 + 0.2, len(df)-0.5, 'Llindar d\'anàlisi (2015 fins al present)', color='red', fontweight='bold', fontsize=12)

plt.show()

#%%

# @njit
# def reducir_generacion_numpy(
#     hidraulica_actual: np.ndarray, 
#     energia_a_reducir: float, 
#     hydro_min: float,
#     # precios_hora: None
# ) -> np.ndarray:
#     """
#     Versión optimizada con Numba que considera precios horarios.
    
#     Mejoras:
#     - Compilación JIT para 10x más velocidad
#     - Considera precios para reducir en horas más baratas primero
#     - Convergencia garantizada con criterio dinámico
#     """
#     precios_hora = None
#     hid_ajustada = hidraulica_actual.copy()
#     energia_restante = energia_a_reducir
#     tolerancia = 1e-6
    
#     for iteracion in range(50):  # Más iteraciones permitidas
#         if energia_restante < tolerancia:
#             break
            
#         capacidad_reduccion = hid_ajustada - hydro_min
#         mascara = capacidad_reduccion > tolerancia
        
#         if not mascara.any():
#             break
        
#         # Si tenemos precios, priorizamos reducir en horas caras
#         if precios_hora is not None:
#             # Crear pesos inversamente proporcionales al precio
#             pesos = np.zeros_like(capacidad_reduccion)
#             pesos[mascara] = 1.0 / (precios_hora[mascara] + 1e-6)
#             pesos = pesos / pesos.sum() if pesos.sum() > 0 else pesos
#         else:
#             # Distribución uniforme si no hay precios
#             num_candidatas = mascara.sum()
#             pesos = np.zeros_like(capacidad_reduccion)
#             pesos[mascara] = 1.0 / num_candidatas
        
#         # Reducción proporcional a los pesos
#         reduccion_objetivo = energia_restante * pesos
#         reduccion_real = np.minimum(reduccion_objetivo, capacidad_reduccion)
        
#         hid_ajustada -= reduccion_real
#         energia_restante -= reduccion_real.sum()
        
#         # Criterio de convergencia dinámico
#         if iteracion > 0 and energia_restante > tolerancia:
#             mejora = abs(energia_anterior - energia_restante)
#             if mejora < tolerancia * 0.01:  # Sin mejora significativa
#                 break
#         energia_anterior = energia_restante
    
#     return hid_ajustada

def suavizar_excedente_numpy_mejorado(
    hidraulica_actual: np.ndarray, 
    energia_a_repartir: float, 
    hydro_min: float,
    potencia_max: float,
    factor_suavizado: float = 0.8
) -> np.ndarray:
    """
    Versión mejorada con algoritmo de water-filling adaptativo.
    
    Mejoras:
    - Factor de suavizado configurable
    - Convergencia adaptativa basada en progreso
    - Priorización por "profundidad" del valle
    """
    hid_ajustada = hidraulica_actual.copy()
    hid_ajustada = np.maximum(hid_ajustada, hydro_min)
    energia_restante = energia_a_repartir
    tolerancia = 1e-6
    
    # Historial para detectar estancamiento
    historial_energia = []
    
    for iteracion in range(50):  # Más iteraciones
        if energia_restante < tolerancia:
            break
            
        capacidad_restante = potencia_max - hid_ajustada
        mascara_candidatas = capacidad_restante > tolerancia
        
        if not mascara_candidatas.any():
            break
        
        # Identificar el "nivel del agua" objetivo
        nivel_actual_min = hid_ajustada[mascara_candidatas].min()
        nivel_actual_max = hid_ajustada[mascara_candidatas].max()
        
        # Objetivo: llenar hasta factor_suavizado entre min y max
        nivel_objetivo = nivel_actual_min + factor_suavizado * (nivel_actual_max - nivel_actual_min)
        
        # Calcular cuánto añadir a cada hora para alcanzar el nivel objetivo
        deficit_hasta_objetivo = np.zeros_like(hid_ajustada)
        deficit_hasta_objetivo[mascara_candidatas] = np.maximum(
            0, nivel_objetivo - hid_ajustada[mascara_candidatas]
        )
        
        # Limitar por capacidad y energía disponible
        total_deficit = deficit_hasta_objetivo.sum()
        if total_deficit > tolerancia:
            factor_escala = min(1.0, energia_restante / total_deficit)
            energia_anadir = deficit_hasta_objetivo * factor_escala
            energia_anadir = np.minimum(energia_anadir, capacidad_restante)
        else:
            # Si ya está nivelado, distribuir uniformemente
            num_candidatas = mascara_candidatas.sum()
            energia_por_hora = energia_restante / num_candidatas
            energia_anadir = np.zeros_like(hid_ajustada)
            energia_anadir[mascara_candidatas] = min(energia_por_hora, capacidad_restante[mascara_candidatas].min())
        
        hid_ajustada += energia_anadir
        energia_restante -= energia_anadir.sum()
        
        # Detección de estancamiento
        historial_energia.append(energia_restante)
        if len(historial_energia) > 5:
            variacion_reciente = np.std(historial_energia[-5:])
            if variacion_reciente < tolerancia * 0.001:
                # print(f"Convergencia por estancamiento en iteración {iteracion}")
                break
    
    return hid_ajustada


#%%


# 1. Crear datos de ejemplo para un día (24 horas)
horas = np.arange(24)
demanda = 100 + 50 * np.sin((horas - 8) * np.pi / 12) + np.random.randn(24) * 5
demanda[demanda < 80] = 80 # Asegurar una demanda mínima

# Generación por capas (simplificado)
nuclear = np.full(24, 40)
cogeneracion = np.full(24, 10)
solar = np.maximum(0, 30 * np.sin((horas - 6) * np.pi / 12))
eolica = 15 + 5 * np.sin(horas * np.pi / 6)
gap_inflexible = demanda - (nuclear + cogeneracion + solar + eolica)
hidraulica = np.maximum(0, gap_inflexible * 0.8).clip(0, 25) # Cubre el 80% del gap, con un límite

# Residuos finales
gap_final_deficit = np.maximum(0, demanda - (nuclear + cogeneracion + solar + eolica + hidraulica))
gap_final_excedente = np.minimum(0, demanda - (nuclear + cogeneracion + solar + eolica + hidraulica))

# 2. Preparar el gráfico con Matplotlib
fig, ax = plt.subplots(figsize=(12, 7))

# Capas de generación
labels = ['1. Nuclear', '2. Cogeneración', '3. Eólica + Solar', '4. Hidráulica', '5. Residuos (Déficit)']
colors = ['#B2A8D1', '#CDB5A5', '#A8D1B2', '#A8C5D1', '#D1A8A8']
ax.stackplot(horas, 
             nuclear, 
             cogeneracion, 
             solar + eolica, 
             hidraulica, 
             gap_final_deficit, 
             labels=labels,
             colors=colors,
             alpha=0.8)

# Curva de Demanda
ax.plot(horas, demanda, color='black', linewidth=2.5, linestyle='-', label='Demanda Total')

# Marcar zona de excedente (opcional, pero visualmente potente)
ax.fill_between(horas, demanda, nuclear + cogeneracion + solar + eolica + hidraulica, 
                where=(gap_final_excedente < 0), 
                color='lightgreen', alpha=0.5, interpolate=True, label='Excedente No Gestionado')

# 3. Estilo y etiquetas
ax.set_title('Reconstrucción de la Cobertura de la Demanda por Capas', fontsize=16)
ax.set_xlabel('Hora del Día', fontsize=12)
ax.set_ylabel('Potencia (MW)', fontsize=12)
ax.set_xlim(0, 23)
ax.set_ylim(0, demanda.max() * 1.1)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()


#%%

from matplotlib import rcParams

# # Configurar tipografía
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Crear gráfico
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(niveles, factores, 'b-', linewidth=2, label='Factor de dessalinització')

# Línea horizontal para el mínimo técnico
min_tecnico = 0.18
ax.axhline(y=min_tecnico, color='r', linestyle='--', linewidth=1.5, 
           label='Mínim tècnic/econòmic')

umbral_prealerta = 60
ax.axvline(x=umbral_prealerta, color='orange', linestyle='--', linewidth=1.5, 
           label='Llindar de Prealerta de sequera')

ax.text(umbral_prealerta - 2, 0.5, 'Llindar de Prealerta', 
        color='orange', va='bottom', ha='left', rotation=90, fontsize=11)

# Añadir etiqueta encima de la línea horizontal
ax.text(20, min_tecnico + 0.01, 'Mínim tècnic/econòmic', 
        color='r', va='bottom', ha='right')

# Configurar ejes
ax.set_xlabel('Nivell de Reserves Hídriques (%)', fontsize=12)
ax.set_ylabel('Factor de Dessalinització', fontsize=12)
ax.set_title('Factor de Dessalinització vs. Nivell de Reserves', fontsize=14)
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend(loc='best')

# Mostrar gráfico
plt.tight_layout()
plt.show()

#%%

def factor_capacitat_eolica(
    multiple, 
    cf_actual=0.24, 
    # cf_modern=0.30, #0.34 (amb offshore abundant)
    # p=2.0, 
    # c=0.5 #1/3
    
    cf_modern=0.34,
    p=2, 
    c=1 #1/3    
):
    """
    Calcula el factor de capacitat mitjà del parc eòlic amb funció sigmoide.
    Versió vectoritzada per a arrays de numpy.
    
    multiple=1: Capacitat actual (100%)
    multiple=2: Doble capacitat → factor ≈ 28.5%
    """
    # Vectorització: per a múltiples > 1 calculem T, sinó 0
    multiple = np.asarray(multiple)
    T = np.where(multiple > 1, 
                 (multiple - 1)**p / ((multiple - 1)**p + c), 
                 0)
    return cf_actual + (cf_modern - cf_actual) * T



multiples = np.linspace(1, 6, 100)
factors = factor_capacitat_eolica(multiples)

plt.figure(figsize=(8, 5))
plt.plot(multiples, factors * 100, 'b-', linewidth=2, label='Factor de capacitat')
# plt.axvline(x=1, color='k', linestyle=':', alpha=0.7, label='Capacitat actual')
# plt.axvline(x=2, color='g', linestyle=':', alpha=0.7, label='Doble capacitat')
plt.axhline(y=24, color='r', linestyle='--', alpha=0.5, label='Actual (24%)')
plt.axhline(y=34, color='orange', linestyle='--', alpha=0.5, label='Modern (34%)')
# plt.scatter([2], [28.5], color='g', s=100, zorder=5, label='Objectiu (2x: 28.5%)')

plt.xlabel('Múltiple de capacitat total respecte l\'actual')
plt.ylabel('Factor de capacitat (%)')
plt.title('Evolució del factor de capacitat eòlica amb nova capacitat')
plt.grid(True, alpha=0.3)
plt.legend()

# # Peu d'imatge
# plt.figtext(0.5, 0.02, 
#     "La funció sigmoide modelitza un increment inicial lent, acceleració en la transició i saturació vers tecnologia moderna.\n"
#     "Paràmetres: p=2.0 (forma), c=0.333 (transició 1→2), asímptota=30%",
#     ha='center', va='bottom', fontsize=9, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

#%%


# ------------------------------------------------------------
# 1. Función: derivada media móvil
# ------------------------------------------------------------
def derivada_movil(level: pd.Series, window_days: int) -> pd.Series:
    window_h = window_days * 24
    deriv = np.zeros(len(level))
    values = level.values

    for i in range(len(level)):
        if i >= window_h:
            deriv[i] = (values[i] - values[i - window_h]) / window_h
        else:
            deriv[i] = 0.0

    return pd.Series(deriv, index=level.index)

# ------------------------------------------------------------
# 2. Ventanas mensuales a probar
# ------------------------------------------------------------
windows_days = [30, 45, 60]

derivadas = {
    w: derivada_movil(level_final, w)
    for w in windows_days
}

# ------------------------------------------------------------
# 3. Estadísticos clave (para calibración)
# ------------------------------------------------------------
stats = {}
for w, d in derivadas.items():
    stats[w] = np.percentile(
        d.values,
        [1, 5, 10, 25, 50, 75, 90, 95, 99]
    )

stats_df = pd.DataFrame(
    stats,
    index=["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
)

print("\nDistribución de derivadas (%/hora):")
print(stats_df)

# ------------------------------------------------------------
# 4. Gráfica comparativa (suavizada semanal)
# ------------------------------------------------------------
plt.figure()
for w, d in derivadas.items():
    plt.plot(
        d.rolling(24 * 7).mean(),
        label=f"{w} días"
    )

plt.axhline(0, linestyle="--")
plt.ylabel("Derivada del nivel (%/hora)")
plt.title("Derivada del nivel con ventanas mensuales")
plt.legend()
plt.show()

#%%


# --- 1. CONFIGURACIÓ ---
# Posa aquí la teva sèrie real o simulada de nivells (%)
# Si no la tens carregada, això crea una dummy per provar
t = np.linspace(0, 1000, 8760) # 1 any
level_final = 60 + 10 * np.sin(t/100) + np.random.normal(0, 0.05, 8760) 
# Simulem una baixada forta artificial al final
level_final[-100:] = level_final[-100:] - np.linspace(0, 5, 100)

finestra_hores = 24  # La mateixa que fas servir al bucle
amplitud = 0.5       # Si tanh=1, multipliques per 1.5 (50% extra)

# --- 2. CÀLCUL DE LA DERIVADA (Igual que al teu model) ---
# (current - past) / window
nivell_series = pd.Series(level_final)
derivada = (nivell_series - nivell_series.shift(finestra_hores)) / finestra_hores
derivada = derivada.dropna()

# --- 3. ANÀLISI ESTADÍSTICA ---
# Ens interessen només les baixades (valors negatius)
baixades = derivada[derivada < 0]

# Busquem què es considera una "baixada forta" (Percentil 5%)
# Això ignora outliers extrems, però agafa les baixades significatives
baixada_tipica_greu = np.percentile(baixades, 5) # Ex: -0.05 %/h

print(f"Mitjana de baixades: {baixades.mean():.4f} %/h")
print(f"Baixada 'Greu' (Percentil 5%): {baixada_tipica_greu:.4f} %/h")
print(f"Baixada MÀXIMA absoluta: {baixades.min():.4f} %/h")

# --- 4. CÀLCUL AUTOMÀTIC DE LA SENSIBILITAT ---
# Volem que tanh(sensibilitat * baixada_greu) ≈ 0.95 (saturació)
# Com que baixada és negativa, posem el menys davant
# tanh(2) ≈ 0.96
sensibilitat_recomanada = 2.0 / abs(baixada_tipica_greu)

print("-" * 40)
print(f"SENSIBILITAT RECOMANADA: {sensibilitat_recomanada:.1f}")
print("-" * 40)

# --- 5. VISUALITZACIÓ DE LA FUNCIÓ ---
# Creem un rang de derivades per veure com actua la funció
x_derivades = np.linspace(baixades.min(), 0, 100)

def factor_derivada(d, sens, amp):
    # Nota: el signe '-' és perquè volem resposta positiva a derivada negativa
    return 1 + amp * np.tanh(-d * sens)

y_factors = factor_derivada(x_derivades, sensibilitat_recomanada, amplitud)

plt.figure(figsize=(10, 6))
plt.plot(x_derivades, y_factors, label=f'Sensibilitat = {sensibilitat_recomanada:.1f}')
plt.axvline(baixada_tipica_greu, color='r', linestyle='--', label='Baixada Greu (P5%)')
plt.title("Resposta del Factor de Derivada")
plt.xlabel("Derivada del Nivell (%/hora)")
plt.ylabel("Factor Multiplicador")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


#%%

# -------------------------------
# Parámetros
# -------------------------------
finestra_hores = 24 * 30   # ventana mensual
percentil_ref = 10        # percentil de referencia

# -------------------------------
# Serie de niveles (%)
# -------------------------------
level = np.asarray(results['level_final'], dtype=float)

# -------------------------------
# Cálculo de derivadas (%/hora)
# -------------------------------
derivadas = np.full_like(level, np.nan)

for i in range(finestra_hores, len(level)):
    derivadas[i] = (level[i] - level[i - finestra_hores]) / finestra_hores

# -------------------------------
# Seleccionar solo pendientes negativas
# -------------------------------
deriv_neg = derivadas[derivadas < 0.0]

# Seguridad básica
if len(deriv_neg) == 0:
    raise ValueError("No hay derivadas negativas en la serie")

# -------------------------------
# Escalador DERIV_REF
# -------------------------------
DERIV_REF = abs(np.percentile(deriv_neg, percentil_ref))

print(f"DERIV_REF estimado = {DERIV_REF:.6f} %/hora")

