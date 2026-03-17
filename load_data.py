# -*- coding: utf-8 -*-
"""
LoadData.py - Mòdul de càrrega i processament de dades per al simulador energètic de Catalunya.

Aquest mòdul encapsula tot el processament de dades necessari per executar simulacions
d'escenaris energètics. Carrega dades de múltiples fonts (SAIH, Gencat, REE, ICAEN)
i les processa per generar sèries sintètiques horàries.

Ús:
    from LoadData import cargar_datos_simulador, hydro_min_for_level
    
    datos = cargar_datos_simulador()
    
    # Accés a les variables:
    df_sintetic = datos.df_sintetic
    demanda = datos.demanda
    ...

Autor: Víctor Garcia
Data de creació: 16/12/2025
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable

from sodapy import Socrata
from scipy.interpolate import interp1d
import cvxpy as cp

from EnerSimFunc import extraer_autoconsumo, insertar_autoconsumo


# =============================================================================
# CONSTANTS DE CONFIGURACIÓ
# =============================================================================

# Dates límit per al rang de simulació
START_DATE_RANGE = '2015-06-03'
END_DATE_RANGE = '2024-12-31'

# Capacitats màximes dels embassaments [hm³]
MAX_CAPACITY_EBRO = 2284
MAX_CAPACITY_INT = 693.0

# Factor de disponibilitat nuclear (pèrdues per serveis auxiliars)
NUCLEAR_AVAILABILITY_FACTOR = 0.973

# Valor d'autoconsum de referència per a desembre 2024 [MW]
AUTOCONSUM_DEC_2024 = 1381.0

# Valor d'autoconsum per insertar en demanda corregida [MW]
AUTOCONSUM_INSERT_VALUE = 1206.7

# Performance ratio per defecte per a autoconsum FV
PR_DEFAULT = 0.75

# Taula de correspondència codis SAIH -> noms embassaments Ebre
CODIGO_A_NOMBRE_EBRO = {
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

# Mapatge noms embassaments Conques Internes
MAPEO_EMBASSAMENTS_INT = {
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


# =============================================================================
# DATACLASS PER AL RESULTAT
# =============================================================================

@dataclass
class DatosSimulador:
    """
    Contenidor amb totes les dades necessàries per al simulador d'escenaris.
    
    Attributes
    ----------
    df_sintetic : pd.DataFrame
        DataFrame horari amb les sèries sintètiques de generació i demanda.
        Columnes: ['Demanda', 'Demanda_w', 'Nuclear', 'Cogeneració', 'Solar', 
                   'Eòlica', 'Solar_w', 'Eòlica_w', 'Cogen_w', 
                   'Hydro_Level_int', 'Hydro_Level_ebro', 'gap']
    
    demanda : pd.Series
        Sèrie horària de demanda elèctrica original [MW].
    
    nuclears_base : pd.DataFrame
        DataFrame horari amb potència de cada reactor nuclear [MW].
        Columnes: ['Asco1', 'Asco2', 'Vandellos2']
    
    cogeneracion_h : pd.Series
        Sèrie horària de generació per cogeneració [MWh].
    
    solar_h : pd.Series
        Sèrie horària de generació solar (FV + termosolar) [MWh].
    
    eolica_h : pd.Series
        Sèrie horària de generació eòlica [MWh].
    
    autoconsum_hourly : pd.Series
        Sèrie horària de potència instal·lada d'autoconsum FV [MW].
    
    potencia : pd.DataFrame
        DataFrame mensual amb potència instal·lada per tecnologia [MW].
    
    df_pct_int_h : pd.Series
        Sèrie horària de fracció d'emmagatzematge (0-1) a Conques Internes.
    
    df_pct_ebre_h : pd.Series
        Sèrie horària de fracció d'emmagatzematge (0-1) a Conca de l'Ebre.
    
    energia_turbinada_mensual_internes : pd.Series
        Sèrie mensual d'energia hidràulica turbinada a Conques Internes [MWh].
    
    energia_turbinada_mensual_ebre : pd.Series
        Sèrie mensual d'energia hidràulica turbinada a Conca de l'Ebre [MWh].
        
    dessalacio_diaria : pd.Series
        Sèrie diària de la dessalinització a les Conques Internes.        

    regeneracio_diaria : pd.Series
        Sèrie diària de la regeneració a l'ERA del Prat.        
        
    """
    df_sintetic: pd.DataFrame
    demanda: pd.Series
    nuclears_base: pd.DataFrame
    cogeneracion_h: pd.Series
    solar_h: pd.Series
    eolica_h: pd.Series
    autoconsum_hourly: pd.Series
    potencia: pd.DataFrame
    df_pct_int_h: pd.Series
    df_pct_ebre_h: pd.Series
    energia_turbinada_mensual_internes: pd.Series
    energia_turbinada_mensual_ebre: pd.Series
    dessalacio_diaria: pd.Series
    regeneracio_diaria: pd.Series


# =============================================================================
# VARIABLE GLOBAL PER A LA FUNCIÓ HYDRO_MIN (es configura durant la càrrega)
# =============================================================================

_f_hydro_min_interpolator = None


def hydro_min_for_level(level_pct: float) -> float:
    """
    Retorna el mínim tècnic hidràulic normalitzat per a un nivell d'emmagatzematge donat.
    
    Aquesta funció utilitza interpolació quadràtica sobre dades històriques del percentil 1%
    de generació hidràulica per cada interval de nivell d'embassament. El resultat representa
    la fracció mínima de generació hidràulica que cal mantenir per restriccions operatives.
    
    Parameters
    ----------
    level_pct : float
        Nivell d'emmagatzematge en percentatge (0-100).
    
    Returns
    -------
    float
        Mínim tècnic hidràulic normalitzat (fracció horària).
    
    Raises
    ------
    RuntimeError
        Si s'invoca abans de carregar les dades amb `cargar_datos_simulador()`.
    
    Notes
    -----
    El nivell es limita internament a l'interval [35, 100]% per evitar extrapolacions
    massa llunyanes. El valor retornat es divideix per 24 per convertir de fracció
    diària a horària.
    """
    global _f_hydro_min_interpolator
    if _f_hydro_min_interpolator is None:
        raise RuntimeError(
            "La funció hydro_min_for_level no està inicialitzada. "
            "Cal executar cargar_datos_simulador() primer."
        )
    lvl = np.clip(level_pct, 35, 100)
    return float(_f_hydro_min_interpolator(lvl) / 24)


# =============================================================================
# FUNCIONS PRIVADES DE CÀRREGA
# =============================================================================

def _cargar_embalses_ebro(filepath: str = 'EmbalsesEbro_diario_2025_08_24.xlsx') -> tuple:
    """
    Carrega dades diàries d'embassaments de la Conca Catalana de l'Ebre (SAIH).
    
    Returns
    -------
    tuple
        (df_volumen_ebre, df_pct_ebre, df_pct_ebre_h)
        - df_volumen_ebre: DataFrame diari amb volums per embassament [hm³]
        - df_pct_ebre: Sèrie diària de fracció total d'emmagatzematge
        - df_pct_ebre_h: Sèrie horària interpolada de fracció d'emmagatzematge
    """
    excel_file = pd.ExcelFile(filepath)
    df_volumen_ebre = pd.DataFrame()
    
    for hoja in excel_file.sheet_names:
        df_hoja = excel_file.parse(hoja)
        
        # Extreure codi del nom del full (ex: "E063_Cavallers" → "063")
        codigo = hoja.split('_')[0][1:4]
        nombre = CODIGO_A_NOMBRE_EBRO.get(codigo, hoja)
        
        if 'FECHA_GRUPO' in df_hoja.columns and 'ACUMULADO (hm³)' in df_hoja.columns:
            df_temp = df_hoja[['FECHA_GRUPO', 'ACUMULADO (hm³)']].copy()
            df_temp['FECHA_GRUPO'] = pd.to_datetime(df_temp['FECHA_GRUPO'], dayfirst=True)
            df_temp = df_temp.set_index('FECHA_GRUPO')
            df_temp.columns = [nombre]
            df_volumen_ebre = pd.concat([df_volumen_ebre, df_temp], axis=1)
    
    # Interpolar dades intermèdies i eliminar NaN
    df_volumen_ebre = df_volumen_ebre.sort_index().interpolate(method='linear', limit_area='inside')
    df_volumen_ebre = df_volumen_ebre.dropna()
    
    # Calcular fracció d'emmagatzematge
    df_pct_ebre = df_volumen_ebre.sum(axis=1) / MAX_CAPACITY_EBRO
    df_pct_ebre.index.name = 'Data'
    
    # Reindexar a freqüència horària i interpolar
    df_pct_ebre_h = df_pct_ebre.resample('h').asfreq().interpolate(method='linear')
    
    return df_volumen_ebre, df_pct_ebre, df_pct_ebre_h


def _cargar_embalses_internes() -> tuple:
    """
    Carrega dades diàries d'embassaments de les Conques Internes Catalanes (Gencat API).
    
    Returns
    -------
    tuple
        (df_volumen_int, df_pct_int, df_pct_int_h)
        - df_volumen_int: DataFrame diari amb volum agregat [hm³]
        - df_pct_int: Sèrie diària de fracció d'emmagatzematge
        - df_pct_int_h: Sèrie horària interpolada de fracció d'emmagatzematge
    """
    # Connexió a Socrata (Transparència Catalunya)
    domain = "analisi.transparenciacatalunya.cat"
    dataset_id = "gn9e-3qhr"
    
    client = Socrata(domain, None)
    records = client.get(dataset_id, limit=200000)
    df = pd.DataFrame.from_records(records)
    
    # Processar dates i columnes numèriques
    df['dia'] = pd.to_datetime(df['dia'])
    numeric_columns = ['nivell_absolut', 'percentatge_volum_embassat', 'volum_embassat']
    df[numeric_columns] = df[numeric_columns].astype(float)
    df.columns = ['Dia', 'Embassament', 'Nivell absolut (msnm)', 
                  'Percentatge volum embassat (%)', 'Volum embassat (hm3)']
    df['Dia'] = pd.to_datetime(df['Dia']).dt.strftime('%Y-%m-%d')
    df['Dia'] = pd.to_datetime(df['Dia'])
    df.set_index('Dia', inplace=True)
    
    # Aplicar mapatge de noms
    df['Embassament'] = df['Embassament'].map(MAPEO_EMBASSAMENTS_INT)
    
    # Agregar per dia
    df_volumen_int = df.groupby('Dia').agg({'Volum embassat (hm3)': 'sum'})
    df_volumen_int.columns = ['']
    
    # Calcular fracció d'emmagatzematge
    df_pct_int = df_volumen_int / MAX_CAPACITY_INT
    
    # Reindexar a freqüència horària
    df_pct_int_h = df_pct_int.resample('h').asfreq().interpolate(method='linear')
    
    return df_volumen_int, df_pct_int, df_pct_int_h


def _configurar_hydro_min_interpolator():
    """
    Configura l'interpolador per a la funció de mínim tècnic hidràulic.
    
    Utilitza dades històriques de generació hidràulica peninsular per calcular
    el percentil 1% de generació per cada interval de nivell d'embassament.
    """
    global _f_hydro_min_interpolator
    
    generacion_v2 = pd.read_excel("generacion_v3.xlsx", index_col="fecha", parse_dates=True)
    
    # Bins de nivell d'emmagatzematge (cada 10%)
    bins = np.arange(0, 101, 10)
    labels = (bins[:-1] + bins[1:]) / 2
    
    df_hydro = generacion_v2[['Hydro_Level', 'Hidráulica']].dropna()
    df_hydro['Hidráulica'] = df_hydro['Hidráulica'] / 17095  # Normalitzar per capacitat peninsular
    df_hydro['bin'] = pd.cut(df_hydro['Hydro_Level'], bins=bins, labels=labels)
    
    # Percentil 1% en cada bin
    min_por_bin = df_hydro.groupby('bin', observed=False)['Hidráulica'].quantile(0.01).dropna()
    x = min_por_bin.index.astype(float)
    y = min_por_bin.values
    
    # Interpolador quadràtic amb extrapolació
    _f_hydro_min_interpolator = interp1d(
        x, y,
        kind='quadratic',
        fill_value='extrapolate',
        assume_sorted=True
    )


def _cargar_nucleares(filepath: str = 'nuclears.xlsx') -> tuple:
    """
    Carrega dades de generació nuclear per reactor.
    
    Returns
    -------
    tuple
        (nuclears_base, nuclears)
        - nuclears_base: DataFrame horari amb potència per reactor [MW]
        - nuclears: Sèrie horària amb potència total [MW]
    """
    # Carregar cada full (un per reactor)
    Asco1 = pd.read_excel(filepath, sheet_name=0, index_col=0, parse_dates=True).astype(float)
    Asco2 = pd.read_excel(filepath, sheet_name=1, index_col=0, parse_dates=True).astype(float)
    Vandellos2 = pd.read_excel(filepath, sheet_name=2, index_col=0, parse_dates=True).astype(float)
    
    # Eliminar duplicats
    Asco1 = Asco1[~Asco1.index.duplicated(keep='first')]
    Asco2 = Asco2[~Asco2.index.duplicated(keep='first')]
    Vandellos2 = Vandellos2[~Vandellos2.index.duplicated(keep='first')]
    
    # Unir i interpolar
    nuclears_df = pd.concat([Asco1, Asco2, Vandellos2], axis=1, join='outer')
    nuclears_df.columns = ['Asco1', 'Asco2', 'Vandellos2']
    nuclears_df = nuclears_df.interpolate(method='linear')
    
    # Convertir a horari i aplicar factor de disponibilitat
    nuclears_base = nuclears_df.resample('h').ffill() * NUCLEAR_AVAILABILITY_FACTOR
    nuclears = nuclears_df.sum(axis=1).resample('h').ffill() * NUCLEAR_AVAILABILITY_FACTOR
    
    return nuclears_base, nuclears


def _cargar_demanda() -> pd.Series:
    """
    Carrega i processa les dades de demanda elèctrica de Catalunya.
    
    Returns
    -------
    pd.Series
        Sèrie horària de demanda [MW].
    """
    from load_data_enercat import load_and_process_electricity_demand_data
    
    demanda = load_and_process_electricity_demand_data('DemandaMWh_20250618.csv', freq='hourly')
    demanda = demanda[demanda > 0.01]
    demanda = demanda.interpolate(method='linear')
    demanda = demanda.squeeze().asfreq('h')
    
    return demanda


def _cargar_potencia_instalada() -> pd.DataFrame:
    """
    Carrega dades de potència instal·lada per tecnologia a Catalunya.
    
    Returns
    -------
    pd.DataFrame
        DataFrame mensual amb potència per tecnologia [MW].
    """
    potencia = pd.read_excel('potencia_cat.xlsx', index_col='Fecha', decimal=',')
    potencia.columns = [
        'Cicles', 'Cogeneració', 'Eòlica', 'Hidràulica', 'Nuclear', 
        'AltresRen', 'Total', 'ResidusNR', 'ResidusR', 'Fotovoltaica', 'Termosolar'
    ]
    
    # Carregar i afegir autoconsum
    fotovoltaica_mes_cum = pd.read_csv('autoconsum.csv', index_col='I_DAT_PEM')
    fotovoltaica_mes_cum.index = pd.to_datetime(fotovoltaica_mes_cum.index)
    fotovoltaica_mes_cum.columns = ['MW']
    
    potencia['Autoconsum'] = fotovoltaica_mes_cum
    potencia.loc['2024-12-31', 'Autoconsum'] = AUTOCONSUM_DEC_2024
    
    # Forçar NaN als mesos amb dades mancants i interpolar
    potencia.loc['2024-07-31':'2024-11-30', 'Autoconsum'] = np.nan
    potencia['Autoconsum'] = potencia['Autoconsum'].interpolate(method='linear')
    
    return potencia


def _cargar_generacion_cat() -> pd.DataFrame:
    """
    Carrega dades de generació mensual de Catalunya.
    
    Returns
    -------
    pd.DataFrame
        DataFrame mensual amb generació per tecnologia [MWh].
    """
    generacio = pd.read_excel('generacio_cat.xlsx', decimal=',', index_col='fecha')
    generacio.drop(['Fuel + Gas'], axis=1, inplace=True)
    generacio.columns = [
        'Cicles', 'Cogeneració', 'Eòlica', 'Total', 'Hidràulica', 
        'Nuclear', 'AltresRen', 'ResidusNR', 'ResidusR', 'Fotovoltaica', 'Termosolar'
    ]
    return generacio


def _cargar_generacion_spain() -> pd.DataFrame:
    """
    Carrega dades de generació horària peninsular (per a perfils).
    
    Returns
    -------
    pd.DataFrame
        DataFrame horari amb generació i potència instal·lada peninsular.
    """
    generacion = pd.read_excel('GeneracionSpain_Horas4.xlsx', index_col='Fecha')
    
    # Reescalar a potència actual
    generacion['Eólica'] = generacion.Eólica0 * generacion.PotenciaEol.iloc[-1] / generacion.PotenciaEol
    generacion['Fotovoltaica'] = generacion.FV0 * generacion.PotenciaFV.iloc[-1] / generacion.PotenciaFV
    generacion['Cogeneracion'] = generacion.Cogeneración * generacion.PotenciaCog.iloc[-1] / generacion.PotenciaCog
    
    # Corregir valors negatius i zeros nocturns per FV
    generacion.loc[generacion.Termosolar < 0, 'Termosolar'] = 0
    generacion.loc[generacion.Fotovoltaica < 0, 'Fotovoltaica'] = 0
    
    # Forçar zero en hores nocturnes segons el mes (alba/posta)
    _aplicar_mascara_nocturna_fv(generacion)
    
    return generacion


def _aplicar_mascara_nocturna_fv(generacion: pd.DataFrame) -> None:
    """Aplica màscara de zeros en hores nocturnes per a generació FV segons el mes."""
    mascaras = [
        (10, 6, 18), (11, 6, 19), (12, 6, 19), (1, 6, 19), (2, 6, 19),
        (3, 5, 19), (4, 4, 20), (5, 4, 21), (6, 4, 21), (7, 4, 21), (8, 4, 21), (9, 5, 20)
    ]
    for mes, hora_ini, hora_fin in mascaras:
        if mes == 10:
            mask = (generacion.index.month == mes) & ((generacion.index.hour <= hora_ini) | (generacion.index.hour >= hora_fin))
        elif mes >= 11 or mes <= 2:
            mask = (generacion.index.month == mes) & ((generacion.index.hour <= hora_ini) | (generacion.index.hour >= hora_fin))
        else:
            mask = (generacion.index.month == mes) & ((generacion.index.hour <= hora_ini) | (generacion.index.hour >= hora_fin))
        generacion.loc[mask, 'Fotovoltaica'] = 0


def _calcular_demanda_corregida(demanda: pd.Series) -> pd.Series:
    """
    Calcula la demanda corregida per tendència (normalitzada a nivell 2024).
    
    Parameters
    ----------
    demanda : pd.Series
        Sèrie horària de demanda original.
    
    Returns
    -------
    pd.Series
        Sèrie horària de demanda corregida.
    """
    ventana_anual = 365 * 24
    tendencia = demanda.rolling(window=ventana_anual, center=True, min_periods=1).mean()
    tendencia_2024 = tendencia.loc['2024'].mean()
    
    dfdem = pd.DataFrame({'demanda': demanda, 'tendencia': tendencia})
    dfdem['año'] = dfdem.index.year
    
    nivel_por_año = dfdem.groupby('año')['tendencia'].mean()
    factor_por_año = tendencia_2024 / nivel_por_año
    factor_por_año[2024] = 1.0
    
    dfdem['factor'] = dfdem['año'].map(factor_por_año)
    demanda_w = demanda * dfdem['factor']
    
    return demanda_w


def _generar_serie_solar(potencia: pd.DataFrame, generacio: pd.DataFrame) -> tuple:
    """
    Genera sèries horàries de generació solar sintètica.
    
    Utilitza perfils de radiació de Copernicus ERA5 i correcció mensual
    amb dades reals de generació.
    
    Returns
    -------
    tuple
        (solar_h, solar_h_w) - sèries sense i amb correcció de tendència
    """
    # Carregar perfil de radiació normalitzat
    sun_cat = pd.read_excel("sun_cat.xlsx", index_col=0, parse_dates=True)
    sun_cat = (sun_cat - sun_cat.min()) / (sun_cat.max() - sun_cat.min())
    sun_cat = sun_cat.sort_index()
    
    # Potència horària instal·lada
    potencia_horaria = (
        potencia.Fotovoltaica.resample('h').ffill()[START_DATE_RANGE:END_DATE_RANGE] + 
        potencia.Termosolar.resample('h').ffill()[START_DATE_RANGE:END_DATE_RANGE]
    )
    
    # Alinear perfil de radiació
    sun_cat_filtrado = sun_cat['ssrd'][START_DATE_RANGE:END_DATE_RANGE]
    sun_cat_filtrado = sun_cat_filtrado[~sun_cat_filtrado.index.duplicated(keep='last')]
    sun_cat_aligned = sun_cat_filtrado.reindex(potencia_horaria.index)
    
    # Generació inicial (limitada per potència instal·lada)
    solar_h_inicial = (potencia_horaria * sun_cat_aligned).dropna()
    solar_h_inicial = solar_h_inicial.clip(upper=potencia_horaria)
    
    # Aplicar mescla de perfils FV fixa/monoeix (80% monoeix)
    solar_h_inicial = _mezcla_perfiles_fv(solar_h_inicial, frac_monoeje=0.8, ganho_rel=0.20)
    
    # Correcció mensual amb dades reals
    corrector_mensual = (
        (generacio.Fotovoltaica + generacio.Termosolar) / 
        solar_h_inicial.resample('ME').sum()
    ).dropna()
    
    solar_h_corregida = solar_h_inicial.copy()
    for mes in corrector_mensual.index:
        mask_mes = solar_h_inicial.index.to_period('M') == mes.to_period('M')
        solar_h_corregida.loc[mask_mes] *= corrector_mensual.loc[mes]
        solar_h_corregida.loc[mask_mes] = solar_h_corregida.loc[mask_mes].clip(
            upper=potencia_horaria.loc[mask_mes]
        )
    
    solar_h = solar_h_corregida.round(0)
    
    # Versió corregida per tendència de potència
    factor_w = (
        (potencia.Fotovoltaica.iloc[-1] + potencia.Termosolar.iloc[-1]) /
        (potencia.Fotovoltaica + potencia.Termosolar)
    ).resample('h').ffill()
    solar_h_w = (solar_h * factor_w).dropna()
    
    return solar_h, solar_h_w


def _fijo_a_monoeje(P_fijo: pd.Series, ganho_rel: float = 0.20) -> pd.Series:
    """
    Converteix un perfil horari de FV fixa a monoeix N-S (seguiment 1 eix).
    
    Parameters
    ----------
    P_fijo : pd.Series
        Perfil horari de generació FV fixa.
    ganho_rel : float
        Guany anual del monoeix respecte a fixa (ex: 0.20 = +20%).
    
    Returns
    -------
    pd.Series
        Perfil horari de generació FV amb seguiment monoeix.
    """
    horas_desde_mediodia = (P_fijo.index.hour + P_fijo.index.minute / 60) - 12
    
    # Perfil de suavitzat: aixeca espatlles i aplana pic
    factor_forma = (
        1 + 0.12 * (1 - np.exp(-(np.abs(horas_desde_mediodia) / 2.5) ** 1.6))
        - 0.08 * np.exp(-(np.abs(horas_desde_mediodia) / 1.2) ** 2)
    )
    
    P_tilted = P_fijo * factor_forma
    
    # Escalat d'energia anual al guany relatiu
    energia_fijo = P_fijo.sum()
    energia_tilted = P_tilted.sum()
    escala = (energia_fijo * (1 + ganho_rel)) / energia_tilted
    P_1T = P_tilted * escala
    
    # Forçar zero de nit
    P_1T[P_fijo <= 0] = 0
    return P_1T


def _mezcla_perfiles_fv(P_fijo: pd.Series, frac_monoeje: float = 0.5, 
                        ganho_rel: float = 0.20) -> pd.Series:
    """Genera mescla ponderada de perfils FV fixa i monoeix."""
    P_1T = _fijo_a_monoeje(P_fijo, ganho_rel)
    return frac_monoeje * P_1T + (1 - frac_monoeje) * P_fijo


def _generar_serie_eolica(potencia: pd.DataFrame, generacio: pd.DataFrame) -> tuple:
    """
    Genera sèries horàries de generació eòlica sintètica.
    
    Utilitza perfils de velocitat de vent de Copernicus ERA5.
    
    Returns
    -------
    tuple
        (eolica_h, eolica_h_w) - sèries sense i amb correcció de tendència
    """
    # Carregar perfil de vent normalitzat
    wind_cat = pd.read_excel("wind_cat.xlsx", index_col=0, parse_dates=True)['speed']
    wind_cat = (wind_cat - wind_cat.min()) / (wind_cat.max() - wind_cat.min())
    
    # Potència horària instal·lada
    potencia_horaria = potencia.Eòlica.resample('h').ffill()[START_DATE_RANGE:END_DATE_RANGE]
    
    # Alinear perfil de vent
    wind_cat_filtrado = wind_cat[START_DATE_RANGE:END_DATE_RANGE]
    wind_cat_aligned = wind_cat_filtrado.reindex(potencia_horaria.index)
    
    # Generació inicial (limitada per potència instal·lada)
    eolica_h_inicial = (potencia_horaria * wind_cat_aligned).dropna()
    eolica_h_inicial = eolica_h_inicial.clip(upper=potencia_horaria)
    
    # Correcció mensual amb dades reals
    corrector_mensual = (generacio.Eòlica / eolica_h_inicial.resample('ME').sum()).dropna()
    
    eolica_h_corregida = eolica_h_inicial.copy()
    for mes in corrector_mensual.index:
        mask_mes = eolica_h_inicial.index.to_period('M') == mes.to_period('M')
        eolica_h_corregida.loc[mask_mes] *= corrector_mensual.loc[mes]
        eolica_h_corregida.loc[mask_mes] = eolica_h_corregida.loc[mask_mes].clip(
            upper=potencia_horaria.loc[mask_mes]
        )
    
    eolica_h = eolica_h_corregida.round(0)
    
    # Versió corregida per tendència de potència
    factor_w = (potencia.Eòlica.iloc[-1] / potencia.Eòlica).resample('h').ffill()
    eolica_h_w = eolica_h * factor_w
    
    return eolica_h, eolica_h_w


def _generar_serie_cogeneracion(potencia: pd.DataFrame, generacio: pd.DataFrame,
                                 generacion_spain: pd.DataFrame) -> tuple:
    """
    Genera sèries horàries de generació per cogeneració.
    
    Returns
    -------
    tuple
        (cogeneracion_h, cogeneracion_h_w) - sèries sense i amb correcció de tendència
    """
    # Factor de càrrega peninsular com a proxy
    factorCog = (
        generacion_spain.Cogeneración / generacion_spain.PotenciaCog
    )[START_DATE_RANGE:END_DATE_RANGE]
    
    # Generació inicial
    cogeneracion_h_inicial = (
        potencia.Cogeneració.resample('h').ffill()[START_DATE_RANGE:END_DATE_RANGE] * factorCog
    ).dropna()
    
    # Correcció mensual amb dades reals (vectoritzat)
    corrector_mensual = (
        generacio.Cogeneració / cogeneracion_h_inicial.resample('ME').sum()
    ).dropna()
    corrector_mensual.index = corrector_mensual.index.to_period('M')
    
    period_index = cogeneracion_h_inicial.index.to_period('M')
    factor_alineado = corrector_mensual.loc[period_index]
    factor_alineado.index = cogeneracion_h_inicial.index
    
    cogeneracion_h = cogeneracion_h_inicial * factor_alineado
    
    # Versió corregida per tendència (eliminació de tendència històrica)
    factor_w = (potencia.Cogeneració.iloc[-1] / potencia.Cogeneració).resample('h').ffill()
    cogeneracion_h_w = cogeneracion_h * factor_w
    
    # Correcció addicional: eliminar tendència temporal amb mitjana mòbil
    tendencia = cogeneracion_h_w.rolling(window=365 * 24, center=True, min_periods=1).mean()
    mask_ref = cogeneracion_h_w.index.year >= 2024
    nivel_ref = tendencia.loc[mask_ref].mean()
    
    factor = pd.Series(1.0, index=cogeneracion_h_w.index)
    factor.loc[~mask_ref] = nivel_ref / tendencia.loc[~mask_ref]
    
    cogeneracion_h_w = cogeneracion_h_w * factor
    cogeneracion_h_w = cogeneracion_h_w.clip(upper=potencia.Cogeneració.iloc[-1])
    
    return cogeneracion_h, cogeneracion_h_w


def _desagregar_hidraulica_por_cuencas(energia_turbinada_mensual: pd.Series,
                                        df_pct_int: pd.Series,
                                        df_pct_ebre: pd.Series) -> tuple:
    """
    Desagrega la generació hidràulica mensual entre les dues conques.
    
    Utilitza ponderació basada en capacitat relativa i nivell d'emmagatzematge.
    
    Parameters
    ----------
    energia_turbinada_mensual : pd.Series
        Sèrie mensual d'energia hidràulica total [MWh].
    df_pct_int : pd.Series
        Sèrie de fracció d'emmagatzematge Conques Internes.
    df_pct_ebre : pd.Series
        Sèrie de fracció d'emmagatzematge Conca de l'Ebre.
    
    Returns
    -------
    tuple
        (energia_turbinada_mensual_internes, energia_turbinada_mensual_ebre)
    """
    # Paràmetres de capacitat relativa
    k_int = 0.1   # Internes tenen el 10% de la capacitat de l'Ebre
    k_ebro = 1.0
    
    # Resamplear a mensual (últim valor de cada mes)
    nivel_int_m = df_pct_int.resample('ME').last().squeeze()
    nivel_ebro_m = df_pct_ebre.resample('ME').last().squeeze()
    
    # Construir DataFrame conjunt
    df = pd.DataFrame({
        'Energia': energia_turbinada_mensual,
        'Nivel_int': nivel_int_m,
        'Nivel_ebro': nivel_ebro_m
    }).dropna()
    
    # Calcular pesos segons capacitat relativa * nivell d'embassament
    df['w_int'] = k_int * df['Nivel_int']
    df['w_ebro'] = k_ebro * df['Nivel_ebro']
    
    # Fraccions de distribució
    df['f_int'] = df['w_int'] / (df['w_int'] + df['w_ebro'])
    df['f_ebro'] = 1 - df['f_int']
    
    # Desagregar energia
    energia_turbinada_mensual_internes = df['Energia'] * df['f_int']
    energia_turbinada_mensual_ebre = df['Energia'] * df['f_ebro']
    
    return energia_turbinada_mensual_internes, energia_turbinada_mensual_ebre


# =============================================================================
# FUNCIÓ PRINCIPAL DE CÀRREGA
# =============================================================================

def cargar_datos_simulador(verbose: bool = True) -> DatosSimulador:
    """
    Carrega i processa totes les dades necessàries per al simulador d'escenaris.
    
    Aquesta funció encapsula tot el pipeline de càrrega i processament de dades,
    retornant únicament les variables necessàries per executar simulacions.
    
    Parameters
    ----------
    verbose : bool, optional
        Si True, mostra missatges de progrés. Per defecte True.
    
    Returns
    -------
    DatosSimulador
        Objecte dataclass amb totes les dades necessàries per al simulador.
    
    Examples
    --------
    >>> from LoadData import cargar_datos_simulador, hydro_min_for_level
    >>> datos = cargar_datos_simulador()
    >>> 
    >>> # Configuració del simulador
    >>> base_config = {
    ...     'df_demanda': datos.demanda,
    ...     'df_nucleares_base': datos.nuclears_base,
    ...     'df_cogeneracion': datos.cogeneracion_h,
    ...     'df_solar': datos.solar_h,
    ...     'df_eolica': datos.eolica_h,
    ...     'df_autoconsum': datos.autoconsum_hourly,
    ...     'df_potencia_historica': datos.potencia,
    ...     'df_capacidad_internes': datos.df_pct_int_h,
    ...     'df_capacidad_ebre': datos.df_pct_ebre_h,
    ...     'energia_turbinada_mensual_internes': datos.energia_turbinada_mensual_internes,
    ...     'energia_turbinada_mensual_ebre': datos.energia_turbinada_mensual_ebre,
    ... }
    """
    if verbose:
        print("=" * 60)
        print("CARREGANT DADES PER AL SIMULADOR D'ESCENARIS")
        print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 1. DADES D'EMBASSAMENTS
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[1/10] Carregant dades d'embassaments Conca de l'Ebre (SAIH)...")
    df_volumen_ebre, df_pct_ebre, df_pct_ebre_h = _cargar_embalses_ebro()
    
    if verbose:
        print("[2/10] Carregant dades d'embassaments Conques Internes (Gencat API)...")
    df_volumen_int, df_pct_int, df_pct_int_h = _cargar_embalses_internes()
    
    # -------------------------------------------------------------------------
    # 2. CONFIGURAR INTERPOLADOR DE MÍNIM TÈCNIC HIDRÀULIC
    # -------------------------------------------------------------------------
    if verbose:
        print("[3/10] Configurant funció de mínim tècnic hidràulic...")
    _configurar_hydro_min_interpolator()
    
    # -------------------------------------------------------------------------
    # 3. DADES DE GENERACIÓ I POTÈNCIA
    # -------------------------------------------------------------------------
    if verbose:
        print("[4/10] Carregant dades de generació i potència instal·lada...")
    
    generacio = _cargar_generacion_cat()
    generacion_spain = _cargar_generacion_spain()
    potencia = _cargar_potencia_instalada()
    nuclears_base, nuclears = _cargar_nucleares()
    demanda = _cargar_demanda()
    
    # Carregar energia hidràulica mensual
    energia_turbinada_mensual = pd.read_excel('generacio_cat.xlsx', decimal=',')
    energia_turbinada_mensual.set_index("fecha", inplace=True)
    energia_turbinada_mensual = energia_turbinada_mensual.Hidráulica
    
    # -------------------------------------------------------------------------
    # 4. CONSTRUIR DATAFRAME SINTÈTIC INICIAL
    # -------------------------------------------------------------------------
    if verbose:
        print("[5/10] Construint sèries sintètiques de generació...")
    
    df_sintetic = pd.concat(
        (demanda['2013-01-01':'2025-01-01'], nuclears['2013-01-01':'2025-01-01']),
        axis=1
    )
    df_sintetic.columns = ['Demanda', 'Nuclear']
    
    # -------------------------------------------------------------------------
    # 5. GENERAR SÈRIES HORÀRIES DE GENERACIÓ
    # -------------------------------------------------------------------------
    solar_h, solar_h_w = _generar_serie_solar(potencia, generacio)
    eolica_h, eolica_h_w = _generar_serie_eolica(potencia, generacio)
    cogeneracion_h, cogeneracion_h_w = _generar_serie_cogeneracion(
        potencia, generacio, generacion_spain
    )
    
    # -------------------------------------------------------------------------
    # 6. PROCESSAR DEMANDA AMB AUTOCONSUM
    # -------------------------------------------------------------------------
    if verbose:
        print("[6/10] Processant demanda i autoconsum...")
    
    autoconsum_hourly = potencia.Autoconsum.resample('h').interpolate('linear')
    
    demanda_w, _ = extraer_autoconsumo(
        df_sintetic.Demanda,
        solar_h_w[START_DATE_RANGE:END_DATE_RANGE],
        autoconsum_hourly,
        pr=PR_DEFAULT
    )
    demanda_w, _ = insertar_autoconsumo(
        demanda_w,
        solar_h_w[START_DATE_RANGE:END_DATE_RANGE],
        AUTOCONSUM_INSERT_VALUE,
        pr=PR_DEFAULT
    )
    
    df_sintetic['Demanda_w'] = demanda_w
    df_sintetic = df_sintetic[['Demanda', 'Demanda_w', 'Nuclear']]
    
    # -------------------------------------------------------------------------
    # 7. ASSEMBLAR DATAFRAME SINTÈTIC COMPLET
    # -------------------------------------------------------------------------
    if verbose:
        print("[7/10] Assemblant DataFrame sintètic complet...")
    
    df_sintetic = pd.concat([
        df_sintetic[START_DATE_RANGE:END_DATE_RANGE],
        cogeneracion_h[:END_DATE_RANGE],
        solar_h[:END_DATE_RANGE],
        eolica_h[:END_DATE_RANGE],
        solar_h_w[:END_DATE_RANGE],
        eolica_h_w[:END_DATE_RANGE],
        cogeneracion_h_w[:END_DATE_RANGE]
    ], axis=1)
    df_sintetic.columns = [
        'Demanda', 'Demanda_w', 'Nuclear', 'Cogeneració', 'Solar', 'Eòlica',
        'Solar_w', 'Eòlica_w', 'Cogen_w'
    ]
    df_sintetic = df_sintetic.dropna()
    
    # Afegir nivells d'embassament
    hydro_hourly_int = df_pct_int_h
    hydro_hourly_ebre = df_pct_ebre_h
    
    df_sintetic['Hydro_Level_int'] = 100 * hydro_hourly_int.reindex(
        df_sintetic.index, method='ffill'
    )
    df_sintetic['Hydro_Level_ebro'] = 100 * hydro_hourly_ebre.reindex(
        df_sintetic.index, method='ffill'
    )
    
    # Calcular gap (demanda no coberta per renovables i nuclear)
    df_sintetic['gap'] = (
        df_sintetic['Demanda_w'] - df_sintetic['Nuclear'] - 
        df_sintetic['Solar_w'] - df_sintetic['Eòlica_w'] - df_sintetic['Cogen_w']
    )
    
    # -------------------------------------------------------------------------
    # 8. DESAGREGAR HIDRÀULICA PER CONQUES
    # -------------------------------------------------------------------------
    if verbose:
        print("[8/10] Desagregant generació hidràulica per conques...")
    
    energia_turbinada_mensual_internes, energia_turbinada_mensual_ebre = \
        _desagregar_hidraulica_por_cuencas(
            energia_turbinada_mensual, df_pct_int, df_pct_ebre
        )
        
 
    # -------------------------------------------------------------------------
    # 9. CARREGAR DADES DE DESSALINITZACIÓ HISTÒRICA
    # -------------------------------------------------------------------------

    if verbose:
        print("[9/10] Carregant dades de dessalinització...")

    dessalacio = pd.read_csv('C:/Users/tirki/Dropbox/Trabajos/Energía/dessalacio_20250723.csv')
    dessalacio = dessalacio.groupby('Dia').sum()['Volum (hm3)']
    dessalacio.index = pd.to_datetime(dessalacio.index, dayfirst=True)
    dessalacio_diaria = dessalacio.resample('D').sum()

    # -------------------------------------------------------------------------
    # 10. CARREGAR I ESTIMAR DADES DE REGENERACIÓ HISTÒRICA
    # -------------------------------------------------------------------------
    if verbose:
        print("[10/10] Carregant i processant dades de regeneració...")
    
    # Datos del proxy
    df_proxy = pd.read_csv("regeneracion_diaria.csv", parse_dates=["Dia"])
    df_proxy = df_proxy.set_index("Dia").sort_index()
    df_proxy.columns = ["Volum"]
    # Calcular mediana móvil (usamos 60 días como compromiso)
    df_proxy["proxy"] = df_proxy["Volum"].rolling(60, center=True, min_periods=1).median()
    # Rellenar NaN en los extremos con el valor más cercano
    df_proxy["proxy"] = df_proxy["proxy"].ffill().bfill()
    
    # Volúmenes anuales objetivo (hm3/año)
    years = np.arange(2016, 2025)
    volumes = np.array([0.01, 0.03, 0.06, 8.1, 9.2, 32.9, 48.3, 54.6, 39.9])
    serie_anual = pd.Series(volumes, index=years)
    # Eje temporal diario completo
    dates = pd.date_range("2016-01-01", "2024-12-31", freq="D")
    n_days = len(dates)
    
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
    problem.solve(solver=cp.CLARABEL, verbose=False)
    
    dates = pd.date_range("2016-01-01", "2024-12-31", freq="D")
    regeneracio_diaria = pd.Series(x.value, index=dates, name="Regeneracio_hm3_dia")
    
    # -------------------------------------------------------------------------
    # RETORNAR RESULTAT
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("CÀRREGA COMPLETADA")
        print(f"  - Rang temporal: {df_sintetic.index.min()} → {df_sintetic.index.max()}")
        print(f"  - Registres horaris: {len(df_sintetic):,}")
        print("=" * 60)
    
    return DatosSimulador(
        df_sintetic=df_sintetic,
        demanda=demanda,
        nuclears_base=nuclears_base,
        cogeneracion_h=cogeneracion_h,
        solar_h=solar_h,
        eolica_h=eolica_h,
        autoconsum_hourly=autoconsum_hourly,
        potencia=potencia,
        df_pct_int_h=df_pct_int_h,
        df_pct_ebre_h=df_pct_ebre_h,
        energia_turbinada_mensual_internes=energia_turbinada_mensual_internes,
        energia_turbinada_mensual_ebre=energia_turbinada_mensual_ebre,
        dessalacio_diaria=dessalacio_diaria,
        regeneracio_diaria=regeneracio_diaria
    )


# =============================================================================
# EXECUCIÓ DIRECTA (per a testing)
# =============================================================================

if __name__ == "__main__":
    # Test de càrrega
    datos = cargar_datos_simulador(verbose=True)
    
    print("\n[TEST] Verificant variables carregades:")
    print(f"  - df_sintetic shape: {datos.df_sintetic.shape}")
    print(f"  - df_sintetic columnes: {list(datos.df_sintetic.columns)}")
    print(f"  - demanda shape: {datos.demanda.shape}")
    print(f"  - nuclears_base shape: {datos.nuclears_base.shape}")
    print(f"  - potencia shape: {datos.potencia.shape}")
    print(f"  - df_pct_int_h shape: {datos.df_pct_int_h.shape}")
    print(f"  - df_pct_ebre_h shape: {datos.df_pct_ebre_h.shape}")
    print(f"  - energia_turbinada_mensual_internes shape: {datos.energia_turbinada_mensual_internes.shape}")
    print(f"  - energia_turbinada_mensual_ebre shape: {datos.energia_turbinada_mensual_ebre.shape}")
    print(f"  - dessalacio_diaria shape: {datos.dessalacio_diaria.shape}")
    print(f"  - regeneracio_diaria shape: {datos.regeneracio_diaria.shape}")

    
    print("\n[TEST] Prova de hydro_min_for_level:")
    for level in [40, 60, 80, 100]:
        print(f"  - Nivell {level}%: {hydro_min_for_level(level):.6f}")
