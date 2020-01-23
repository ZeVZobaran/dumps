# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 00:01:00 2019

Series temporais

Model selection:
    interno (Modelos certo de Johansen, VEC, VAR, Box Jenkins)
        A principio, Box Jenkis vai requere que o arima e SARIMA sejam dados
    externo (Se deve usar VEC, VAR)

Testes:
    cointegração
    normalidade
    sazonalidade
    tendência
    estacionariedade (RU)
    causalidade de Granger entre papeis díspares

Feito:
    Granger, versão coxa de BJ

Fazer:
    Interpretar Granger, versão coxa de BJ


@author: josez
"""
import pandas as pd
import numpy as np
import sklearn as sk
import statsmodels.api as sm
import os
import sys
from comtypes import COMError
sys.path.append(r'C:\Users\josez\Desktop\Python\Estudos')
from basicos import gridsearch_box_jenkins_eviews
from basicos import teste_granger

DIR_BASE = r'C:\Users\josez\Desktop\Economia\Finanças'
BOLSA = os.listdir(r'{}\database\acoes'.format(DIR_BASE))


def bj_bolsa(col):
    global DIR_BASE, BOLSA
    dir_base = DIR_BASE
    bolsa = BOLSA
    lixo = []
    output_dir = r'{0}\testes\box_jenkins'.format(dir_base)
    for papel in bolsa:
        results = {}
        ticker = papel[:-4]
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_pickle(r'{0}\database\acoes\{1}'.format(dir_base, papel))
        if col not in df.columns.levels[1]:
            lixo.append(papel)
            continue
        serie = pd.DataFrame(
                df.loc[:, (ticker, col)],
                )
        index = pd.date_range(
                start=df.index[0],
                freq='B',
                end=df.index[-1]
                )
        serie.columns = serie.columns.droplevel()
        serie = serie.reindex(
                index,
                method='ffill'
                )
        serie.fillna(method='ffill', inplace=True)
        while True:
            try:
                gridsearch = gridsearch_box_jenkins_eviews(
                    serie,
                    arima=(1, 0, 1),
                    sarima=(1, 0, 1, 5),
                    c='c',
                    name=ticker,
                    col=col,
                    output_dir=output_dir
                    )
                results[ticker] = gridsearch
                break
            except COMError:
                os.system('taskkill /F /IM Eviews10.exe')
                print('eviews morto, reiniciando')
            except TypeError as e:
                print('No papel {}'.format(ticker))
                print(e)
                lixo.append(papel)
                break
    lixo = pd.DataFrame(lixo)
    lixo.to_csv(r'{}\lixo\lixo.csv'.format(dir_base))
    return results


def load_df(papel, cols=False):
    ticker = papel[:-4]
    global DIR_BASE
    df = pd.read_pickle(
            r'{0}\database\acoes\{1}'.format(DIR_BASE, papel)
            )
    index = pd.date_range(
            start=df.index[0],
            freq='B',
            end=df.index[-1]
            )
    df_f = df.reindex(index, method='ffill')
    df_f.fillna(method='ffill', inplace=True)
    if cols:
        slicer = pd.IndexSlice
        df_f = df_f.loc[:, slicer[:, cols]]
    return ticker, df_f


def pipeline_granger(cols, maxlag=5):
    # Pensar em como adicionar volume depois
    global BOLSA, DIR_BASE
    bolsa = BOLSA
    dir_base = DIR_BASE
    output_dir_base = r'{0}\testes\granger'.format(dir_base)
    os.makedirs(output_dir_base, exist_ok=True)
    slicer = pd.IndexSlice
    for papel_1 in bolsa:
        ticker_1, df_1 = load_df(papel_1, cols=cols)
        if pd.isnull(df_1).all().all():
            continue
        output_dir = r'{0}\{1}'.format(output_dir_base, ticker_1)
        os.makedirs(output_dir, exist_ok=True)
        for papel_2 in bolsa:
            ticker_2, df_2 = load_df(papel_2, cols=cols)
            if pd.isnull(df_2).all().all():
                continue
            df = pd.concat([df_1, df_2], sort=True, axis=1)
            # A partir daquie operamos sem a coluna de volume
            df_granger = df.loc[:, slicer[:, 'PX_LAST']]
            df_granger = df_granger.droplevel(level=1, axis=1)
            try:
                tg = teste_granger(df_granger, maxlag=maxlag, verbose=False)
            except Exception as e:
                # TODO log e
                continue
            tg_limpo = {}
            for ordem, values in tg.items():
                tg_limpo[ordem] = {}
                for lag, results_e_eqs in values.items():
                    results = results_e_eqs[0]
                    p_vals = {}
                    for tipo_teste, params in results.items():
                        p_vals[tipo_teste] = params[1]
                    tg_limpo[ordem][lag] = p_vals
            for ordem, values in tg_limpo.items():
                report = pd.DataFrame(values)
                report.to_csv(r'{0}\{1}.csv'.format(output_dir, ordem))
    return

# %%


if __name__ == "__main__":
    grids_preço = bj_bolsa(col='PX_LAST')
    grids_volume = bj_bolsa(col='VOLUME')
    pipeline_granger(['PX_LAST', 'VOLUME'])
