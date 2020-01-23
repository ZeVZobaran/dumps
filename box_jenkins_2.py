# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:29:42 2019

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

Ferramentas
    normalização
    vizualisação
    backtest
    previsão
    dessazonalização

@author: josez
"""
# %%
import pandas as pd
import itertools
import statsmodels.api as sm


def box_jenkins_report(y, p, d, q, P, D, Q, s, return_results=False, trend=''):
    pdq = list(itertools.product(range(p + 1), range(d + 1), range(q + 1)))
    s_pdq_constr = list(
            itertools.product(range(P + 1), range(D + 1), range(Q + 1))
            )
    s_pdq = []
    for i in s_pdq_constr:
        params = list(i)
        params.append(s)
        s_pdq.append(tuple(params))

    # gerando esqueleto do df de saida
    index = []
    for param in pdq:
        for s_param in s_pdq:
            index.append((param, s_param))
    output_df = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(index),
            columns=[
                    'convergência', 'condição de  estacionariedade',
                    'condição de  invertibilidade', 'SE of reg', 'AIC', 'SC',
                    'HQ', 'P[Q(9)]', 'P[Q(12)]', 'par. não-signif. (10%)'
                      ]
            )

    if return_results:
        results_dict = {}
    # Rodando os modelos
    for param in pdq:
        for s_param in s_pdq:
            try:
                param = (1, 0, 1)
                s_param = (1, 0, 1, 12)
                y = y_diff_menos_3
                mod = sm.tsa.statespace.SARIMAX(
                        y,
                        order=param,
                        seasonal_order=s_param,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        trend=trend
                        )
                results = mod.fit(maxiter=1000)
                results.summary()
                results.aic
                output_df.loc[(param, s_param), :] = [
                        True, True, True, 'Fazer', results.aic, 'fazer',
                        results.hqic, 'fazer', 'fazer', 'fazer']
                results.summary()
                if return_results:
                    results_dict[str(param, s_param)] = results
            except Exception:
                continue
    if return_results:
        output = {
                'results': results_dict,
                'report': output_df
                }
        return output
    else:
        return output_df

# %%


if __name__ == '__main__':
    y = pd.read_excel(
            r'C:\Users\josez\Desktop\Economia\Econometri III\ipc_fipe.xlsx',
            index_col=0
            )
    y_menos_3 = y.iloc[:-3]
    y_diff_menos_3 = y_menos_3.diff().iloc[1:]
    teste_diff = box_jenkins_report(
            y_diff_menos_3,
            p=1, d=0, q=3, P=1, D=0, Q=1, s=12, return_results=True
            )

    teste_int = box_jenkins_report(
            y_menos_3,
            p=1, d=1, q=3, P=1, D=1, Q=1, s=12, return_results=True
            )
