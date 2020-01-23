# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:24:57 2019

Estudando

@author: josez
"""
# %%

from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import itertools
import subprocess
from comtypes import COMError
import pyeviews as evp
import re
import os
USER = os.getlogin()
GLOBALEVAPP = None

# %% Box Jenkins


def _GetApp(app=None):
    global GLOBALEVAPP
    if app is not None:
        return app
    if GLOBALEVAPP is None:
        globalevapp = evp.GetEViewsApp(instance='new', showwindow=True)
    return globalevapp


def gridsearch_box_jenkins_eviews(y, arima, sarima=None, c='c',
                                  optmethod='legacy', arma='cls, z',
                                  t=0.1, q1=6, q2=12, name=None, col=None,
                                  output_dir=None, app=None):
    app = _GetApp(app)

    evp.PutPythonAsWF(y, app=app)
    app.Run('pageselect Untitled')
    if name:
        app.Run('pagerename Untitled {}'.format(name))
    if not col:
        col = y.columns[0]

    p, d, q = arima[0], arima[1], arima[2]
    P, D, Q, s = sarima[0], sarima[1], sarima[2], sarima[3]
    pdq = list(itertools.product(range(p + 1), range(d + 1), range(q + 1)))
    s_pdq_constr = list(
            itertools.product(range(P + 1), range(D + 1), range(Q + 1))
            )
    s_pdq = []
    for i in s_pdq_constr:
        params = list(i)
        params.append(s)
        s_pdq.append(tuple(params))

    # Dataframe de saída
    index = []
    for param in pdq:
        for s_param in s_pdq:
            index.append((param, s_param))
    output_df = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(index),
            columns=[
                    'convergência', 'estacionariedade',
                    'invertibilidade', 'SE of reg', 'AIC', 'SC',
                    'HQ', 'P[Q({})]'.format(q1), 'P[Q({})]'.format(q2),
                    'par. não-signif. a {}'.format(t)
                      ]
            )
    for param in pdq:
        p, d, q = param[0], param[1], param[2]
        if p > 1:
            p = '1 to {}'.format(p)
        if q > 1:
            q = '1 to {}'.format(q)
        for s_param in s_pdq:
            P, D, Q, s = s_param[0], s_param[1], s_param[2], s_param[3]
            # nome unico p/ id
            eq_name = 'bj_{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(
                    param[0], param[1], param[2], P, D, Q, s)
            print('Estimando {}'.format(eq_name))
            P, Q = P*s, Q*s
            estimate_cmd = 'equation {0}.ls(arma={1}, '\
                'optmethod={2}) {3} {4} ar({5}) ma({6}) '\
                'sar({7}) sma({8})'.format(
                        eq_name, arma, optmethod, col, c, p, q, P, Q
                        )
            # Case algum dos valores seja 0, devemos limpa-los
            find_inv_terms_regex = re.compile(r'''(\sar\(0\))?
                                              (\sma\(0\))?
                                              (\ssar\(0\))?
                                              (\ssma\(0\))?
                                              ''', re.VERBOSE)
            estimate_cmd_limpo = find_inv_terms_regex.sub('', estimate_cmd)
            try:
                app.Run(estimate_cmd_limpo)
            except COMError:
                continue

            # Checando convergências
            # Muitas views não tem acesso
            # O workaround é criar uma tabela e acessar a célula direto
            # Por hora requer salvar o csv, não conseguia acessar os valores em
            # texto pela interface do EViews

            gen_estimation_table = 'freeze(estm_tab) {}.output'.format(eq_name)
            app.Run(gen_estimation_table)
            global USER
            app.Run('estm_tab.save eviews_tab')
            dir = r'C:\Users\{}\OneDrive\Documents\eviews_tab.csv'.format(USER)
            tab = pd.read_csv(dir)
            app.Run('delete estm_tab')
            os.remove(dir)
            converg = tab.iloc[4, 0]
            if type(converg) == float:
                output_df.loc[
                        (param, s_param),
                        ['convergência']
                        ] = 'Não se aplica'
                # Se o modelo não tem termos ARIMA, segue para o proximo
                continue
            if 'not' in converg:
                output_df.loc[
                        (param, s_param),
                        ['convergência']
                        ] = 'Não converge'
                continue  # Se o modelo não converge, segue para o proximo
            else:
                output_df.loc[
                        (param, s_param),
                        ['convergência']
                        ] = 'Converge'
            # Pega as estatísticas do modelo
            # Pegando coefs os quais estat t exclui
            find_parenteses_regex = re.compile(r'((\()?(\))?)')
            gen_coefs = 'group coefs {}.@coeflabels'.format(eq_name)
            app.Run(gen_coefs)
            coefs = app.GetGroup('coefs')[0][0]
            index_coefs =\
                {val: num+1 for num, val in enumerate(coefs.split())}
            excluidos_a_t = ''
            for coef, index in index_coefs.items():
                # Temos que tirar os parenteses
                coef_limpo = find_parenteses_regex.sub('', coef)
                gen_pval = 'group {0}_pval {1}.@pvals({2})'.format(
                        coef_limpo,
                        eq_name,
                        index
                        )
                app.Run(gen_pval)
                pval = app.GetGroup('{}_pval'.format(coef_limpo))[0][0]
                app.Run('delete {}_pval'.format(coef_limpo))
                if pval > t:
                    if excluidos_a_t == '':
                        excluidos_a_t = excluidos_a_t + '{}'.format(coef)
                    else:
                        excluidos_a_t = excluidos_a_t + ', {}'.format(coef)
            app.Run('delete coefs')
            # Pegando raízes
            gen_roots_cmd = '{}.arma(type=root, save=root)'.format(eq_name)
            app.Run(gen_roots_cmd)
            try:
                roots_ar = app.Get('ROOT_AR')
                app.Run('delete ROOT_AR')
                estacionario = all(roots_ar) <= 1
            except COMError:
                estacionario = 'não se aplica'
            try:
                roots_ma = app.Get('ROOT_MA')
                app.Run('delete ROOT_MA')
                invertivel = all(roots_ma) <= 1
            except COMError:
                invertivel = 'não se aplica'

            # Pegando erro padrão e criterios de info
            gen_se_ic = 'group se_ic {0}.@se {0}.@aic {0}.@schwarz'\
                ' {0}.@hq'.format(eq_name)
            app.Run(gen_se_ic)
            ic_se = app.GetGroup('se_ic')[0]
            app.Run('delete se_ic')

            # Pegando q-val dos correlogramas
            # Accessa uma view, assim como convergência
            gen_correl = 'freeze(corr_tab) {0}.correl'.format(eq_name)
            app.Run(gen_correl)
            access_p_q1 = 'scalar p_q1 = @val(corr_tab({0}, 7))'.format(q1 + 7)
            app.Run(access_p_q1)
            access_p_q2 = 'scalar p_q2 = @val(corr_tab({0}, 7))'.format(q2 + 7)
            app.Run(access_p_q2)
            p_q1 = app.Get('p_q1')
            p_q2 = app.Get('p_q2')
            app.Run('delete p_q1')
            app.Run('delete p_q2')
            app.Run('delete corr_tab')

            output_df.loc[
                    (param, s_param), ['SE of reg', 'AIC', 'SC', 'HQ']
                    ] = ic_se
            output_df.loc[
                    (param, s_param), ['estacionariedade', 'invertibilidade']
                    ] = (estacionario, invertivel)
            output_df.loc[
                    (param, s_param), ['par. não-signif. a {}'.format(t)]
                    ] = excluidos_a_t
            output_df.loc[
                    (param, s_param),
                    ['P[Q({})]'.format(q1), 'P[Q({})]'.format(q2)]
                    ] = (p_q1, p_q2)

    # Salva o df de detalhes e o worfile do Eviews
    if output_dir:
        if name:
            output_df.to_excel(os.path.join(output_dir, name+'.xls', ))
            gen_save_command = 'pagesave \"{}_Eviews\"'.format(
                    os.path.join(output_dir, name)
                    )
        else:
            output_df.to_excel(os.path.join(output_dir, 'box_jenkins.xls'))
            gen_save_command = 'wfsave \"{}\"'.format(
                    os.path.join(output_dir, 'Eviews')
                    )
        app.Run(gen_save_command)
    # Cleanup
    try:
        app.Hide()
    except Exception:
        pass
    app = None
    evp.Cleanup()
    subprocess.call('taskkill /F /IM Eviews10.exe')

    return output_df

# %% Causalidade de granger


def teste_granger(df, maxlag=1, verbose=False):
    teste = grangercausalitytests
    result = {}
    nomes = df.columns
    df_teste = df.copy()
    df_teste.fillna(method='ffill', inplace=True)
    df_teste_invert = df_teste.iloc[:, [1, 0]]
    result['{1} causa {0}'.format(nomes[0], nomes[1])] = teste(
            df_teste, maxlag=maxlag, verbose=verbose
            )
    result['{0} causa {1}'.format(nomes[1], nomes[0])] = teste(
            df_teste_invert, maxlag=maxlag, verbose=verbose
            )
    return result

# %%


if __name__ == '__main__':
    output_dir = r'C:\Users\josez\Desktop\Economia\Econometri III\box jenkins'
    ipc_fipe = pd.read_excel(
            r'C:\Users\josez\Desktop\Economia\Econometri III\ipc_fipe.xlsx',
            index_col=0
            )
    ipc_fipe = ipc_fipe.iloc[:-3]
    desemprego = pd.read_excel(
            r'C:\Users\josez\Desktop\Economia\Econometri III\desemprego.xlsx',
            index_col=0
            )
    desemprego = desemprego.iloc[:-3]
    try:
        app = _GetApp()
    except COMError:
        os.system('taskkill /F /IM Eviews10.exe')
        app = _GetApp()
    gridsearch_ipc = gridsearch_box_jenkins_eviews(
            ipc_fipe,
            arima=(1, 0, 3),
            sarima=(1, 0, 1, 12),
            c='c',
            name='ipc_fipe',
            col='d(ipc_fipe)',
            output_dir=output_dir,
            app=app
            )
    print('ipc_done')
    gridsearch_desemp = gridsearch_box_jenkins_eviews(
        desemprego,
        arima=(3, 0, 3),
        sarima=(1, 0, 1, 12),
        c='',
        q1=7,
        name='desemprego',
        col='d(desemprego, 1, 12)',
        output_dir=output_dir,
        app=app
        )
