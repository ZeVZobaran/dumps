# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:44:32 2018

Under construction

@author: jzobaran
"""

def bestModelSelector(X, y, yTrue = 0):
    '''
    Generates a full report of selected models' performance for given qualitative target and non standirdized numeric features
    If yTrue is stated, compares the probabilities with the actual, non qualitative, target data
    Train size = 0.75
    Models: SVM, tree, random forest, gaussian naive Bayes
    '''
    import pandas as pd
    import sklearn as sk
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

    #mask output dictionary with all constructed classifiers and reports
    #the actual output dictionary will be organized at the end of the function
    result = {}
    
# =============================================================================
#     Preprocessing
# =============================================================================

    #splits train and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

    #standardize features
    scaler = sk.preprocessing.StandardScaler().fit(X_train)
    X_train, X_test = pd.DataFrame(scaler.transform(X_train), index = X_train.index), pd.DataFrame(scaler.transform(X_test), index = X_test.index)
    
    #gets the test values for yTrue
    if type(yTrue) == pd.core.series.Series:
        yTrue = yTrue.loc[X_test.index]

    #creates Kfolds for model validation
    cv = KFold(5, shuffle = True, random_state = 42)    
    
    #índice para uso futuro
    index = X_test.index

# =============================================================================
# Result reporting
# =============================================================================

    #basic result reporting function
    def report_results(model, X, y):
        pred = model.predict(X)                          
        
        acc = accuracy_score(y, pred)                         #overall accuracy
        f1 = f1_score(y, pred)                                #harmonic mean of precision and recall
        rec = recall_score(y, pred)                           #true positives/(total positives)
        prec = precision_score(y, pred)                       #true positives/(predicted positives)
        result = {'f1 score': f1, 'accuracy': acc, 'precision': prec, 'recall': rec}
        return result

    #TO DO: dumb classification algorithms for perfomance checking 
# =============================================================================
#     Criar algoritmos
#     dumbAlgos = {
#             'All positive': 'placeholder',
#             'All negative':'placeholder',
#             'Neg Pos Neg Pos':'placeholder',
#             'Random walk': 'placeholder'
#                 }
# =============================================================================

# =============================================================================
#     SVM
# =============================================================================
    
    #SVM parameters
    parametersSVM = {'kernel': ('linear', 'rbf'),
                  'C': [0.001, 0.01, 0.1, 1, 10],
                  'class_weight': [None, 'balanced']}
    
    #scv model, probability enabled
    svc = SVC(probability = True)

    #grid search for the best SVC model    
    clfSVM = GridSearchCV(
                       svc, 
                       param_grid = parametersSVM, 
                       cv = cv,
                       verbose = 0,
                       )

    clfSVM.fit(X_test, y_test)

    #dataframe with predicted vs actual testing values
    proba = pd.DataFrame(clfSVM.best_estimator_.predict_proba(X_test), index = index, columns = clfSVM.classes_)
    if type(yTrue) == pd.core.series.Series:
        proba['Obs'] = yTrue

    reportSVM = {
            'Best parameters': clfSVM.best_params_,
            'Scores against training set': report_results(clfSVM, X_train, y_train),
            'Scores against testing set': report_results(clfSVM, X_test, y_test),
            'Confusion matrix on testing set': confusion_matrix(y_test, clfSVM.best_estimator_.predict(X_test)),
            'Confusion matrix on training set': confusion_matrix(y_train, clfSVM.best_estimator_.predict(X_train))
                        }
    
    result['SVM'] = {
            'Classifier': clfSVM.best_estimator_,
            'Probability': proba,
            'Report': reportSVM
                            }

# =============================================================================
#     Naive Bayes
# =============================================================================
    from sklearn.naive_bayes import GaussianNB
    
    GNB = GaussianNB()

    #Gaussian naive bayes' only parameter is what prior should it attibute to the data distribution
    #It is usually better to go with None    
    if len(clfSVM.classes_) == 3:
        parametersGNB = {'priors': [None, (0.25, 0.25, 0.5), (0.25, 0.5, 0.25), (0.5, 0.25, 0.25)]}
    elif len(clfSVM.classes_) == 2:
        parametersGNB = {'priors': [None, (0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]}
        
    clfGNB  = GridSearchCV(
                       GNB, 
                       param_grid = parametersGNB, 
                       cv = cv,
                       verbose = 0,
                       )
    
    clfGNB.fit(X_train, y_train)
    
    reportGNB = {
            'Best parameters': clfGNB.best_params_,
            'Scores against training set': report_results(clfGNB, X_train, y_train),
            'Scores against testing set': report_results(clfGNB, X_test, y_test),
            'Confusion matrix on testing set': confusion_matrix(y_test, clfGNB.best_estimator_.predict(X_test)),
            'Confusion matrix on training set': confusion_matrix(y_train, clfGNB.best_estimator_.predict(X_train))
                        }
    
    probaGNB = pd.DataFrame(clfGNB.best_estimator_.predict_proba(X_test), index = index, columns = clfGNB.classes_)
    if type(yTrue) == pd.core.series.Series:
        probaGNB['Obs'] = yTrue

    result['GNB'] = {
            'Classifier': clfGNB,
            'Probability': probaGNB,
            'Report': reportGNB
            }

# =============================================================================
#     Tree
# =============================================================================
    from sklearn.tree import DecisionTreeClassifier
    
    #possible parameters for the tree algorithm, will be explored by the grid search
    parametersTree = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [1, 2, 3, 5, len(X_train), None]
            }
    
    
    tree = DecisionTreeClassifier()

    clfTree  = GridSearchCV(
                       tree, 
                       param_grid = parametersTree, 
                       cv = cv,
                       verbose = 0,
                       )
    
    clfTree.fit(X_train, y_train)

    reportTree = {
            'Best parameters': clfTree.best_params_,
            'Scores against training set': report_results(clfTree, X_train, y_train),
            'Scores against testing set': report_results(clfTree, X_test, y_test),
            'Confusion matrix on testing set': confusion_matrix(y_test, clfTree.best_estimator_.predict(X_test)),
            'Confusion matrix on training set': confusion_matrix(y_train, clfTree.best_estimator_.predict(X_train))
                        }

    probaTree = pd.DataFrame(clfTree.best_estimator_.predict_proba(X_test), index = index, columns = clfTree.classes_)
    if type(yTrue) == pd.core.series.Series:
        probaTree['Obs'] = yTrue

    probaTree['Obs'] = y_test

    result['Tree'] = {
            'Classifier': clfTree,
            'Probability': probaTree,
            'Report': reportTree
            }

# =============================================================================
#     random forest
# =============================================================================
    from sklearn.ensemble import RandomForestClassifier

    RF = RandomForestClassifier(random_state = 42)

    #possible parameters for the RF algorithm, will be explored by the grid search
    parametersRF = {
            'n_estimators': [10, 30, 50, 80],
            'criterion': ['gini', 'entropy'],
            'max_depth': [1, 2, 3, 5, len(X_train), None]
            }

    clfRF  = GridSearchCV(
                       RF, 
                       param_grid = parametersRF, 
                       cv = cv,
                       verbose = 0,
                       )
    
    clfRF.fit(X_train, y_train)

    reportRF = {
            'Best parameters': clfRF.best_params_,
            'Scores against training set': report_results(clfRF, X_train, y_train),
            'Scores against testing set': report_results(clfRF, X_test, y_test),
            'Confusion matrix on testing set': confusion_matrix(y_test, clfRF.best_estimator_.predict(X_test)),
            'Confusion matrix on training set': confusion_matrix(y_train, clfRF.best_estimator_.predict(X_train))
                        }    

    probaRF = pd.DataFrame(clfRF.best_estimator_.predict_proba(X_test), index = index, columns = clfRF.classes_)
    if type(yTrue) == pd.core.series.Series:
        probaRF['Obs'] = yTrue

    probaRF['Obs'] = y_test

    result['Random Forest'] = {
            'Classifier': clfRF,
            'Probability': probaRF,
            'Report': reportRF
            }

# =============================================================================
#     best model suggestion
# =============================================================================
    #best model based on in and out of sample testing

    #creates dataframes with IS and OS test results
    scoresOS = {}
    scoresIS = {}
    for model in result.keys():
        scoresOS[model] = result[model]['Report']['Scores against testing set']
        scoresIS[model] = result[model]['Report']['Scores against training set']

    scoresOS = pd.DataFrame(scoresOS)
    scoresIS = pd.DataFrame(scoresOS)

    print('\nBest performing OS model, according to:')

    for label in scoresOS.index:
        print(label + ': ' + str(scoresOS.T[label].idxmax()))

    print('\nBest performing IS model, according to:')

    for label in scoresIS.index:
        print(label + ': ' + str(scoresIS.T[label].idxmax()))
        
    print('\nFull OS scores by model:\n')
    print(scoresOS)

    print('\nFull IS scores by model:\n')
    print(scoresIS)

    final = {
            'Scores': {'In sample': scoresIS, 'Out of sample': scoresIS},
            'Classifiers': {
                    'SVM': result['SVM']['Classifier'],
                    'GNB': result['GNB']['Classifier'],
                    'Tree': result['Tree']['Classifier'],
                    'Random Forest': result['Random Forest']['Classifier']
                    },
            'Probabilities':{
                    'SVM': result['SVM']['Probability'],
                    'GNB': result['GNB']['Probability'],
                    'Tree': result['Tree']['Probability'],
                    'Random Forest': result['Random Forest']['Probability']
                    },
            'Reports':{
                    'SVM': result['SVM']['Report'],
                    'GNB': result['GNB']['Report'],
                    'Tree': result['Tree']['Report'],
                    'Random Forest': result['Random Forest']['Report']
                    },
            }
        
    return final

def fetchBolsa():
    
    '''
    Outputs a matrix with IBOV data and relevant independent variables
    '''

    import numpy as np
    import pandas as pd
    import os
    #import Produto_Cambio <-- volta quando arrumarmos o quandl
    from scipy.stats.mstats import gmean
    os.chdir(r'I:\ECONOMIC\Working\Outros\Luis_produto')

    # Bovespa - Bloomberg
    bolsas = pd.read_pickle('Bolsas-Bloomberg.pkl')
    bovespa = bolsas['IBOV Index'].astype(float)
    bovespa = bovespa.resample('M').mean().to_period('M')

    # =============================================================================
    # # Cambio - Quandl
    # cambio = Produto_Cambio.atualiza_cambio_quandl()
    # cambio = cambio.resample('M').mean().to_period('M')
    # cambio.columns = ['Cambio']
    # Arrumar
    # =============================================================================

    #cambio - miguelado
    cambio = pd.read_csv('cambio.csv')
    cambio['Date'] = pd.to_datetime(cambio['Date'])
    cambio.set_index('Date', inplace = True)
    cambio = cambio.resample('M').mean().to_period('M')
    cambio.columns = ['Cambio']

    #Forward- LCA e Mesa
    forward = pd.read_pickle('LCA&Mesa-Forward.pkl')
    forward = ((1+forward/100).resample('M').apply(gmean).to_period('M')-1)*100
    forward = forward['Forward 1y']
    forward.columns = ['Juros']

    #Target FED Funds - LCA
    juros_internacinonal = pd.read_excel('Taxas_de_juros_internacionais.xlsx', sheet_name='Média Mensal')
    fed_target = juros_internacinonal.iloc[23:, [0,2]]
    fed_target.set_index('Unnamed: 0', inplace=True, drop=True)
    fed_target.columns = ['FedTarget']
    fed_target = fed_target.to_period('M')

    #MSCI - Bloomberg
    msci = pd.read_pickle('MSCI-Bloomberg.pkl')
    msci_em = msci['MXEF Index']
    msci_em  = msci_em .resample('M').mean().to_period('M')
    msci_wo = msci['GDLEACWF Index']
    msci_wo  = msci_wo .resample('M').mean().to_period('M')

    #Commodity - Bloomberg
    commodity = msci['BCOMTR Index']
    commodity  =commodity .resample('M').mean().to_period('M')

    #Feature dataframe
    matrix = pd.concat([bovespa.apply(lambda x: np.log(x)).diff(),
                        msci_wo.apply(lambda x: np.log(x)).diff(),
                        forward.apply(lambda x: np.log(x)).diff(),
                        cambio.apply(lambda x: np.log(x)).diff(),
                        fed_target['FedTarget'].apply(lambda x: np.log(x)).diff(),
                        commodity.apply(lambda x: np.log(x)).diff()], axis=1).dropna().astype(float)

    return matrix

def checkStdVar(series):
    '''
    checa se o desvio padrão é uma métrica boa para definir dados quali
    faz mais sentido ver como é usado la embaixo
    '''
    total = len(series)
    caiu = len([x for x in series if x == -1])
    lado = len([x for x in series if x == 0])
    subiu = len([x for x in series if x == 1])
    return [caiu/total, lado/total, subiu/total]

def classifierBalanceado(X, y, Xprev, metric = 'accuracy', yPrev = False):
    '''
    Cria os classifiers de bestModelSelector para X e y, e depois usa uma versão ponderada deles para classificar Xprev
    Uma métrica específica (entre accuracy, f1 score, precision e recall) pode ser definida
    Se nenhuma métrica for especificada, usamos accuracy
    Retora um dataframe com Xprev, as previsões por modelo e a ponderada
    '''
    
    import pandas as pd
    import sklearn as sk
    
    relat = bestModelSelector(X, y, yPrev)
    classifiers = relat['Classifiers']
    scores = pd.DataFrame(relat['Scores']['Out of sample'].loc[metric])
    
    #scale down de Xprev

    scaler = sk.preprocessing.StandardScaler().fit(X)
    Xprev = scaler.transform(Xprev)
    
    #previsão usando cada um dos classifiers
    pred = {}
    for name in classifiers.keys():
        mask = pd.DataFrame(classifiers[name].predict_proba(Xprev))
        pred[name] = pd.DataFrame(mask.iloc[:, 1].values, columns = [name])
    
    pred = pd.concat([pred[df] for df in pred.keys()], axis = 1)
    
    #valor pelo qual ponderar cada previsão
    total = scores[metric].sum()
    scores['pond'] = scores/total
        
    #ponderação
    predPond = pred.copy()
    mask = pred.copy()
    for model in scores['pond'].index:
        mask[model] = pred[model]*scores.loc[model, 'pond']
        
    predPond['Ponderados'] = mask.sum(axis = 1)
    
    return predPond


if __name__ == "__main__":
    matrixBolsa = fetchBolsa()
    
    #generates X and y values for classification
    #tranforms target data in qualitative values
    XBolsa = matrixBolsa.drop(columns = 'IBOV Index')
    yBolsa2Cat = matrixBolsa['IBOV Index'].apply(lambda x: 1 if x > 0 else 0)
    yBolsaTrue = matrixBolsa['IBOV Index']
    
    bolsa2Cat = bestModelSelector(XBolsa, yBolsa2Cat, yBolsaTrue)
    
    #for classification in "falls, rises, sideways", we use +-half the std var as definition for "sideways"
    lado = matrixBolsa['IBOV Index'].std()/5
    yBolsa3Cat = matrixBolsa['IBOV Index'].apply(lambda x: 1 if x > lado else (0 if lado > x > -lado else -1))
    #tests if using half the std var for "operating sideways" is a good metric
    checkStd = checkStdVar(yBolsa3Cat)
    
    bolsa3Cat = bestModelSelector(XBolsa, yBolsa3Cat, yBolsaTrue)
    
    
    
    XBolsaTeste = XBolsa[:200]
    yBolsaTeste = yBolsa2Cat[:200]
    yBolsaGabarito = yBolsa2Cat[200:222]
    xBolsaPrev = XBolsa[200:222]
    
    predPond = classifierBalanceado(XBolsaTeste,yBolsaTeste, Xprev = xBolsaPrev)
    
    
    #ponderar melhorou?
    #Teste sem levar grau de ctz em consideração
    
    check = predPond.applymap(lambda x: 1 if x > 0.5 else 0)
    for coluna in predPond.columns:
        check['teste ' + coluna] = check[coluna] - yBolsaGabarito
        #0 se acerto, 1 se erro
        check['teste ' + coluna].apply(lambda x: 0 if x == 0 else 1)
    
    check.sum()
    #veredicto: provavelmente ponderar compensa
    
