# -*- coding: utf-8 -*-
"""
Created on Thu May 10 18:24:23 2018
@author: jzobaran
"""

def inform():
    print('''
    Funções:
    .classificar() - Para classificar os comentários
    .ajustar() - para ajustar classificações erradas
    .treinar(candidato) - treina, testa e informa sobre o algoritmo para analisar os comentários de um candidato.
            Uma vez criado, o classifier ficará salvo em um dicionário, de forma que não é preciso rodar essa função para 
            analisar novos comentários
    .analisar(candidato) - analisa os comentarios com o analisador criado por .treinar(candidato). Ainda por criar
''')
    return

def classificar():
    ''' A função intera pela lista ['cirogomesoficial', 'hmeirellesoficial', 'ad.alvarodias', 'aldorebelo', 'fernandohaddad', 'geraldoalckmin', 'guilhermeboulos.oficial', 'jairmessias.bolsonaro', 'jaqueswagneroficial', 'JoaoAmoedoNOVO', 'Lula', 'manueladavila', 'marinasilva.oficial', 'MichelTemer', 'RodrigoMaiaRJ']
        Caso queira pular um candidato (por já terem classificado ele, por exemplo), é só responder 0 quando a função perguntar quantos comentários serão classificados. Ele seguirá para o próximo candidato.
        A avaliação pode ser 'pos', 'neg', ou 'neut'. 1, 2 ou 3 também são aceitos, sendo traduzidos para, respectivamente, 'pos', 'neg', e 'neut' 
        Se a qualquer momento se quiser parar a avaliação, no meio, digite 'sair' ou 0 no lugar da avaliação.
    '''
    tempo = ''
    import pickle
    dfDict =pickle.load(open('I:\\ECONOMIC\\Working\\Política\\Eleições 2018\\Analise de sentimento\\Database\\DataFrames\\dfDict.p', 'rb' ))
    treinoDict = pickle.load(open('I:\\ECONOMIC\\Working\\Política\\Eleições 2018\\Analise de sentimento\\Database\\DataFrames\\treinoDict.p', 'rb' ))
    print('Já foram classificados: ' + str([key for key in treinoDict.keys() if len(treinoDict[key]) != 0]))
    for candidato in dfDict.keys():
        if tempo == 'sim':
            break
        print(candidato + ':')
        print(str(max(dfDict[candidato]['index'])) + ' comentários. Quantos serão classificados? (digite o total mesmo que já tenha começado)')
        quant = input()
        if quant == '0':
            continue
        volta = 0
        if candidato in treinoDict.keys():    
            volta = len(treinoDict[candidato])
        else:
            treinoDict[candidato] = []
        serieTreino = dfDict[candidato]['comentário'][volta:int(quant)]
        count = volta
        for comment in serieTreino:
            count += 1
            print(count)
            print(comment)
            print('avaliação:')
            aval = input()
            if aval == '1':
                aval = 'pos'
            elif aval == '2':
                aval = 'neg'
            elif aval == '3':
                aval = 'neut'
            elif aval == 'sair' or aval == '0':             
                print('Tem certeza?')
                tempo2 = input()
                if tempo2 == 'sim':
                    break
            while aval not in ['pos', 'neg', 'neut']:
                print('A avaliação deve ser pos, neg ou neut')
                aval = input()
            treinoDict[candidato].append((comment, aval))
            pickle.dump(treinoDict, open('I:\\ECONOMIC\\Working\\Política\\Eleições 2018\\Analise de sentimento\\Database\\DataFrames\\treinoDict.p', 'wb' ))
        print('Fazer uma pausa? ("sim" para sair)')
        tempo = input()
    return

def ajustar():
    import pickle
    treinoDict = pickle.load(open('I:\\ECONOMIC\\Working\\Política\\Eleições 2018\\Analise de sentimento\\Database\\DataFrames\\treinoDict.p', 'rb' ))
    print(treinoDict.keys())
    print('Candidato:')
    cand = input()
    while True:
        print("Número:")
        num = int(input()) - 1
        if num == -1:
            break
        listTD = list(treinoDict[cand][num])
        print(listTD)
        print('Ajuste:')
        listTD[1] = input()
        treinoDict[cand][num] = tuple(listTD)
        pickle.dump(treinoDict, open('I:\\ECONOMIC\\Working\\Política\\Eleições 2018\\Analise de sentimento\\Database\\DataFrames\\treinoDict.p', 'wb' ))
    return

def trainNBC(candidato, Type, example):
    '''
    If Type == 'neut', classifies features as positive or negative. 
    If Type == 'pos', classifies features as positive or non positive.
    If Type  == 'neg', classifies features as negative or non negative.
    '''
    import nltk, nltk.classify.util, pickle
    from nltk.tokenize import word_tokenize

    #dictionary in which the classifiers will be stored. Important, as this model takes some time to run    
    dictClass = pickle.load(open('I:\\ECONOMIC\\Working\\Política\\Eleições 2018\\Analise de sentimento\\Database\\dictClass.p', 'rb' ))
  
    #checks wheter the chosen candidate was correctly set
    #this can be freely customized for other datasets
    candidatos = ['cirogomesoficial', 'hmeirellesoficial', 'ad.alvarodias', 'fernandohaddad', 'geraldoalckmin', 'jairmessias.bolsonaro', 'Lula', 'marinasilva.oficial']
    if candidato not in candidatos:
        print('Inválido. Os nomes válidos são:')
        print(candidatos)
        return
    print(candidato.upper())
    print('NBC model for ' + Type + ' and non ' + Type)
    #loads the pre classified sample. This can be freely customizes, as long as it is a pandas DataFrame with columns
    #'texto' containing the classified text and 'sentimento' containing the sentiment evaluation (pos, neut, neg) 
    treinoDict = pickle.load(open('I:\\ECONOMIC\\Working\\Política\\Eleições 2018\\Analise de sentimento\\Database\\DataFrames\\treinoDict.p', 'rb' ))
    avaliados = []
    #exclued neutral sentiments for Type == neut
    #otherwise, tranforms neutral sentiments in non Type
    if Type == 'neut':
        for i in range(len(treinoDict[candidato])):
            if treinoDict[candidato][i][1] != Type:
                avaliados.append(treinoDict[candidato][i])
    else:
        for i in range(len(treinoDict[candidato])):
            if treinoDict[candidato][i][1] != Type:
                avaliados.append((treinoDict[candidato][i][0], '~' + Type))
            else:
                avaliados.append(treinoDict[candidato][i])
    if example == True:
        avaliados = avaliados[:50]
    palavras = set(word.lower() for passage in avaliados for word in word_tokenize(passage[0])) #Preprocessing; this takes time
    t = [({word: (word in word_tokenize(x[0])) for word in palavras}, x[1]) for x in avaliados] #Preprocessing; this takes time
    train = t[:int(((len(t))*0.75))]
    test = t[int(((len(t))*0.75)):]   #splits training and testing samples
    print('t values built')
    
    #creates and stores the classifier
    classifier = nltk.NaiveBayesClassifier.train(train)
    dictClass[candidato]['NBC'] = [classifier, Type]
    pickle.dump(dictClass, open('I:\\ECONOMIC\\Working\\Política\\Eleições 2018\\Analise de sentimento\\Database\\dictClass.p', 'wb' ))
    
    #reporting result from now on
    print('Most informative features:')
    print(classifier.show_most_informative_features())
    accuracyTest = nltk.classify.util.accuracy(classifier, test)
    accuracyTrain = nltk.classify.util.accuracy(classifier, train)
    print('Accuracy against training set:')
    print(accuracyTrain)
    print('Accuracy against testing set:')
    print(accuracyTest)
    return

def trainSVM(candidato, clas):
    '''
    SVM sentiment analisys model. Classifies features as 'clas' and '~clas'
    '''
    import numpy as np
    import pandas as pd
    
    import pickle
    from nltk.corpus import stopwords
    from nltk.tokenize import TweetTokenizer
    from bs4 import BeautifulSoup
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
   
    #checks wheter the chosen candidate was correctly set
    #this can be freely customized for other datasets
    candidatos = ['cirogomesoficial', 'hmeirellesoficial', 'ad.alvarodias', 'fernandohaddad', 'geraldoalckmin', 'jairmessias.bolsonaro', 'Lula', 'marinasilva.oficial']
    if candidato not in candidatos:
        print('Choose a valid name from:')
        print(candidatos)
        return
    
    print(candidato.upper())
    print("Testing " + str(clas) + ' and not ' + str(clas))
    
    #loads the pre classified sample. This can be freely customizes, as long as it is a pandas DataFrame with columns
    # 'texto' containing the classified text and 'sentimento' containing the sentiment evaluation (pos, neut, neg) 
    treinoDict = pickle.load(open('I:\\ECONOMIC\\Working\\Política\\Eleições 2018\\Analise de sentimento\\Database\\DataFrames\\treinoDict.p', 'rb' ))
    dados_raw = pd.DataFrame(treinoDict[candidato], columns = ['texto', 'sentimento'])        
    
    dados = dados_raw.copy()
    
    #tranforms the sentiment strings (pos, neg and neut) in integers
    dados['sentimento'] = dados['sentimento'].apply(lambda x: 1 if x == clas else 0)
    
    #parses the facebook text eliminating html specific syntax
    dados['texto_parsed'] = dados['texto'].apply(lambda x: BeautifulSoup(x, 'lxml').text)
    
    #keeps only the relevant feature and target
    dados = dados.loc[:, ['texto_parsed', 'sentimento']]
    
    #splits data in training and testing samples and extracts its values matrixes
    train, test = train_test_split(dados, test_size = 0.2, random_state = 1)
    X_train = train['texto_parsed'].values
    X_test = test['texto_parsed'].values
    y_train = train['sentimento']
    y_test = test['sentimento']
    
    #tokenizes the text data. TweetTokenizer works fine for facebook data
    def tokenize(texto):
        tknzr = TweetTokenizer()
        return tknzr.tokenize(texto)        

    #this should be altered according to the feature language    
    pt_stopwords = set(stopwords.words('portuguese'))
    
    #strips stopwrods
    def stripSW(tokens):
        stripped = [w for w in tokens if not w in pt_stopwords]
        return stripped
    
    #preprocesses features
    vectorizer = CountVectorizer(
            analyzer = 'word',
            tokenizer = tokenize,
            lowercase = True,
            ngram_range = (1, 1),
            strip_accents = 'ascii',
            stop_words = pt_stopwords)
    
    kfolds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)
    
    np.random.seed(1)
    
    pipeline_svm = make_pipeline(vectorizer,
                                 SVC(probability = True, kernel = 'linear', class_weight = 'balanced', max_iter = 100000))
    grid_svm = GridSearchCV(pipeline_svm,
                            param_grid = {'svc__C': [0.01, 0.1, 1]},
                            cv = kfolds,
                            scoring = 'roc_auc',
                            verbose = 10)
    
    grid_svm.fit(X_train, y_train)

    #tests the model against the testing set and return relevant scores
    def report_results(model, X, y):
        pred = model.predict(X)                          
        
        acc = accuracy_score(y, pred)                    #accuracy
        f1 = f1_score(y, pred)                           #2*(prec * rec) / (prec + rec)
        rec = recall_score(y, pred)                      #true positives/(true positives + false negatives)
        prec = precision_score(y, pred)                  #true positives/(true positives+ false positives)
        result = {'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
        return result
    
    #classifies data and return the predictions. For consultation only, not included in final results
    def report_clas(model, X):
        pred_proba = model.predict_proba(X)[:, 1]
        pred = model.predict(X)                          
        return [pred_proba, pred]        

    resultTest = report_results(grid_svm.best_estimator_, X_test, y_test)
    resultTrain = report_results(grid_svm.best_estimator_, X_train, y_train)
    
    result = {'Scores against training set': resultTrain,
              'Scores against testing set': resultTest,       #in order to use classifier for classifying new data,
              'Best classifier': grid_svm.best_estimator_}    #self['Best classifier'].predict(newData) should be passed
    
    return result
    