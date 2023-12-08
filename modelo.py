# 1. Importação das bibliotecas utilizadas no processo de tratamento dos dados

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# 1. Bibliotecas utilizadas

import pandas as pd # pandas
import matplotlib.pyplot as plt # matplotlib
from wordcloud import WordCloud # wordcloud
from sklearn.feature_extraction.text import TfidfVectorizer

# 2. Downloads necessários

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

vectorizer = TfidfVectorizer()

class Modelo():

    # def __init__(self) -> None:
    #     self.analiseAux = None
    # #### Análise exploratória dos dados
    def analise(self):

        # 2. Leitura do dataset utilizado

        df_sentimentos = pd.read_csv("./data/data.csv",sep=",",encoding="utf-8")

        df_sentimentos['sentence_length'] = df_sentimentos['Sentence'].apply(lambda x: len(x.split()))

        # Dropando valores duplicados
        df_sentimentos_clean = df_sentimentos.drop_duplicates()

        # pegando os dados duplicados e vendo que algumas sentencas estão com sentimentos diferentes
        df_sentimentos_duplicados = df_sentimentos_clean[df_sentimentos_clean.duplicated(subset=['Sentence'], keep=False)]

        #deixando somente os dados unicos
        df_sentimentos_unicos = df_sentimentos_clean.drop_duplicates(subset=['Sentence'], keep=False)

        # #### Tratamento dos dados

        stop_words = set(stopwords.words('english'))

        def clean_text(texto):
            texto = re.sub(r'[^\w\s]', '', texto) 
            palavras = word_tokenize(texto)
            lemmatizer = WordNetLemmatizer()
            text = [lemmatizer.lemmatize(palavra.lower()) for palavra in palavras if palavra.lower() not in stop_words]
            return text

        df_sentimentos_unicos = df_sentimentos_unicos.copy()
        df_sentimentos_unicos['cleaned_sentence'] = df_sentimentos_unicos['Sentence'].apply(lambda x: clean_text(x))
        df_sentimentos_unicos['cleaned_sentence_length'] = df_sentimentos_unicos['cleaned_sentence'].apply(lambda x: len(x))

        # 4. Tratamento dos dados

        def quebrar_sentencas(text):
            return nltk.sent_tokenize(text)

        # Aplicar a função na coluna 'Sentence'
        df_sentimentos_unicos['Sentence_sents'] = df_sentimentos_unicos['Sentence'].apply(quebrar_sentencas)

        # Contar o número de sentenças em cada linha
        df_sentimentos_unicos['qtd_sentencas'] = df_sentimentos_unicos['Sentence_sents'].apply(len)

        # 6. Preparando o dataset para treinar o modelo

        df_sentimentos_unicos = df_sentimentos_unicos.drop(['sentence_length', 'cleaned_sentence_length'], axis=1)

        df_sentimentos_unicos['cleaned_sentence_str'] = df_sentimentos_unicos['cleaned_sentence'].apply(' '.join)

        df_sentimentos_unicos['Sentimento_Num'] = df_sentimentos_unicos['Sentiment'].replace({'negative': 0, 'neutral': 1, 'positive': 2})

        return df_sentimentos_unicos


    # #### SVM
    def SVM(self,df_sentimentos_unicos):
        # 7. Balanceamento de Classes

        # ### Classificação

        from sklearn.model_selection import train_test_split
        from sklearn import svm
        from sklearn.metrics import classification_report
        from sklearn.model_selection import GridSearchCV

        df_sentimentos_unicos
        print(df_sentimentos_unicos)

        # Carregue os dados e realize a divisão entre treino e teste
        print("Treinando modelo ...")
        X = vectorizer.fit_transform(df_sentimentos_unicos['cleaned_sentence_str'])
        X_train, X_test, y_train, y_test = train_test_split(X, df_sentimentos_unicos['Sentimento_Num'], test_size=0.3, random_state=20)
        # Defina os parâmetros a serem testados
        param_grid = {'C': [0.01, 1, 10, 100, 1000],  
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                    'kernel': ['rbf', 'poly', 'sigmoid']}
        # Inicialize o classificador SVM
        clf = svm.SVC(kernel='linear', gamma='scale')
        # Realize a busca em grade com validação cruzada
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1_macro')
        grid_search.fit(X_train, y_train)
        # Obtenha os melhores parâmetros
        best_params = grid_search.best_params_
        # Use os melhores parâmetros para treinar o modelo final
        clf = svm.SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'])
        clf.fit(X_train, y_train)
        print("Modelo treinado")
        
        return clf

    def clean_text(self,texto):
        stop_words = set(stopwords.words('english'))
        texto = re.sub(r'[^\w\s]', '', texto) 
        palavras = word_tokenize(texto)
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(palavra.lower()) for palavra in palavras if palavra.lower() not in stop_words]
        return text

    def converte_string(self,texto):
        texto =  vectorizer.transform(texto)
        return texto
