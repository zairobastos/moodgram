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

class Modelo():

    # def __init__(self) -> None:
    #     self.analiseAux = None
    # #### Análise exploratória dos dados
    def analise(self):

        # 2. Leitura do dataset utilizado

        df_sentimentos = pd.read_csv("./data/data.csv",sep=",",encoding="utf-8")
        print(df_sentimentos.head())

        # 3. extraindo algumas infos

        # Visualizando os dados de forma aleatória
        print(df_sentimentos.sample(5))

        # Visualizando informações gerais sobre o dataset
        print(df_sentimentos.info())

        # Visualizando a distribuição dos dados
        colunas = list(df_sentimentos.columns)
        for coluna in colunas:
            print(f"Distribuição da coluna {coluna}")
            print(df_sentimentos[coluna].value_counts())
            print("\n")

        # Tamanho do dataset
        print("Tamanho do dataset")
        print("Quantidade de linhas: ", df_sentimentos.shape[0])
        print("Quantidade de colunas: ", df_sentimentos.shape[1])

        # Visualizando a quantidade de dados faltantes
        print("Dados faltantes: ")
        print(df_sentimentos.isnull().sum())

        # Ver estatísticas descritivas do datasetp
        print(df_sentimentos.describe())

        df_sentimentos['sentence_length'] = df_sentimentos['Sentence'].apply(lambda x: len(x.split()))
        print(df_sentimentos['sentence_length'].describe())

        #valores duplicados
        print("Valores duplicados: ")
        df_sentimentos_duplicados = df_sentimentos[df_sentimentos.duplicated()]
        print(df_sentimentos_duplicados)
        # Dropando valores duplicados
        print("Dropando valores duplicados: ")
        df_sentimentos_clean = df_sentimentos.drop_duplicates()
        print(df_sentimentos_clean)

        # pegando os dados duplicados e vendo que algumas sentencas estão com sentimentos diferentes
        df_sentimentos_duplicados = df_sentimentos_clean[df_sentimentos_clean.duplicated(subset=['Sentence'], keep=False)]
        df_sentimentos_duplicados_ordenados = df_sentimentos_duplicados.sort_values(by='Sentence')
        print(df_sentimentos_duplicados_ordenados)
        print("Quantidade de sentimentos: ", df_sentimentos_duplicados_ordenados['Sentiment'].value_counts())

        #deixando somente os dados unicos
        df_sentimentos_unicos = df_sentimentos_clean.drop_duplicates(subset=['Sentence'], keep=False)
        print(df_sentimentos_unicos)

        # sentimentos_counts = df_sentimentos_unicos['Sentiment'].value_counts()
        # Criar um gráfico de pizza
        # plt.figure(figsize=(10,6))
        # patches, texts, autotexts = plt.pie(sentimentos_counts, labels = sentimentos_counts.index, autopct=lambda p: '{:.2f}% ({:.0f})'.format(p,(p/100)*sentimentos_counts.sum()))
        # plt.title('Distribuição dos Sentimentos')
        # plt.legend(patches, sentimentos_counts.index, loc="best")
        # plt.show()

        # print("Exemplo: ",df_sentimentos_unicos['Sentence'].iloc[1])

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
        print(df_sentimentos_unicos.head(5))

        # 4. Tratamento dos dados

        def quebrar_sentencas(text):
            return nltk.sent_tokenize(text)

        # Aplicar a função na coluna 'Sentence'
        df_sentimentos_unicos['Sentence_sents'] = df_sentimentos_unicos['Sentence'].apply(quebrar_sentencas)

        # Contar o número de sentenças em cada linha
        df_sentimentos_unicos['qtd_sentencas'] = df_sentimentos_unicos['Sentence_sents'].apply(len)

        # Filtrar e imprimir as linhas onde o número de sentenças é maior ou igual a 2
        print(df_sentimentos_unicos[df_sentimentos_unicos['qtd_sentencas'] >= 2])

        print(df_sentimentos_unicos.loc[11,'Sentence'])

        for frase in df_sentimentos_unicos.loc[11, 'Sentence_sents']:
            print(frase)

        print(df_sentimentos_unicos[df_sentimentos_unicos['qtd_sentencas'] >= 2].value_counts('Sentiment'))

        # 5. Nuvem de palavras

        # def nuvem_palavras(sentiment):
        #     df = df_sentimentos_unicos[df_sentimentos_unicos['Sentiment'] == sentiment]
        #     text = ' '.join(palavra for sublist in df.cleaned_sentence for palavra in sublist)
        #     wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
        #     plt.imshow(wordcloud, interpolation="bilinear")
        #     plt.axis("off")
        #     plt.title(f"Nuvem de palavras para o sentimento: {sentiment}")
        #     plt.show()

        # sentimentos = df_sentimentos_unicos['Sentiment'].unique()
        # for sentimento in sentimentos:
        #     nuvem_palavras(sentimento)

        # 6. Preparando o dataset para treinar o modelo

        df_sentimentos_unicos = df_sentimentos_unicos.drop(['sentence_length', 'cleaned_sentence_length'], axis=1)
        print(df_sentimentos_unicos.sample(3))

        df_sentimentos_unicos['cleaned_sentence_str'] = df_sentimentos_unicos['cleaned_sentence'].apply(' '.join)
        print(df_sentimentos_unicos.sample(3))

        df_sentimentos_unicos['Sentimento_Num'] = df_sentimentos_unicos['Sentiment'].replace({'negative': 0, 'neutral': 1, 'positive': 2})
        print(df_sentimentos_unicos.sample(3))

        print(df_sentimentos_unicos['Sentimento_Num'].value_counts())

        return df_sentimentos_unicos


    # #### SVM
    def SVM(self,df_sentimentos_unicos):
        # 7. Balanceamento de Classes

        # ### Classificação

        from sklearn.model_selection import train_test_split
        from sklearn import svm
        from sklearn.metrics import classification_report
        from sklearn.model_selection import GridSearchCV
        from imblearn.over_sampling import ADASYN
        from imblearn.over_sampling import SMOTE
        from collections import Counter

        df_sentimentos_unicos
        print(df_sentimentos_unicos)

        # Carregue os dados e realize a divisão entre treino e teste
        print("chegou aqui")
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df_sentimentos_unicos['cleaned_sentence_str'])
        X_train, X_test, y_train, y_test = train_test_split(X, df_sentimentos_unicos['Sentimento_Num'], test_size=0.3, random_state=20)
        print("chegou aqui 1")
        # Defina os parâmetros a serem testados
        param_grid = {'C': [0.01, 1, 10, 100, 1000],  
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                    'kernel': ['rbf', 'poly', 'sigmoid']}
        print("chegou aqui 2")
        # Inicialize o classificador SVM
        clf = svm.SVC(kernel='linear', gamma='scale')
        print("chegou aqui 3")
        # Realize a busca em grade com validação cruzada
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1_macro')
        grid_search.fit(X_train, y_train)
        print("chegou aqui 4")
        # Obtenha os melhores parâmetros
        best_params = grid_search.best_params_
        print("chegou aqui 5")
        # Use os melhores parâmetros para treinar o modelo final
        clf = svm.SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'])
        clf.fit(X_train, y_train)
        print("chegou aqui 6")
        # return(clf)
        # Faça previsões e imprima o relatório de classificação
        print("Terminou execução")
        print(clf)
        return clf
        # print(classification_report(y_test, y_pred))

    def clean_text(self,texto):
        stop_words = set(stopwords.words('english'))
        texto = re.sub(r'[^\w\s]', '', texto) 
        palavras = word_tokenize(texto)
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(palavra.lower()) for palavra in palavras if palavra.lower() not in stop_words]
        return text

    def converte_string(self,texto):
        vectorizer = TfidfVectorizer()
        texto =  vectorizer.fit_transform(texto)
        return texto
