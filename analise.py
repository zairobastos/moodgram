# Bibliotecas utilizadas
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# 2. Downloads necessários
nltk.download('stopwords')
nltk.download('punkt')


def carrega_dataset(caminho_arquivo: str):
    """
        Carrega o conjunto de dados de sentimento de um arquivo CSV.

        Parâmetros:
        - caminho_arquivo (str): O caminho para o arquivo CSV.

        Retorna:
        - pd.DataFrame: O conjunto de dados carregado.
    """
    if os.path.exists(caminho_arquivo) == False:
        raise Exception("Arquivo não encontrado")
    else:
        try:
            df_sentimentos = pd.read_csv(
                caminho_arquivo, sep=",", encoding="utf-8")
        except UnicodeDecodeError:
            print("ERROR: Problema ao decodificar o arquivo. Tente outro encoding ou verifique a integridade do arquivo.")
        except Exception as e:
            print("ERROR: ", e)
        finally:
            df_sentimentos_clean = df_sentimentos.drop_duplicates()
            df_sentimentos_duplicados = df_sentimentos_clean[df_sentimentos_clean.duplicated(
                subset=['Sentence'], keep=False)]
            df_sentimentos_unicos = df_sentimentos_clean.drop_duplicates(
                subset=['Sentence'], keep=False)
            print("Arquivo carregado")
            return df_sentimentos_unicos


def clean_text(texto: str, language='english'):
    """
        Realiza a limpeza e pré-processamento de um texto.

        Parâmetros:
        - texto (str): O texto a ser processado.
        - language (str): O idioma do texto (padrão: 'english').

        Retorna:
        - list: Lista de palavras após a limpeza e pré-processamento.
    """
    stop_words = set(stopwords.words(language))
    lemmatizer = WordNetLemmatizer()

    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto).lower()

    palavras = word_tokenize(texto)

    text = [lemmatizer.lemmatize(
        palavra) for palavra in palavras if palavra not in stop_words and len(palavra) > 2]
    return text


def analise(dataset: pd.DataFrame):
    """
        Realiza análise no conjunto de dados de sentimentos.

        Parâmetros:
        - dataset (pd.DataFrame): O DataFrame contendo os dados de sentimento.

        Retorna:
        - pd.DataFrame: DataFrame com colunas adicionais para texto limpo e sentimentos numéricos.
    """
    dataset['cleaned_sentence'] = dataset['Sentence'].apply(clean_text)
    dataset['cleaned_sentence_str'] = dataset['cleaned_sentence'].apply(
        ' '.join)
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    dataset['Sentimento_Num'] = dataset['Sentiment'].map(sentiment_mapping)
    print("Análise realizada")

    return dataset


vectorizer = TfidfVectorizer()


def treina_modelo(dataset: pd.DataFrame):
    """
        Treina o modelo SVM.

        Parâmetros:
        - dataset (pd.DataFrame): O DataFrame contendo os dados de sentimento.

        Retorna:
        - svm.SVC: Modelo SVM treinado.
    """
    print("Treinando modelo...")

    X = vectorizer.fit_transform(dataset['cleaned_sentence_str'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, dataset['Sentimento_Num'], test_size=0.3, random_state=20)

    param_grid = {'C': [0.01, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'poly', 'sigmoid']}

    clf = svm.SVC(kernel='linear', gamma='scale')

    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1_macro')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    clf = svm.SVC(
        kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'])
    clf.fit(X_train, y_train)

    print("Modelo treinado")
    return clf, vectorizer


def converte_string(texto: str):
    """
        Converte um texto em um vetor de características.

        Parâmetros:
        - texto (str): O texto a ser convertido.

        Retorna:
        - list: Lista de características.
    """
    texto = vectorizer.transform(texto)
    return texto


def modelo():
    df_sentimentos_unicos = carrega_dataset(caminho_arquivo="./data/data.csv")
    df_sentimentos_unicos = analise(dataset=df_sentimentos_unicos)
    clf, vectorizer = treina_modelo(dataset=df_sentimentos_unicos)
    return clf, vectorizer
