import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud  # wordcloud
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

st.set_page_config(page_title='Dashboard MOODGRAM - Distribuição',
                   page_icon='./static/img/favicon/favicon.ico', layout='wide', initial_sidebar_state='auto')

st.title('Dashboard MOODGRAM - Distribuição')

df_sentimentos = pd.read_csv("./data/data.csv", sep=",", encoding="utf-8")

df_sentimentos_clean = df_sentimentos.drop_duplicates()
df_sentimentos_unicos = df_sentimentos_clean.drop_duplicates(
    subset=['Sentence'], keep=False)

st.markdown(
    '#### 7. Como ficou destribuido os sentimentos após a limpeza dos dados')

df_sentimentos_unicos = df_sentimentos_unicos[['Sentence', 'Sentiment']]
sentimentos_counts = df_sentimentos_unicos['Sentiment'].value_counts()

fig, ax = plt.subplots(figsize=(10, 6))
patches, texts, autotexts = ax.pie(sentimentos_counts, labels=sentimentos_counts.index,
                                   autopct=lambda p: '{:.2f}% ({:.0f})'.format(p, (p/100)*sentimentos_counts.sum()))
ax.set_title('Distribuição dos Sentimentos')
ax.legend(patches, sentimentos_counts.index, loc="best")

# Mostrar o gráfico no Streamlit
st.pyplot(fig)

# Definir o idioma das stopwords
stop_words = set(stopwords.words('english'))


def clean_text(texto):
    texto = re.sub(r'[^\w\s]', '', texto)
    palavras = word_tokenize(texto)
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(
        palavra.lower()) for palavra in palavras if palavra.lower() not in stop_words]
    return text


df_sentimentos_unicos = df_sentimentos_unicos.copy()
df_sentimentos_unicos['cleaned_sentence'] = df_sentimentos_unicos['Sentence'].apply(
    lambda x: clean_text(x))
df_sentimentos_unicos['cleaned_sentence_length'] = df_sentimentos_unicos['cleaned_sentence'].apply(
    lambda x: len(x))


def nuvem_palavras(sentiment):
    df = df_sentimentos_unicos[df_sentimentos_unicos['Sentiment'] == sentiment]
    text = ' '.join(
        palavra for sublist in df.cleaned_sentence for palavra in sublist)
    wordcloud = WordCloud(max_font_size=50, max_words=100,
                          background_color="white").generate(text)

    # Criar um gráfico para a nuvem de palavras
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Nuvem de palavras para o sentimento: {sentiment}")

    # Mostrar o gráfico no Streamlit
    st.pyplot(fig)


# Criar um seletor de sentimento no Streamlit
sentimento_selecionado = st.selectbox(
    "Selecione um sentimento:", df_sentimentos_unicos['Sentiment'].unique())

# Gerar a nuvem de palavras para o sentimento selecionado
nuvem_palavras(sentimento_selecionado)
