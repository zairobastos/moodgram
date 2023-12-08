import streamlit as st
import pandas as pd
st.set_page_config(page_title='Dashboard MOODGRAM - Problemas',
                   page_icon='./static/img/favicon/favicon.ico', layout='wide', initial_sidebar_state='auto')

st.title('Dashboard MOODGRAM - Problemas')
df_sentimentos = pd.read_csv("./data/data.csv", sep=",", encoding="utf-8")

st.markdown('#### 6. Problemas Encontrados')
valores_repetidos = df_sentimentos[df_sentimentos.duplicated(keep=False)].groupby(
    ['Sentence', 'Sentiment']).size().reset_index(name='Quantidade de Repetições')
st.markdown('##### 6.1 Problema 1 -  Registros Repitidos')
st.write(
    valores_repetidos[['Sentence', 'Sentiment', 'Quantidade de Repetições']])
st.metric(label="Quantidade Total de Valores Repetidos",
          value=valores_repetidos['Quantidade de Repetições'].sum())

df_sentimentos_clean = df_sentimentos.drop_duplicates()
st.markdown(
    '##### 6.2. Solução para o problema 1 -  Remorção dos registros repetidos')
st.write(df_sentimentos_clean[['Sentence', 'Sentiment']])
st.metric(label="Quantidade de Registros",
          value=df_sentimentos_clean.shape[0])

st.markdown(
    '##### 6.3. Problema 2 - Sentença repetida com sentimentos diferentes')
df_sentimentos_duplicados = df_sentimentos_clean[df_sentimentos_clean.duplicated(
    subset=['Sentence'], keep=False)]
df_sentimentos_duplicados_ordenados = df_sentimentos_duplicados.sort_values(
    by='Sentence')
st.write(df_sentimentos_duplicados_ordenados[['Sentence', 'Sentiment']])
st.write("Quantidade de sentimentos: ",
         df_sentimentos_duplicados_ordenados['Sentiment'].value_counts())

st.markdown('##### 6.4. Solução para o problema 2 - Remoção dos registros')
df_sentimentos_unicos = df_sentimentos_clean.drop_duplicates(
    subset=['Sentence'], keep=False)
st.write(df_sentimentos_unicos[['Sentence', 'Sentiment']])
st.metric(label="Quantidade de Registros",
          value=df_sentimentos_unicos.shape[0])
