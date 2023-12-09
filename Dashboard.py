import streamlit as st
import pandas as pd
import seaborn as sns

st.set_page_config(page_title='Dashboard MOODGRAM',
                   page_icon='./static/img/favicon/favicon.ico', layout='wide', initial_sidebar_state='auto')

st.title('Dashboard MOODGRAM - Home')
st.markdown('### Dataset')

df_sentimentos = pd.read_csv("./data/data.csv", sep=",", encoding="utf-8")
df_sentimentos

st.markdown('### Informações do Dataset')
st.markdown('#### 1. Shape')

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Quantidade de Registros", value=df_sentimentos.shape[0])

with col2:
    st.metric(label="Quantidade de Colunas", value=df_sentimentos.shape[1])

col3, col4 = st.columns(2)
with col3:
    st.markdown('#### 2. Tipos de Dados')
    st.write(df_sentimentos.dtypes)

with col4:
    st.markdown('#### 3. Valores Faltantes')
    st.write(df_sentimentos.isna().sum())
