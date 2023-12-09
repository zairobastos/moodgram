import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(page_title='Dashboard MOODGRAM - Análise de Frequência',
                   page_icon='./static/img/favicon/favicon.ico', layout='wide', initial_sidebar_state='auto')

st.title('Dashboard MOODGRAM - Análise de Frequência')
st.markdown('#### 4. Análise de Frequência')
df_sentimentos = pd.read_csv("./data/data.csv", sep=",", encoding="utf-8")

st.markdown('##### 4.1. Frequência de Sentimentos')
st.bar_chart(df_sentimentos['Sentiment'].value_counts())
st.write("Quantidade de valores únicos: ", len(
    df_sentimentos['Sentiment'].unique()))

st.markdown('##### 4.2. Frequência de Sentenças')
st.bar_chart(df_sentimentos['Sentence'].value_counts().nlargest(10))
st.write("Quantidade de valores únicos: ", len(
    df_sentimentos['Sentence'].unique()))

st.markdown('#### 5. Análise de Sentenças')
# Adicionando coluna com a quantidade de palavras e caracteres
df_sentimentos['Num Palavras'] = df_sentimentos['Sentence'].str.split().str.len()
df_sentimentos['Num Caracteres'] = df_sentimentos['Sentence'].str.len()

st.markdown('##### 5.1. Quantidade de palavras por sentença')
# Criando histogramas interativos com Plotly Express
fig = px.histogram(df_sentimentos, x='Num Palavras', color='Sentiment', nbins=20,
                   color_discrete_map={'positive': '#86bf91',
                                       'negative': '#fc7b5e', 'neutral': '#75a3a3'},
                   marginal='rug')

# Atualizando layout
fig.update_layout(
    xaxis_title='Quantidade de Palavras',
    yaxis_title='Frequência',
    barmode='group',  # 'overlay' para sobrepor as barras
    bargap=0.1,  # Espaço entre as barras
)

# Adicionando rótulos para cada subplot
fig.update_traces(marker_opacity=0.7)  # Ajustando a opacidade das barras

# Exibindo o gráfico usando Streamlit
st.plotly_chart(fig)


# Exibindo as sentenças com maior quantidade de palavras
st.markdown('##### 5.2. Sentenças com maior quantidade de palavras')
st.write(df_sentimentos.loc[df_sentimentos['Num Palavras'].nlargest(
    10).index][['Sentence', 'Sentiment', 'Num Palavras']])


# Exibindo as sentenças com menor quantidade de palavras
st.markdown('##### 5.3. Sentenças com menor quantidade de palavras')
st.write(df_sentimentos.loc[df_sentimentos['Num Palavras'].nsmallest(
    10).index][['Sentence', 'Sentiment', 'Num Palavras']])

st.markdown('##### 5.4. Estatísticas Descritivas das Sentenças por palavras')
st.write(df_sentimentos['Num Palavras'].describe())

st.markdown('##### 5.5. Quantidade de caracteres por sentença')
fig = px.histogram(df_sentimentos, x='Num Caracteres', color='Sentiment', nbins=20,
                   color_discrete_map={'positive': '#86bf91',
                                       'negative': '#fc7b5e', 'neutral': '#75a3a3'},
                   marginal='rug')

# Atualizando layout
fig.update_layout(
    xaxis_title='Quantidade de Caracteres',
    yaxis_title='Frequência',
    barmode='group',  # 'overlay' para sobrepor as barras
    bargap=0.1,  # Espaço entre as barras
)

# Adicionando rótulos para cada subplot
fig.update_traces(marker_opacity=0.7)  # Ajustando a opacidade das barras

# Exibindo o gráfico usando Streamlit
st.plotly_chart(fig)

st.markdown('##### 5.6. Sentenças com maior quantidade de caracteres')
st.write(df_sentimentos.loc[df_sentimentos['Num Caracteres'].nlargest(
    10).index][['Sentence', 'Sentiment', 'Num Caracteres']])

st.markdown('##### 5.7. Sentenças com menor quantidade de caracteres')
st.write(df_sentimentos.loc[df_sentimentos['Num Caracteres'].nsmallest(
    10).index][['Sentence', 'Sentiment', 'Num Caracteres']])

st.markdown('##### 5.8. Estatísticas Descritivas das Sentenças por caracteres')
st.write(df_sentimentos['Num Caracteres'].describe())
