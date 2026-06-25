import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Configuração da Página
st.set_page_config(page_title="JobMatch AI", page_icon="🚀", layout="wide")

# 2. Função de Limpeza
def limpar_texto(texto):
    if pd.isna(texto): return ""
    texto = str(texto).lower()
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# 3. Carregar os modelos (com cache para não recarregar a cada clique)
@st.cache_resource
def carregar_dados():
    modelo_fit = joblib.load('modelo_fit.pkl')
    tfidf_clf = joblib.load('tfidf_clf.pkl')
    regressor = joblib.load('modelo_salario.pkl')
    tfidf_reg = joblib.load('tfidf_reg.pkl')
    df_vagas = pd.read_csv('vagas_app.csv')
    return modelo_fit, tfidf_clf, regressor, tfidf_reg, df_vagas

# 4. Interface Web
st.title("🎯 JobMatch AI")
st.subheader("O seu Sistema de Recomendação Inteligente de Vagas")
st.write("Insira o seu perfil profissional abaixo e descubra as melhores vagas para si, com análise de Fit e estimativa salarial.")

try:
    modelo_fit, tfidf_clf, regressor, tfidf_reg, df_vagas = carregar_dados()

    # Encontrar as colunas corretas dinamicamente
    cols_vagas = [c.lower() for c in df_vagas.columns]
    col_vaga_desc = 'description' if 'description' in cols_vagas else df_vagas.columns[1]
    col_vaga_title = 'title' if 'title' in cols_vagas else df_vagas.columns[0]

except FileNotFoundError:
    st.error("Modelos não encontrados. Por favor, execute as células de treino primeiro.")
    st.stop()

# Caixa de texto para o utilizador
curriculo = st.text_area("Cole o seu Resumo Profissional / Currículo aqui:", height=150)

# Botão Mágico
if st.button("Analisar o Meu Perfil 🚀"):
    if curriculo.strip() == "":
        st.warning("Por favor, escreva alguma coisa no currículo!")
    else:
        with st.spinner('A IA está a analisar o seu perfil face ao mercado...'):
            cv_limpo = limpar_texto(curriculo)

            # Limpar vagas
            vagas_limpas = df_vagas[col_vaga_desc].apply(limpar_texto)

            # Calcular Similaridade
            tfidf_rec = TfidfVectorizer(stop_words='english', max_features=5000)
            textos_totais = [cv_limpo] + vagas_limpas.tolist()
            matriz_tfidf = tfidf_rec.fit_transform(textos_totais)

            scores = cosine_similarity(matriz_tfidf[0], matriz_tfidf[1:]).flatten()
            top_indices = scores.argsort()[-5:][::-1] # Top 5

            st.markdown("---")
            st.header("🏆 As Suas Top 5 Vagas")

            # Mostrar os resultados de forma elegante
            for i, idx in enumerate(top_indices, 1):
                vaga = df_vagas.iloc[idx]
                score_match = scores[idx] * 100
                vaga_desc_limpa = vagas_limpas.iloc[idx]

                # Prever Fit/No Fit
                texto_fit = cv_limpo + " " + vaga_desc_limpa
                vetor_fit = tfidf_clf.transform([texto_fit])
                is_fit = modelo_fit.predict(vetor_fit)[0]

                # Prever Salário
                vetor_salario = tfidf_reg.transform([vaga_desc_limpa])
                salario_estimado = regressor.predict(vetor_salario)[0]

                # Formatação Visual
                status_fit = "✅ FIT EXCELENTE" if is_fit == 1 else "⚠️ FIT PARCIAL (Requer Skills Adicionais)"
                cor_fit = "green" if is_fit == 1 else "orange"

                with st.expander(f"{i}. {vaga[col_vaga_title]}  |  Match: {score_match:.1f}%"):
                    st.markdown(f"**Análise da IA:** :{cor_fit}[{status_fit}]")
                    st.write(f"**Salário Estimado:** $ {salario_estimado:,.2f} / ano")
                    st.write(f"**Descrição da Vaga:** {str(vaga[col_vaga_desc])[:600]}...")
