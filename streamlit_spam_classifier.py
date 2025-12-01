import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
import warnings
warnings.filterwarnings("ignore")

st.title("Classificação de Emails: Spam vs Ham")

# https://www.kaggle.com/datasets/venky73/spam-mails-dataset <-- LINK DO DATASET UTILIZADO
# João Vitor Garcia RA 191022756

class ParaDenso(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None): 
        return self
    def transform(self, X, y=None): 
        return X.toarray()


if "treinado" not in st.session_state:
    st.session_state.treinado = False

if "modelos" not in st.session_state:
    st.session_state.modelos = None


def carregar_dataset():
    try:
        df = pd.read_csv("spam_ham_dataset.csv", encoding="utf-8")
    except:
        return None

    df.columns = [c.strip().lstrip("#").strip() for c in df.columns]

    if "label" not in df.columns or "text" not in df.columns:
        return None

    if "label_num" not in df.columns:
        df["label_num"] = df["label"].apply(lambda x: 1 if str(x).lower() == "spam" else 0)

    return df.dropna(subset=["text", "label"])


df = carregar_dataset()
if df is None:
    st.error("Erro ao carregar spam_ham_dataset.csv.")
    st.stop()

st.write("Exemplo do dataset:")
st.dataframe(df.sample(10))
st.write(f"Total de registros: **{len(df)}**")


st.sidebar.header("Configurações do Treino")
tamanho_teste = st.sidebar.slider("Tamanho do teste (%)", 5, 50, 20)
max_profundidade = st.sidebar.slider("Profundidade da Árvore", 1, 50, 10)
num_arvores = st.sidebar.slider("Árvores no Random Forest", 10, 500, 100, step=10)
k_vizinhos = st.sidebar.slider("k do k-NN", 1, 50, 5)


x = df["text"].astype(str)
y = df["label_num"].astype(int)

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=tamanho_teste / 100, stratify=y
)

vetor = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=2)

modelo_dt = Pipeline([
    ("tfidf", vetor),
    ("clf", DecisionTreeClassifier(max_depth=max_profundidade))
])

modelo_rf = Pipeline([
    ("tfidf", vetor),
    ("clf", RandomForestClassifier(n_estimators=num_arvores))
])

modelo_knn = Pipeline([
    ("tfidf", vetor),
    ("denso", ParaDenso()),
    ("escala", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=k_vizinhos))
])

modelos_disponiveis = {
    "Árvore de Decisão": modelo_dt,
    "Random Forest": modelo_rf,
    "k-NN": modelo_knn
}


if st.button("Treinar modelos"):
    modelos_treinados = {
        nome: m.fit(x_treino, y_treino) for nome, m in modelos_disponiveis.items()
    }

    st.session_state.modelos = modelos_treinados
    st.session_state.treinado = True

    resultados = []
    for nome, m in modelos_treinados.items():
        pred = m.predict(x_teste)
        resultados.append({
            "Modelo": nome,
            "Acurácia": accuracy_score(y_teste, pred),
            "Precisão": precision_score(y_teste, pred),
            "Recall": recall_score(y_teste, pred),
            "F1": f1_score(y_teste, pred)
        })

    tabela = pd.DataFrame(resultados).set_index("Modelo")

    st.subheader("Resultados dos Modelos")
    st.write("Precisão = % de previsões de spam corretas")
    st.write("Recall = % de spams identificados")
    st.write("Acurácia = % total de acertos")
    st.write("F1 = média harmônica entre precisão e recall")

    st.dataframe(tabela.style.format("{:.4f}"))
    st.bar_chart(tabela["Acurácia"])

    melhor = tabela["F1"].idxmax()
    st.success(f"Melhor modelo: {melhor}")


if st.session_state.treinado:

    st.subheader("Teste Manual")
    texto = st.text_area("Digite um email para classificar:")

    if st.button("Classificar texto"):
        if texto.strip() == "":
            st.warning("Digite um texto válido.")
        else:
            st.write("### Resultados dos 3 modelos:")

            for nome, modelo in st.session_state.modelos.items():
                pred = modelo.predict([texto])[0]
                resultado = "spam" if pred == 1 else "ham"
                st.write(f"**{nome}:** {resultado}")
