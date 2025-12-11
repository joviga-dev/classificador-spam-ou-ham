# Classificador de Spam ou Ham

Este projeto implementa um classificador de mensagens utilizando técnicas de Machine Learning para distinguir entre mensagens do tipo **spam** e **ham**. Ele inclui um modelo treinado e uma interface simples desenvolvida com **Streamlit** para testes interativos.

---

## Visão Geral

A aplicação carrega um dataset rotulado, realiza o pré-processamento do texto, treina um modelo de classificação e fornece uma interface para que o usuário informe uma mensagem e receba o resultado da classificação.

---

## Funcionalidades

- Treinamento de modelo a partir do dataset `spam_ham_dataset.csv`
- Pré-processamento das mensagens
- Classificação de novas mensagens usando Streamlit
- Código organizado e de fácil extensão

---

## Pré-requisitos

- Python 3.7 ou superior  
- Pip

---

## Instalação

### Clone o repositório

```bash
git clone https://github.com/joviga-dev/classificador-spam-ou-ham.git
cd classificador-spam-ou-ham
```

### Instale as dependências
```bash
pip install -r requirements.txt
```

--- 

## Como Executar

### Via Streamlit

```bash
streamlit run streamlit_spam_classifier.py
```

Após isso, abra o endereço exibido no terminal, normalmente:

```bash
http://localhost:8501
```

---

## Funcionamento

O texto é pré-processado e convertido para vetores numéricos utilizando técnicas de NLP.

O modelo é treinado com base no dataset rotulado.

A aplicação classifica mensagens novas como spam ou ham.

---

## Avaliação do Modelo

O modelo pode ser avaliado com as seguintes métricas:

- Acurácia  
- Precisão  
- Recall  
- F1-Score  

Essas métricas permitem verificar o desempenho do classificador.

---

## Tecnologias Utilizadas

- Python  
- Streamlit  
- Scikit-learn  
- Pandas e NumPy  
- Técnicas de Machine Learning e Processamento de Linguagem Natural

---

## Dataset

O arquivo `spam_ham_dataset.csv` contém as mensagens rotuladas utilizadas para treinamento e validação do modelo.

