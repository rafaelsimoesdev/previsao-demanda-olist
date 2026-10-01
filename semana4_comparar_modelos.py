# Semana 4 — Comparando modelos (Regressão Linear x Árvore x Random Forest)
# Dataset: Brazilian E-Commerce Public Dataset by Olist
# Arquivo esperado: olist_orders_dataset.csv (mesma pasta deste arquivo)
#
# OBJETIVO: rodar TRÊS modelos no mesmo problema da Semana 3 e comparar o erro
# (MAE) de cada um contra o baseline. Escolher o melhor por NÚMERO, não por
# achismo. Essa é a habilidade central de quem faz ML.

# %% [markdown]
# # Semana 4 — Qual modelo prevê melhor?
# **Autor:** Rafael
# **Ideia:** treinar 3 modelos diferentes e comparar qual erra menos ao prever
# o volume de pedidos.

# %% Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression          # modelo da Semana 3
from sklearn.tree import DecisionTreeRegressor             # NOVO: árvore de decisão
from sklearn.ensemble import RandomForestRegressor         # NOVO: floresta
from sklearn.metrics import mean_absolute_error

plt.rcParams["figure.figsize"] = (11, 5)

# %% Carregar e preparar os dados (igual à Semana 3)
df = pd.read_csv("olist_orders_dataset.csv")
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

volume = (
    df.set_index("order_purchase_timestamp")
      .resample("D")
      .size()
      .reset_index()
)
volume.columns = ["data", "pedidos"]
volume = volume[(volume["pedidos"] > 20)].reset_index(drop=True)

# %% [markdown]
# ## Melhorando as pistas (features)
# Na Semana 3 demos só 2 pistas e o modelo aprendeu pouco. Aqui damos MAIS
# pistas pro modelo, pra ele ter mais com o que trabalhar:
# - dia da semana
# - se é fim de semana
# - dia do mês (começo/meio/fim importam na operação)
# - mês
# Dar pistas melhores é o que mais faz o erro cair. Isso se chama feature engineering.

# %% Montar as features
volume["dia_semana"] = volume["data"].dt.dayofweek
volume["fim_de_semana"] = (volume["dia_semana"] >= 5).astype(int)
volume["dia_do_mes"] = volume["data"].dt.day
volume["mes"] = volume["data"].dt.month

X = volume[["dia_semana", "fim_de_semana", "dia_do_mes", "mes"]]
y = volume["pedidos"]

# %% Separar treino e teste (igual à Semana 3)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Dias para treinar:", len(X_treino))
print("Dias para testar: ", len(X_teste))

# %% [markdown]
# ## Treinar os 3 modelos de uma vez
# Vamos guardar os modelos num dicionário e treinar todos num laço (loop).
# Assim o código fica organizado e fácil de comparar.

# %% Treinar e avaliar todos
modelos = {
    "Regressão Linear": LinearRegression(),
    "Árvore de Decisão": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
}

resultados = {}
for nome, modelo in modelos.items():
    modelo.fit(X_treino, y_treino)                    # treina
    previsoes = modelo.predict(X_teste)               # prevê no teste
    mae = mean_absolute_error(y_teste, previsoes)     # mede o erro
    resultados[nome] = mae
    print(f"{nome}: erra em média {mae:.1f} pedidos")

# %% Baseline (o baseline: sempre a média)
mae_baseline = mean_absolute_error(y_teste, np.full(len(y_teste), y_treino.mean()))
resultados["Baseline (média)"] = mae_baseline
print(f"\nBaseline (baseline): {mae_baseline:.1f} pedidos")

# %% [markdown]
# ## Comparar tudo num gráfico
# Um gráfico de barras deixa na cara qual modelo erra menos.
# Quanto MENOR a barra, melhor (menos erro).

# %% Gráfico de comparação
nomes = list(resultados.keys())
erros = list(resultados.values())

plt.bar(nomes, erros)
plt.ylabel("Erro médio (MAE em pedidos)")
plt.title("Qual modelo erra menos?")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Descobrir o campeão automaticamente (ignorando o baseline)
so_modelos = {k: v for k, v in resultados.items() if k != "Baseline (média)"}
melhor = min(so_modelos, key=so_modelos.get)
print(f"\n>> Melhor modelo: {melhor}, com erro de {so_modelos[melhor]:.1f} pedidos")

# %% [markdown]
# ## Minha conclusão da Semana 4 (primeira versão)
#
# Nesta primeira tentativa, a regressão linear foi a melhor, errando em média
# 61,3 pedidos. A árvore de decisão (96,3) e o Random Forest (74,8) ficaram
# piores até que o baseline (prever sempre a média), que errava 63,5.
# Dois motivos explicam isso: faltava a pista de tendência (o volume cresce
# ao longo do tempo e o modelo não sabia em que ponto do período estava), e a
# árvore decorou os dados de treino (overfitting), criando regras para dias
# específicos. Esses dois problemas foram corrigidos no arquivo
# semana4b_modelos_melhorados.py.
