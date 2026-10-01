# Semana 4 (parte 2) — Consertando os modelos
# Dataset: Brazilian E-Commerce Public Dataset by Olist
# Arquivo esperado: olist_orders_dataset.csv (mesma pasta deste arquivo)
#
# Na primeira versão, a Árvore e o Random Forest ficaram PIORES que o chute
# burro. Dois motivos:
#   1. Faltava a pista de TENDÊNCIA (o volume cresce ao longo do tempo)
#   2. A árvore decorou o treino (overfitting)
# Aqui corrigimos as duas coisas e comparamos de novo.

# %% Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

plt.rcParams["figure.figsize"] = (11, 5)

# %% Carregar e preparar os dados (igual antes)
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

# %% Montar as features
volume["dia_semana"] = volume["data"].dt.dayofweek
volume["fim_de_semana"] = (volume["dia_semana"] >= 5).astype(int)
volume["dia_do_mes"] = volume["data"].dt.day
volume["mes"] = volume["data"].dt.month

# CORREÇÃO 1: a pista de tendência.
# Conta quantos dias se passaram desde o primeiro dia da base.
# Assim o modelo sabe se o dia é do começo (pouco volume) ou do fim (muito volume).
volume["dias_desde_inicio"] = (volume["data"] - volume["data"].min()).dt.days

X = volume[["dia_semana", "fim_de_semana", "dia_do_mes", "mes", "dias_desde_inicio"]]
y = volume["pedidos"]

# %% Separar treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% Treinar e avaliar
# CORREÇÃO 2: max_depth limita quantas perguntas seguidas a árvore pode fazer.
# Com poucas perguntas, ela não consegue criar uma regra pra cada dia,
# então é obrigada a aprender o padrão geral em vez de decorar.
modelos = {
    "Regressão Linear": LinearRegression(),
    "Árvore de Decisão": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(max_depth=8, random_state=42),
}

resultados = {}
for nome, modelo in modelos.items():
    modelo.fit(X_treino, y_treino)
    previsoes = modelo.predict(X_teste)
    mae = mean_absolute_error(y_teste, previsoes)
    resultados[nome] = mae
    print(f"{nome}: erra em média {mae:.1f} pedidos")

mae_baseline = mean_absolute_error(y_teste, np.full(len(y_teste), y_treino.mean()))
resultados["Baseline (média)"] = mae_baseline
print(f"\nBaseline (baseline): {mae_baseline:.1f} pedidos")

so_modelos = {k: v for k, v in resultados.items() if k != "Baseline (média)"}
melhor = min(so_modelos, key=so_modelos.get)
print(f"\n>> Melhor modelo: {melhor}, com erro de {so_modelos[melhor]:.1f} pedidos")

# %% Gráfico de comparação
plt.bar(list(resultados.keys()), list(resultados.values()))
plt.ylabel("Erro médio (MAE em pedidos)")
plt.title("Depois das correções: qual modelo erra menos?")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Minha conclusão da Semana 4 (versão corrigida) 
#
# O melhor modelo foi o Random Forest, que caiu de 74,8 pedidos de erro na
# primeira versão para 19,7 na segunda, errando cerca de 3 vezes menos que o
# baseline (63,5). Todos os modelos tiveram uma melhora significativa, e o
# maior responsável foi a pista de tendência (dias desde o início): só com ela,
# a regressão linear já caiu de 61,3 para 29,3. Limitar a profundidade das
# árvores também ajudou, porque impediu que elas decorassem os dados de treino.
