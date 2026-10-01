# Semana 5 — Gradient Boosting x Random Forest
# Dataset: Brazilian E-Commerce Public Dataset by Olist
# Arquivo esperado: olist_orders_dataset.csv (mesma pasta deste arquivo)
#
# OBJETIVO: colocar o Gradient Boosting pra disputar com o campeão da Semana 4,
# o Random Forest (erro de 19,7 pedidos). Mesmas pistas, mesma divisão
# treino/teste, pra comparação ser justa.

# %% Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor          # campeão da Semana 4
from sklearn.ensemble import GradientBoostingRegressor      # NOVO: boosting
from sklearn.metrics import mean_absolute_error

plt.rcParams["figure.figsize"] = (11, 5)

# %% Carregar e preparar os dados (igual à Semana 4)
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

# %% Features (as mesmas da versão melhorada da Semana 4)
volume["dia_semana"] = volume["data"].dt.dayofweek
volume["fim_de_semana"] = (volume["dia_semana"] >= 5).astype(int)
volume["dia_do_mes"] = volume["data"].dt.day
volume["mes"] = volume["data"].dt.month
volume["dias_desde_inicio"] = (volume["data"] - volume["data"].min()).dt.days

X = volume[["dia_semana", "fim_de_semana", "dia_do_mes", "mes", "dias_desde_inicio"]]
y = volume["pedidos"]

# %% Separar treino e teste (mesmo random_state = mesma divisão da Semana 4)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% [markdown]
# ## As configurações do Gradient Boosting
# - n_estimators=200: quantas árvores em sequência (quantos "funcionários")
# - learning_rate=0.05: o quanto cada árvore corrige do erro anterior.
#   Valor baixo = correções pequenas e cuidadosas, menos chance de decorar.
# - max_depth=3: árvores pequenas. No boosting, cada árvore só precisa
#   corrigir um pedaço do erro, então não precisa ser grande.

# %% Treinar e avaliar
modelos = {
    "Random Forest": RandomForestRegressor(max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    ),
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
print(f"\nBaseline (prever sempre a média): {mae_baseline:.1f} pedidos")

so_modelos = {k: v for k, v in resultados.items() if k != "Baseline (média)"}
melhor = min(so_modelos, key=so_modelos.get)
print(f"\n>> Melhor modelo: {melhor}, com erro de {so_modelos[melhor]:.1f} pedidos")

# %% Gráfico de comparação
plt.bar(list(resultados.keys()), list(resultados.values()))
plt.ylabel("Erro médio (MAE em pedidos)")
plt.title("Random Forest x Gradient Boosting")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Minha conclusão da Semana 5 
#
# O Gradient Boosting teve o melhor resultado, com erro médio de 17,8 pedidos,
# pouco à frente do Random Forest (19,7). A diferença foi pequena, cerca de 2
# pedidos, mas ele foi melhor nas mesmas condições de teste. Por isso, escolho
# o Gradient Boosting para o projeto final. Como o teste tem só 120 dias, essa
# vantagem ainda precisa ser confirmada com validação cruzada, testando os
# dois modelos em várias divisões diferentes dos dados.