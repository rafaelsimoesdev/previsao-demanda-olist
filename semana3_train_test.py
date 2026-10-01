# Semana 3 — Train/Test Split na prática
# Dataset: Brazilian E-Commerce Public Dataset by Olist
# Arquivo esperado: olist_orders_dataset.csv (mesma pasta deste arquivo)
#
# OBJETIVO: ver na prática a ideia da Semana 3 — separar os dados em treino e
# teste, treinar um modelo simples e dar uma NOTA a ele usando dados que ele
# nunca viu. Ainda é um modelo básico de propósito; o foco é o método, não o
# resultado.

# %% [markdown]
# # Semana 3 — Separando treino e teste
# **Autor:** Rafael
# **Ideia:** treinar um modelo para prever o volume de pedidos e medir o erro
# dele em dados que ele nunca viu.

# %% Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ferramentas novas do scikit-learn (já vieram instaladas com o pacote):
from sklearn.model_selection import train_test_split   # separa treino/teste
from sklearn.linear_model import LinearRegression      # um modelo simples
from sklearn.metrics import mean_absolute_error        # a "nota" (MAE)

plt.rcParams["figure.figsize"] = (11, 5)

# %% Carregar e preparar os dados
df = pd.read_csv("olist_orders_dataset.csv")
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

# Vamos montar uma tabela de volume por dia (como fizemos na Semana 1)
volume = (
    df.set_index("order_purchase_timestamp")
      .resample("D")
      .size()
      .reset_index()
)
volume.columns = ["data", "pedidos"]

# Cortamos as bordas do dataset (dias quase vazios do começo/fim que vimos
# na Semana 1), pra não atrapalhar. Ficamos com o miolo com dados de verdade.
volume = volume[(volume["pedidos"] > 20)].reset_index(drop=True)

print("Total de dias na base:", len(volume))
volume.head()

# %% [markdown]
# ## Criar as "pistas" (features) para o modelo
# Um modelo não entende uma data direto. Precisamos transformar a data em
# números que ele possa usar como pista. Vamos dar a ele duas pistas simples:
# - dia da semana (0 = segunda ... 6 = domingo)
# - se é fim de semana ou não
# Mais pra frente você aprende a criar pistas melhores. Por ora, o foco é o método.

# %% Montar as features
volume["dia_semana"] = volume["data"].dt.dayofweek
volume["fim_de_semana"] = (volume["dia_semana"] >= 5).astype(int)

# X = as pistas (o que o modelo usa pra chutar)
# y = a resposta certa (o que ele tenta prever)
X = volume[["dia_semana", "fim_de_semana"]]
y = volume["pedidos"]

# %% [markdown]
# ## O passo central da Semana 3: separar treino e teste
# train_test_split divide os dados. test_size=0.2 quer dizer:
# 80% pra treinar, 20% pra testar (o modelo NUNCA vê esses 20% no treino).
# random_state=42 só garante que a divisão seja sempre a mesma toda vez que roda.

# %% Separar
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dias para treinar:", len(X_treino))
print("Dias para testar: ", len(X_teste))

# %% [markdown]
# ## Treinar o modelo
# .fit() é o "aprender": o modelo olha as pistas do treino e a resposta certa.

# %% Treinar
modelo = LinearRegression()
modelo.fit(X_treino, y_treino)
print("Modelo treinado!")

# %% [markdown]
# ## Testar e dar a NOTA
# Agora pedimos ao modelo que preveja os dias de teste (que ele nunca viu)
# e comparamos com a resposta certa usando o MAE (erro médio em nº de pedidos).

# %% Prever e medir
previsoes = modelo.predict(X_teste)
mae = mean_absolute_error(y_teste, previsoes)

print(f"MAE (erro médio): {mae:.1f} pedidos")
print("Ou seja: em média, o modelo erra cerca de", round(mae), "pedidos por dia.")

# %% [markdown]
# ## Comparar com um "baseline" (baseline)
# LEMBRE da regra de ouro: todo modelo só vale comparado a um baseline.
# O chute mais simples possível é "sempre prever a média de pedidos".
# Se nosso modelo não ganhar da média, ele não está ajudando em nada.

# %% Baseline
chute_medio = np.full(len(y_teste), y_treino.mean())  # sempre a média
mae_baseline = mean_absolute_error(y_teste, chute_medio)

print(f"MAE do modelo:   {mae:.1f} pedidos")
print(f"MAE do baseline: {mae_baseline:.1f} pedidos")

if mae < mae_baseline:
    print(">> O modelo ganhou do baseline. Ele aprendeu algo útil.")
else:
    print(">> O modelo NÃO ganhou do baseline. Precisa de pistas melhores.")

# %% [markdown]
# ## Minha conclusão da Semana 3
#
# O modelo errou em média 60,5 pedidos por dia. Ele ganhou do baseline
# (prever sempre a média), que errava 63,5, mas ganhou por pouco, só 3 pedidos
# de diferença. Isso mostra que o método funciona (treinar, testar e comparar),
# mas o modelo ainda é fraco, porque só demos duas pistas simples pra ele (dia
# da semana e fim de semana). Com pistas melhores, o erro deve cair bastante.
# Esse é o ponto de partida pra melhorar nas próximas fases.
