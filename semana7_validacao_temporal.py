# Semana 7 — Validação temporal ("treinar no passado, testar no futuro")
# Dataset: Brazilian E-Commerce Public Dataset by Olist
# Arquivo esperado: olist_orders_dataset.csv (mesma pasta deste arquivo)
#
# POR QUE ESTA SEMANA EXISTE:
# Na Semana 6, o KFold(shuffle=True) embaralhou os dias. Assim, o modelo treinava
# com dias de DEPOIS do dia que ele estava prevendo, algo impossível na vida real.
# Aqui cada rodada treina só com o passado e testa no período seguinte
# (TimeSeriesSplit), como seria numa operação de verdade.
#
# Saídas:
#   - erros de cada modelo em cada rodada (no terminal)
#   - imagens/serie_diaria.png        (volume diário + Black Friday)
#   - imagens/previsto_vs_real.png    (última rodada: previsto x real)

# %% Importar bibliotecas
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

plt.rcParams["figure.figsize"] = (11, 5)
os.makedirs("imagens", exist_ok=True)

# %% Carregar os dados e montar o volume diário (igual às semanas anteriores)
df = pd.read_csv("olist_orders_dataset.csv")
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

volume = (
    df.set_index("order_purchase_timestamp")
      .resample("D")
      .size()
      .reset_index()
)
volume.columns = ["data", "pedidos"]
volume = volume[(volume["pedidos"] > 20)].sort_values("data").reset_index(drop=True)

DATA_INICIO = volume["data"].min()


def criar_pistas(tabela):
    datas = pd.to_datetime(tabela["data"])
    return pd.DataFrame({
        "dia_semana": datas.dt.dayofweek,
        "fim_de_semana": (datas.dt.dayofweek >= 5).astype(int),
        "dia_do_mes": datas.dt.day,
        "mes": datas.dt.month,
        "dias_desde_inicio": (datas - DATA_INICIO).dt.days,
    })


def montar_pipeline(modelo):
    return Pipeline([
        ("pistas", FunctionTransformer(criar_pistas)),
        ("modelo", modelo),
    ])


modelos = {
    "Random Forest": lambda: montar_pipeline(
        RandomForestRegressor(max_depth=8, random_state=42)),
    "Gradient Boosting": lambda: montar_pipeline(
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                  max_depth=3, random_state=42)),
}

X = volume[["data"]]
y = volume["pedidos"]

# %% [markdown]
# ## Baselines honestos para série temporal
# - "Média do passado": prevê a média de todos os dias do treino.
# - "Semana passada": prevê para cada dia o volume do mesmo dia 7 dias antes.
#   É o baseline mais difícil de bater, e o que alguém da operação usaria.

# %% Validação temporal: 5 rodadas, cada uma testando ~60 dias à frente
divisao = TimeSeriesSplit(n_splits=5, test_size=60)
semana_passada = y.shift(7)

erros = {nome: [] for nome in ["Média do passado", "Semana passada", *modelos]}
ultima_rodada = None

for rodada, (idx_treino, idx_teste) in enumerate(divisao.split(X), start=1):
    X_tr, X_te = X.iloc[idx_treino], X.iloc[idx_teste]
    y_tr, y_te = y.iloc[idx_treino], y.iloc[idx_teste]

    erros["Média do passado"].append(
        mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean())))
    erros["Semana passada"].append(
        mean_absolute_error(y_te, semana_passada.iloc[idx_teste]))

    previsoes = {}
    for nome, criar in modelos.items():
        modelo = criar().fit(X_tr, y_tr)
        previsoes[nome] = modelo.predict(X_te)
        erros[nome].append(mean_absolute_error(y_te, previsoes[nome]))

    ini, fim = volume["data"].iloc[idx_teste[0]], volume["data"].iloc[idx_teste[-1]]
    print(f"Rodada {rodada}: treino até {volume['data'].iloc[idx_treino[-1]]:%d/%m/%Y}"
          f" | teste {ini:%d/%m/%Y} a {fim:%d/%m/%Y}")
    ultima_rodada = (idx_teste, previsoes)

# %% Resumo
print("\nErro médio (MAE) por modelo, em pedidos por dia:")
resumo = pd.DataFrame(erros, index=[f"Rodada {i}" for i in range(1, 6)]).T
resumo["Média"] = resumo.mean(axis=1)
print(resumo.round(1).sort_values("Média").to_string())

# %% Gráfico 1: série diária com a Black Friday marcada
pico = volume.loc[volume["pedidos"].idxmax()]
plt.plot(volume["data"], volume["pedidos"], linewidth=1)
plt.annotate(f"Black Friday\n{int(pico['pedidos'])} pedidos",
             xy=(pico["data"], pico["pedidos"]),
             xytext=(pico["data"] + pd.Timedelta(days=40), pico["pedidos"] * 0.85),
             arrowprops={"arrowstyle": "->"})
plt.title("Volume diário de pedidos — Olist")
plt.ylabel("Pedidos por dia")
plt.tight_layout()
plt.savefig("imagens/serie_diaria.png", dpi=150)
plt.close()

# %% Gráfico 2: última rodada, previsto x real
idx_teste, previsoes = ultima_rodada
melhor = min(modelos, key=lambda n: erros[n][-1])
datas_teste = volume["data"].iloc[idx_teste]
plt.plot(datas_teste, y.iloc[idx_teste], label="Real", linewidth=2)
plt.plot(datas_teste, previsoes[melhor], label=f"Previsto ({melhor})", linestyle="--")
plt.plot(datas_teste, semana_passada.iloc[idx_teste], label="Semana passada (baseline)",
         alpha=0.6)
plt.title("Últimos 60 dias: previsão sem ver o futuro")
plt.ylabel("Pedidos por dia")
plt.legend()
plt.tight_layout()
plt.savefig("imagens/previsto_vs_real.png", dpi=150)
plt.close()
print("\nGráficos salvos em imagens/")
