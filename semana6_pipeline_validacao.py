# Semana 6 — Pipeline + Validação Cruzada
# Dataset: Brazilian E-Commerce Public Dataset by Olist
# Arquivo esperado: olist_orders_dataset.csv (mesma pasta deste arquivo)
#
# OBJETIVO:
# 1. Organizar o processo numa "linha de montagem" (Pipeline): a data entra,
#    as pistas são criadas e o modelo prevê, sempre na mesma ordem.
# 2. Confirmar se o Gradient Boosting é mesmo melhor que o Random Forest,
#    testando em 5 divisões diferentes dos dados (validação cruzada).

# %% Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

plt.rcParams["figure.figsize"] = (11, 5)

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
volume = volume[(volume["pedidos"] > 20)].reset_index(drop=True)

# %% [markdown]
# ## Etapa 1 da linha de montagem: criar as pistas
# Antes, as pistas eram criadas "soltas" no código. Agora viram uma função.
# Ela recebe só a data e devolve todas as pistas prontas.
# Vantagem: qualquer data nova passa exatamente pelo mesmo processo.

# %% Função que cria as pistas a partir da data
DATA_INICIO = volume["data"].min()   # ponto zero da linha do tempo

def criar_pistas(tabela):
    datas = pd.to_datetime(tabela["data"])
    return pd.DataFrame({
        "dia_semana": datas.dt.dayofweek,
        "fim_de_semana": (datas.dt.dayofweek >= 5).astype(int),
        "dia_do_mes": datas.dt.day,
        "mes": datas.dt.month,
        "dias_desde_inicio": (datas - DATA_INICIO).dt.days,
    })

# Agora X é só a data. As pistas são criadas DENTRO do pipeline.
X = volume[["data"]]
y = volume["pedidos"]

# %% [markdown]
# ## A linha de montagem completa (Pipeline)
# Duas etapas, sempre nessa ordem:
#   1. "pistas": transforma a data nas pistas
#   2. "modelo": faz a previsão

# %% Montar os pipelines
def montar_pipeline(modelo):
    return Pipeline([
        ("pistas", FunctionTransformer(criar_pistas)),
        ("modelo", modelo),
    ])

pipelines = {
    "Random Forest": montar_pipeline(
        RandomForestRegressor(max_depth=8, random_state=42)
    ),
    "Gradient Boosting": montar_pipeline(
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                  max_depth=3, random_state=42)
    ),
}

# %% [markdown]
# ## Validação cruzada: 5 rodadas
# KFold divide os dados em 5 pedaços. Em cada rodada, um pedaço vira teste
# e os outros 4 viram treino. No fim, tiramos a média do erro.
#
# Detalhe técnico: o scikit-learn devolve o erro com sinal negativo
# ("neg_mean_absolute_error"). Por isso multiplicamos por -1.

# %% Rodar a validação cruzada
divisao = KFold(n_splits=5, shuffle=True, random_state=42)

medias = {}
for nome, pipe in pipelines.items():
    notas = cross_val_score(pipe, X, y, cv=divisao,
                            scoring="neg_mean_absolute_error")
    erros = -notas
    medias[nome] = erros.mean()
    print(f"\n{nome}")
    for i, erro in enumerate(erros, start=1):
        print(f"  Rodada {i}: erro de {erro:.1f} pedidos")
    print(f"  MÉDIA: {erros.mean():.1f} pedidos")

campeao = min(medias, key=medias.get)
print(f"\n>> Melhor na média das 5 rodadas: {campeao} ({medias[campeao]:.1f} pedidos)")

# %% [markdown]
# ## Usando o pipeline campeão para prever
# Treinamos o campeão com TODOS os dados e pedimos a previsão da semana
# seguinte ao fim da base. Repare: entra só a data, e o pipeline faz o resto.

# %% Treinar o campeão e prever a semana seguinte
pipeline_final = pipelines[campeao]
pipeline_final.fit(X, y)

ultima_data = volume["data"].max()
proxima_semana = pd.DataFrame({
    "data": pd.date_range(ultima_data + pd.Timedelta(days=1), periods=7)
})
proxima_semana["previsao"] = pipeline_final.predict(proxima_semana).round()

print("\nPrevisão para a semana seguinte:")
print(proxima_semana.to_string(index=False))

# %% Gráfico: erro médio de cada modelo na validação cruzada
plt.bar(list(medias.keys()), list(medias.values()))
plt.ylabel("Erro médio nas 5 rodadas (pedidos)")
plt.title("Validação cruzada: Random Forest x Gradient Boosting")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Minha conclusão da Semana 6
#
# Na média das 5 rodadas, o Gradient Boosting se saiu melhor, com erro de 21,7
# pedidos, contra cerca de 23 do Random Forest. Isso confirmou a escolha da
# Semana 5: o Gradient Boosting continua sendo o melhor modelo para o projeto.
#
# O erro médio ficou maior que os 17,8 da Semana 5, mas isso não quer dizer que
# o modelo piorou. Naquela semana o teste foi feito com uma divisão só, e ela
# pegou dias mais fáceis. A média das 5 rodadas mostra o erro mais real.
#
# O erro variou bastante de uma rodada pra outra, de 17,8 até 30,3 pedidos.
# Isso mostra que testar com uma divisão só pode enganar, e que alguns dias são
# bem mais difíceis de prever, como a Black Friday. Uma próxima melhora seria
# dar ao modelo uma pista de datas promocionais.
