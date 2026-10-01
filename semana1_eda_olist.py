# Semana 1 — Notebook de Exploração (EDA inicial)
# Dataset: Brazilian E-Commerce Public Dataset by Olist (Kaggle)
# Arquivo esperado: olist_orders_dataset.csv
#
# COMO USAR:
# 1. Baixe o dataset no Kaggle: "Brazilian E-Commerce Public Dataset by Olist"
# 2. Coloque o arquivo olist_orders_dataset.csv na MESMA pasta deste notebook
# 3. No VS Code, abra este arquivo e rode célula por célula (ou converta em .ipynb)
#
# Objetivo da Semana 1: explorar os dados, entender volume ao longo do tempo,
# e escrever 3 observações. NÃO é modelar ainda — é conhecer o terreno.

# %% [markdown]
# # Semana 1 — Exploração de Volume de Pedidos (Olist)
# **Autor:** Rafael
# **Objetivo:** entender o volume de pedidos ao longo do tempo, um primeiro passo
# rumo à previsão de demanda (Projeto 1).

# %% Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Deixa os gráficos com um tamanho confortável
plt.rcParams["figure.figsize"] = (11, 5)

# %% Carregar os dados
# Se o arquivo estiver em outra pasta, ajuste o caminho abaixo.
df = pd.read_csv("olist_orders_dataset.csv")

# Primeira olhada: quantas linhas/colunas e como são os dados
print("Formato (linhas, colunas):", df.shape)
df.head()

# %% Conhecer as colunas e os tipos
# .info() mostra o tipo de cada coluna e quantos valores não-nulos existem.
df.info()

# %% Converter as colunas de data
# No CSV, datas vêm como texto. Precisamos convertê-las para o tipo datetime
# para conseguir agrupar por dia/mês. Esta é a coluna que nos interessa:
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

# Confirmando o período coberto pelos dados
print("Primeiro pedido:", df["order_purchase_timestamp"].min())
print("Último pedido: ", df["order_purchase_timestamp"].max())

# %% [markdown]
# ## Verificação de qualidade dos dados
# Antes de analisar, todo bom trabalho checa nulos e duplicatas.
# Isso é feijão-com-arroz de analista, e você já conhece.

# %% Checar valores nulos e duplicatas
print("Valores nulos por coluna:")
print(df.isnull().sum())
print("\nLinhas duplicadas:", df.duplicated().sum())

# %% [markdown]
# ## Análise 1 — Volume de pedidos por dia
# Aqui está o coração da Semana 1: quantos pedidos entram por dia?
# Essa é a variável que, mais pra frente, vamos tentar PREVER.

# %% Volume diário
# Agrupa por data (só o dia, sem a hora) e conta os pedidos.
pedidos_por_dia = (
    df.set_index("order_purchase_timestamp")
      .resample("D")          # "D" = agrupar por dia
      .size()                 # conta quantos pedidos em cada dia
)

print("Média de pedidos por dia:", round(pedidos_por_dia.mean(), 1))
print("Dia de maior volume:", pedidos_por_dia.idxmax().date(),
      "com", pedidos_por_dia.max(), "pedidos")

# Gráfico do volume diário
pedidos_por_dia.plot(title="Volume de pedidos por dia")
plt.ylabel("Nº de pedidos")
plt.xlabel("Data")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Análise 2 — Volume por mês
# O gráfico diário costuma ser "ruidoso". Agrupar por mês revela a TENDÊNCIA
# (crescimento, sazonalidade) com mais clareza.

# %% Volume mensal
pedidos_por_mes = (
    df.set_index("order_purchase_timestamp")
      .resample("ME")         # "ME" = fim de mês (month end)
      .size()
)

pedidos_por_mes.plot(kind="bar", title="Volume de pedidos por mês")
plt.ylabel("Nº de pedidos")
plt.xlabel("Mês")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Análise 3 — Padrão por dia da semana
# Operação de logística vive de padrão semanal. Será que segunda tem mais
# pedido que domingo? Isso importa MUITO pra previsão de demanda.

# %% Volume por dia da semana
# dt.dayofweek: 0 = segunda, 6 = domingo
dias_nomes = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
volume_dia_semana = (
    df["order_purchase_timestamp"].dt.dayofweek
      .value_counts()
      .sort_index()
)
volume_dia_semana.index = dias_nomes

volume_dia_semana.plot(kind="bar", title="Volume de pedidos por dia da semana")
plt.ylabel("Nº de pedidos")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Minhas 3 observações
#
# 1. (Tendência) O volume de pedidos cresceu de forma consistente ao longo
#    de 2017 e 2018, refletindo a expansão do e-commerce no período. Os meses
#    finais do dataset mostram queda brusca, mas isso é o corte dos dados
#    (a coleta termina em setembro de 2018), e não uma queda real de operação.
#
# 2. (Sazonalidade) O volume varia bastante ao longo da semana, com os dias
#    úteis concentrando mais pedidos que o fim de semana. Isso faz sentido
#    para a operação: pico de pedidos nos dias úteis significa maior demanda
#    de separação e expedição no começo da semana seguinte.
#
# 3. (Evento e qualidade) O maior pico do período foi em 24/11/2017, com 1176
#    pedidos contra uma média de 128 por dia — é a Black Friday, quase 10x o
#    volume normal. Também há valores nulos na coluna de data de entrega ao
#    cliente (~2965), que correspondem a pedidos ainda não entregues quando
#    os dados foram coletados.
 
# %% [markdown]
# ## Semana 2 — Entendendo overfitting
#
# Overfitting é quando um modelo decora os dados que já viu em vez de aprender
# a regra por trás deles. Ele fica muito bom nos casos conhecidos, mas se dá
# mal quando aparece uma situação diferente.
#
# Um exemplo da operação logística: um funcionário responsável por receber e
# atender caminhões. Se toda segunda-feira chegam 200 caminhões, ele decora
# esse número e organiza a equipe sempre do mesmo jeito. Mas se numa segunda
# chegarem 500, ele trava e precisa pedir ajuda, porque decorou "segunda = 200"
# em vez de entender de verdade como dimensionar a equipe conforme a demanda.
#
# Isso é ruim porque o objetivo de um modelo é acertar em situações novas. Um
# modelo que só vai bem no que já viu, e falha no que é diferente, não serve
# pra prever o futuro.
