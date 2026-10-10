# Previsão de Demanda Diária — E-commerce (Olist)

Projeto de Machine Learning para prever o volume diário de pedidos de um e-commerce brasileiro, usando dados reais da Olist.

**Autor:** Rafael Simões · Estudante de Análise e Desenvolvimento de Sistemas (FMU), com experiência em operação logística (warehouse/fulfillment).

---

## O problema

Em uma operação de logística, saber quantos pedidos vão entrar amanhã muda tudo: quantas pessoas escalar, quantas docas abrir, como organizar a separação e a expedição. Errar pra menos gera atraso; errar pra mais gera custo com equipe parada.

O objetivo deste projeto é prever o **volume diário de pedidos** a partir da data, e medir o quanto essa previsão erra em dias que o modelo nunca viu.

## Os dados

- **Fonte:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)
- **Arquivo usado:** `olist_orders_dataset.csv` (cerca de 99 mil pedidos, de 2016 a 2018)
- Os pedidos foram agrupados por dia, gerando 597 dias de operação.
- Os dias das bordas do período (início e fim da coleta, com quase nenhum pedido) foram removidos, porque não representam a operação real.

> O CSV não está neste repositório. Para rodar o projeto, baixe o dataset no Kaggle e coloque o `olist_orders_dataset.csv` na pasta do projeto.

## O que os dados mostram

- **Tendência:** o volume cresceu de forma consistente ao longo de 2017 e 2018.
- **Padrão semanal:** os dias úteis concentram mais pedidos que o fim de semana.
- **Evento:** o maior pico foi em 24/11/2017, com 1.176 pedidos contra uma média de 128 por dia. É a Black Friday, quase 10 vezes o volume normal.

## Como o projeto evoluiu

Cada etapa está em um arquivo separado, mostrando a evolução do raciocínio:

| Etapa | Arquivo | O que foi feito | Erro médio (MAE) |
|---|---|---|---|
| 1 | `semana1_eda_olist.py` | Exploração dos dados | — |
| 2 | — | Estudo dos conceitos (sem script) | — |
| 3 | `semana3_train_test.py` | Primeiro modelo (regressão linear), com treino/teste e baseline | 60,5 pedidos |
| 4 | `semana4_comparar_modelos.py` | Árvore e Random Forest. Ficaram piores que o baseline (overfitting e falta de pista de tendência) | 61,3 pedidos |
| 4 | `semana4b_modelos_melhorados.py` | Correções: pista de tendência e limite de profundidade das árvores | 19,7 pedidos |
| 5 | `semana5_gradient_boosting.py` | Gradient Boosting contra Random Forest | 17,8 pedidos |
| 6 | `semana6_pipeline_validacao.py` | Pipeline e validação cruzada em 5 rodadas | **21,7 pedidos** |

**Baseline** (prever sempre a média): 63,5 pedidos de erro.

## Resultado final

O modelo escolhido foi o **Gradient Boosting**. Na validação cruzada (5 rodadas), ele errou em média **21,7 pedidos por dia**, contra cerca de 23 do Random Forest e 63,5 do baseline. Ou seja, erra cerca de **3 vezes menos** que simplesmente usar a média.

O erro variou de 17,8 a 30,3 pedidos entre as rodadas. Isso mostra que testar com uma divisão só pode enganar, e que alguns dias, como a Black Friday, são bem mais difíceis de prever.

## O que aprendi

- **A pista certa vale mais que o modelo certo.** Só adicionar a pista de tendência (dias desde o início) fez a regressão linear cair de 61,3 para 29,3 pedidos de erro. Trocar de modelo trouxe ganhos bem menores.
- **Modelo potente sem controle decora.** A árvore de decisão sem limite de profundidade errou 96,3 pedidos, pior que o baseline, porque decorou os dias do treino (overfitting).
- **Sempre comparar com um baseline.** Sem ele, não dá pra saber se o modelo está ajudando de verdade.
- **Uma divisão de teste só não basta.** A validação cruzada mostrou um erro mais realista que o teste único.

## Limitações e próximos passos

- **Prever o futuro:** a validação usada embaralha os dias. Para medir a previsão de dias futuros de verdade, o próximo passo é uma validação por período (treinar no passado e testar no período seguinte).
- **Modelos de árvore não extrapolam:** eles não conseguem continuar a tendência depois do fim dos dados.
- **Datas promocionais:** adicionar uma pista de Black Friday e outras datas de pico deve reduzir os maiores erros.
- **Fim da coleta:** os últimos dias da base já mostram queda causada pelo encerramento da coleta, e não pela operação.

## Como rodar

1. Instale as bibliotecas:
   ```
   pip install -r requirements.txt
   ```
2. Baixe o `olist_orders_dataset.csv` no Kaggle e coloque na pasta do projeto.
3. Rode qualquer arquivo, por exemplo:
   ```
   python semana6_pipeline_validacao.py
   ```

## Tecnologias

Python · pandas · NumPy · Matplotlib · scikit-learn
