---
layout: post
title: Detecção de Transações Fraudulentas no Cartão de Crédito com PCA
excerpt: "Detecção de anomalias nos dados usando PCA"
lang: pt
modified: 11/01/2020, 9:00:24
tags: [python, machine learn, pca]
comments: true
category: blog
---

### Indice

1. Introdução
2. Análise Exploratória
3. Preparação e Modelagem dos Dados
4. Resultados e Conclusões
5. Referências

***

## Introdução

Nos dias atuais em meio à evolução dos meios digitais o número de transações feitas por meio destes canais tem aumentado ano após anos. Com esse aumento temos que elevar técnicas de segurança, a medida que fraudes são cada vez mais recorrentes nesses canais.
O objetivo deste trabalho é explorar alguns métodos de detecção de fraude com o intúito de verificar a prova de conceito dos métodos e suas características na aplicação de soluções junto ao negócio.

***
## Análise Exploratória

{% highlight python %}
# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Configurações necessárias
pd.set_option('display.max_columns', None)
{% endhighlight %}

{% highlight python %}
# importando os dados necessários para um pandas dataframe
# A base está no item 6 da referência
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

# Tipos das variáveis e se há nulos
display(df.info())

# Estatística básica das variáveis
df.describe()
{% endhighlight %}


### Conclusões Parciais

A pricípio, um ponto importante a ser considerado no conjunto de dados é que para proteção dos dados dos cliente uma primeira transformação nos dados foi feita e o que temos são os 28 componentes principais desta transfomação (com adição da coluna 'Amount', que representa o valor da transação e a coluna 'Time', que representa o delta T daquela linha para a primeira transação no conjunto de dados).

Uma rápida análise nas distribuições dos dados podemos notar que a coluna que traz o valor da transação aparece zerada em algumas linhas e isso terá de ser tratado, uma vez que transações zeradas podem representar falhas na leitura da transação e mesmo podendo inferir este valor (substituição pela média, entre outras técnicas) vamos tentar não carregar um erro, a princípio, para as próximas etapas.

Por fim podemos notar que não só pela origem do problema, mas também pela própria coluna 'Class', os dados estão muito desbalanceados, a medida que o evento fraude é extremamente raro em meio a diversas transações, o que pode causar problemas em técnicas de classificação.

***

## Preparação e Modelagem dos Dados

{% highlight python %}
# Quebrando o dataframe em treino e teste (70% treino e 30% teste).

X = df.drop(['Time', 'Class'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
{% endhighlight %}

* Baseline model

Aqui precisamos de um modelo básico para comparação com outras técnicas. O modelo escolhido foi a regressão logística.

{% highlight python %}
# Criando o modelo e treinando com os dados de treino
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
{% endhighlight %}

* Reconstrução do Erro com PCA

Aqui a idéia é transformarmos mais um vez o consjunto de dados visando uma variancia explicada de 95% (deixando o algoritmo dizer o melhor número de componentes para representar os dados), assim aplicamos a transformação inversa para reconstruir os dados e por fim calculamos o MSE (Mean Square Error) de cada ponto para detecção das anomalias. Aqui o threshold adotado é o intervalo de desvios padrões da média que minimiza a quantidade de FPs nas predições.

{% highlight python %}
#Instanciando o PCA do sklearn
pca = PCA(n_components=0.95)

# Aplicando a transformação do PCA nos dados
df_reduced = pca.fit_transform(X_train)
# Aplicando a reconstrução das componentes principais
df_inv = pd.DataFrame(pca.inverse_transform(df_reduced), columns=X_train.keys())
# Calculando o MSE de cada ponto da base de treino
result = pd.DataFrame(np.sqrt(np.sum(np.asarray(X_train.values - df_inv.values)**2, axis=1)))
result['Class'] = y_train.values.tolist()
# Calculando a média e o Desvio padrão da classe majoritária
mean_zero = result[result.Class == 0][0].mean()
std_zero = result[result.Class == 0][0].std()
# Aplicando a predição de fraud para dados maiores que a média mais um intervalo de desvio padrão
result['predicted'] = result[0].apply(lambda x: 1 if (x > mean_zero + std_zero) else 0)
{% endhighlight %}

***

# Resultados e Conclusões

Para a avaliação dos dados foi escolhida a métrica ROC AUC (Area Under the Receiver Operating Characteristic) e os dados obtidos de cada medição, no treino e no teste pode ser visto na tabela a seguir:

|Classificador|Score no Treino|Socore no Teste|
|:-:|:-:|:-:|
|Regressão Logística|0.807|0.809|
|Reconstrução do Erro com PCA|0.906|0.920|


***

# Referências

1. [PCA-Based Outlier Detection](https://ieeexplore.ieee.org/abstract/document/4907305?casa_token=w9MUoiiYek0AAAAA:fesefux_fHbeYovRlIdo7iGaM7sZ4yNOXVv4VCdtKCc_8WoaE6cDzu4pUQq3OGXG889Ot57Pf4OcFQ)
2. [Question in stackexchange: Anomaly detection using PCA reconstruction error](https://stats.stackexchange.com/questions/259806/anomaly-detection-using-pca-reconstruction-error)
3. [A Survey on Outlier Detection Techniques for Credit Card Fraud Detection](https://www.semanticscholar.org/paper/A-Survey-on-Outlier-Detection-Techniques-for-Credit-Pawar-Kalavadekar/863e77593b9c3abed4d83348e2dc898a0bd9e850?p2df)
4. [sklearn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
5. [PCA — how to choose the number of components?](https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/)
6. [Kaggle Credit Cart Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
