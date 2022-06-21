#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# ## <font color='blue'>Estudo de Caso 4</font>
# 
# ## <font color='blue'>Engenharia de Atributos Para E-Commerce Analytics</font>

# Obs: Estaremos trabalhando em um grande projeto de Ciência de Dados distribuído em 3 capítulos:
# 
# - Análise Exploratória de Dados
#     - EDA Parte 1
#     - EDA Parte 2
# - **Engenharia de Atributos**
# - Pré-Processamento de Dados

# A Engenharia de Atributos refere-se ao processo de usar o conhecimento do domínio (área de negócio) para remover, selecionar e transformar os dados, mantendo somente os atributos mais relevantes.

# ![title](imagens/EstudoCaso4.png)

# In[1]:


# Versão da Linguagem Python
from platform import python_version
print('Versão da Linguagem Python Usada Neste Jupyter Notebook:', python_version())


# In[2]:


# Para atualizar um pacote, execute o comando abaixo no terminal ou prompt de comando:
# pip install -U nome_pacote

# Para instalar a versão exata de um pacote, execute o comando abaixo no terminal ou prompt de comando:
#!pip install nome_pacote==versão_desejada

# Depois de instalar ou atualizar o pacote, reinicie o jupyter notebook.

# Instala o pacote watermark. 
# Esse pacote é usado para gravar as versões de outros pacotes usados neste jupyter notebook.
#!pip install -q -U watermark


# In[1]:


# Imports
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats#PACOTE PARA FUNÇÃO MATEMATICA


# In[4]:


# Versões dos pacotes usados neste jupyter notebook
get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Data Science Academy" --iversions')


# ## Carregando o Dataset

# In[2]:


# Carrega o dataset
df = pd.read_csv('dados/dataset.csv')


# In[ ]:





# ## Analise Exploratoria dos Dados 

# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


# Variáveis numéricas
nums = ['numero_chamadas_cliente', 
        'avaliacao_cliente', 
        'compras_anteriores', 
        'desconto', 
        'custo_produto', 
        'peso_gramas']


# In[8]:


# Variáveis categóricas
cats = ['modo_envio', 
        'prioridade_produto', 
        'genero',
        'corredor_armazem']


# In[9]:


# Variável target
target = ['entregue_no_prazo'] 


# In[10]:


df[nums].describe()


# In[11]:


df[cats].describe()


# In[12]:


df[target].value_counts()


# ## Limpeza de Dados

# ### Tratamento de Valores Ausentes
# 
# **ATENÇÃO**: Valor ausente significa ausência de informação e não ausência de dado!
# 
# O tratamento pode ser feito antes, durante ou depois da Análise Exploratória, mas idealmente deve ser feito antes da Engenharia de Atributos. Mas fique atento: a Engenharia de Atributos e o Pré-Processamento podem gerar valores ausentes, o que precisa ser tratado.

# In[13]:


df.head(3)


# In[14]:


# Verifica se há valores ausentes
df.isna().sum()


# ### Tratamento de Valores Duplicados
# 
# Valores duplicados significam duplicidade dos dados em toda a linha (todo o registro).
# 
# O tratamento pode ser feito antes, durante ou depois da Análise Exploratória, mas idealmente deve ser feito antes da Engenharia de Atributos.

# In[15]:


df.head(3)


# In[16]:


# Verifica se há valores duplicados
df.duplicated().sum()


# ### Tratamento de Valores Outliers
# 
# Leia o manual em pdf com a definição do que é o z-score e a definição de valor outlier.
# 
# O tratamento pode ser feito antes, durante ou depois da Análise Exploratória, mas idealmente deve ser feito antes da Engenharia de Atributos. 
# 
# 
# Antes vamos saber o comprimento dos dados com o LEN 

# In[17]:


print(f'Número de linhas antes de filtrar valores extremos (outliers): {len(df)}')


# In[18]:


df[nums].head()


# In[ ]:





# Vamos calcular a média e o desvio padrão. Vamos escolher a coluna desconto. 
# 

# In[19]:


df.desconto.mean()


# In[20]:


df.desconto.std()


# In[21]:


df.desconto.hist();


# Temos Outliers em nossa base de dados. Analisando o grafico tivemos um desconto abaixo da média. 

# In[ ]:





# In[22]:


# Calcula os limites superior e inferior
# Um valor outlier é aquele que está abaixo do limite inferior ou acima do limite superior
limite_superior = df.desconto.mean() + 3 * df.desconto.std()
print("Valor superior:", limite_superior)
limite_inferior = df.desconto.mean() - 3 * df.desconto.std()
print("Valor inferior:", limite_inferior)


# In[ ]:





# In[ ]:





# In[23]:


# Extra os registros com outliers na coluna desconto
# Fatiar utilizando o PANDAS


df_outliers_desconto = df[(df.desconto <= limite_inferior) | (df.desconto >= limite_superior)]
df_outliers_desconto.head()


# In[24]:


Teve um desconto muito acima da média 


# In[25]:


#Remover todos os OUTLIERS da coluna Desconto.


# Filtra o dataframe removendo os registros com outliers na coluna desconto
df = df[(df.desconto > limite_inferior) & (df.desconto < limite_superior)]


# In[26]:


#Mostar a quantidade de linhas sem os Outliers.

print(f'Número de linhas antes de filtrar valores extremos (outliers): {len(df)}')


# In[ ]:





# In[ ]:





# CRIAR  UMA FUNÇÃO LOOP PARA TRATAR TODOS OS Outliers.
# 
# 

# In[27]:




registros = np.array([True] * len(df))


# In[28]:


type(registros)


# In[29]:


np.count_nonzero(registros == True)


# In[30]:


np.count_nonzero(registros == False)


# In[31]:





# Variáveis numéricas (sem a variável desconto)
nums2 = ['numero_chamadas_cliente', 
         'avaliacao_cliente', 
         'compras_anteriores', 
         'custo_produto', 
         'peso_gramas']


# In[32]:


# Loop por cada variável numérica
for col in nums2:
    
    # Calcula o z-score absoluto
    zscore = abs(stats.zscore(df[col])) 
    
    # Mantém valores com menos de 3 z-score absoluto
    registros = (zscore < 3) & registros


# In[33]:


np.count_nonzero(registros == True)


#Sobrou 10643 linhas, tiramos os outliers


# In[34]:


np.count_nonzero(registros == False)


# In[35]:


# Removemos registros com o z-score abaixo de 3 nas colunas numéricas
#Filtrando o registro

df = df[registros] 


# In[36]:


print(f'Número de linhas após filtrar valores extremos (outliers): {len(df)}')


# ### Tratamento de Desbalanceamento de Classe
# 
# Deve ser feito nos dados de treino, após o pré-processamento dos dados.

# In[37]:


df.columns


# In[38]:


df['entregue_no_prazo'].value_counts()


# ------------------------------------------------------------------------------

# ## Engenharia de Atributos

# ## Feature Selection 
# 
# Aqui tomamos as decisões sobre quais variáveis serão usadas na Engenharia de Atributos.
# 
# 
# 
# A  seleção  de  recursos(feature  selection)é  o  processo  de  isolar  os  recursos  mais consistentes,  não  redundantes  e  relevantes  a  serem  usados na  construção  de  ummodelo. Reduzir metodicamente o tamanho dos conjuntos de dados é importante, pois o tamanho e a variedade  dos  conjuntos  de  dados  continuam  a  crescer.  
# 
# O  principal  objetivo  da  seleção  de recursos é melhorar o desempenho de um modelo preditivo e reduzir o custo computacional da modelagem.A seleção de recursos, um dos principais componentes da engenharia de recursos, é o processo  de  seleção  dos  recursos  mais  importantes  a  serem  inseridos  em  algoritmos  de aprendizado de máquina. Técnicas de seleção de recursos são empregadas para reduzir o número de  variáveis de  entrada,  eliminando  recursos  redundantes  ou  irrelevantes  e  estreitando  o conjunto de recursos para aqueles mais relevantes para o modelo de aprendizado de máquina.
# 
# Os principais benefícios de realizar a seleção de recursos com antecedência, em vez de deixar o modelo de aprendizado de máquina descobrir quais recursos são mais importantes, incluem:
# 
# •Modelos mais simples: modelos simples são fáceis de explicar -um modelo muito complexo e inexplicável não é valioso.
# 
# •Tempos  de  treinamento  mais  curtos:  um  subconjunto  mais  preciso  de  recursos diminui a quantidade de tempo necessária para treinar um modelo.
# 
# •Redução de variância: aumentaa precisão das estimativas que podem ser obtidas para uma determinada simulação
# 
# •Evitar a “maldição”da alta dimensionalidade:à medida que a dimensionalidade e o número de recursos aumentam, o volume de espaço aumenta tão rapidamente que os dados disponíveis se tornam limitados. Selecionar e reduzir o número de recursos evita que não tenhamos dados suficientes para o treinamento o modelo.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[39]:


df.columns


# In[40]:


df.sample(5)


# In[41]:


# Correlação
df.corr()


# In[42]:


# Mapa de correlação 
plt.figure(figsize = (10, 8))
sns.heatmap(df.corr(), cmap = 'Purples', annot = True, fmt = '.2f');


# **Decisão**: Na figura acima, os recursos que ultrapassaram o limite (+/- 0,05) na correlação com o recurso de destino (entregue_no_prazo) serão escolhidos para serem processados na Engenharia de Atributos, nesse caso: numero_chamadas_cliente, custo_produto, compras_anteriores, desconto e peso_gramas.
# 
# A variável avaliacao_cliente não parece relevante e não será processada na Engenharia de Atributos. A variável ID será descartada, pois não representa informação útil.

# In[43]:


# Gráfico de barras
df_sample = df.sample(1000, random_state = 42)
plt.figure(figsize = (20,8))
for i in range(0, len(cats)):
    plt.subplot(2, 3, i+1)
    sns.countplot(x = df_sample[cats[i]], color = 'blue', orient = 'v')
    plt.tight_layout()


# **Decisão**: Na figura acima pode-se ver que todas as variáveis categóricas têm uma diferença bastante grande em termos de contagens das classes, exceto a variável de Gênero. Sendo assim, não vamos considerar a variável genero na Engenharia de Atributos.

# **Obs: Como as tarefas a seguir são complexas, demonstraremos apenas como algumas colunas. Fique à vontade para refazer a Engenharia de Atributos usando todas as colunas selecionadas conforme regras definidas acima.**

# -------------------------------------------------------------------------------

# ## Feature Extraction 
# 
# Aqui fazemos a extração de novas variáveis a partir da informação contida em outras variáveis.
# 
# 
# A  extração  de  recursos(feature  extraction)é  um processo  de  redução  de dimensionalidaderealizado dentro da Engenharia de Atributos, no qual um conjunto inicial de dados  brutos  é  dividido  e  reduzido  a  grupos  mais  gerenciáveis.  
# 
# Então,  quando  você  quiser processar, será mais fácil. A característica mais importante degrandes conjuntos de dados é que eles possuem um grande número de variáveis. 
# 
# Essas variáveis exigem muitos recursos de computação para serem processadas. Portanto, a extração de recursos ajuda a obter o melhor recurso desses conjuntos selecionando e combinando variáveis em recursos, reduzindo efetivamente a quantidade de dadose permitindocompreender os dados por diferentes perspectivas. 
# 
# Esses recursos são fáceis de  processar, eainda  são  capazes  de  descrever  o  conjunto  de  dados  real  com  precisão  e originalidade.
# 
# Existem diversas maneiras de realizar feature extractione no Estudo de Caso usaremos uma abordagem baseada na compreensão do problema de negócio.

# In[44]:


df.columns


# In[45]:


df.head()


# In[46]:


# Cria uma cópia do dataframe
df_eng = df.copy()


#é bom criar caso surge algum erro, recomendavel.


# In[47]:


df_eng.shape


# In[70]:


df_eng.dtypes


# ### 1- Performance de Envio do Produto Por Prioridade do Produto
# 
# Todo atraso no envio dos produtos é igual, ou seja, tem a mesma proporção? A prioridade de envio dos produtos gera mais ou menos atrasos?
# 
# Criaremos uma nova variável que representa a performance do envio do produto com base na seguinte regra de negócio:
# 
# - Se a prioridade do produto era alta e houve atraso no envio, o atraso é crítico.
# - Se a prioridade do produto era média e houve atraso no envio, o atraso é problemático.
# - Se a prioridade do produto era baixa e houve atraso no envio, o atraso é tolerável.
# - Outra opção significa que o envio foi feito no prazo e não apresenta problema.

# In[71]:


df_eng.prioridade_produto.value_counts()


# In[72]:


# Criamos a variável e preenchemos com nan
df_eng['performance_prioridade_envio'] = np.nan


# In[73]:


# ATENÇÃO!!!! Apenas para compreender o conceito!!!
df_eng['performance_prioridade_envio'] = np.where(
    (df_eng['prioridade_produto'] == 'alta') & (df_eng['entregue_no_prazo'] == 0), "Ruim", "Excelente")


# In[74]:


df_eng['performance_prioridade_envio'].value_counts()


# In[75]:


# Limpamos todos os valores da variável
df_eng['performance_prioridade_envio'] = np.nan


# In[76]:


# Agora sim a solução!!!
df_eng['performance_prioridade_envio'] = np.where(
        (df_eng['prioridade_produto'] == 'alta') & (df_eng['entregue_no_prazo'] == 0), "Atraso Crítico", 
    np.where(
        (df_eng['prioridade_produto'] == 'media') & (df_eng['entregue_no_prazo'] == 0), "Atraso Problemático",
    np.where(
        (df_eng['prioridade_produto'] == 'baixa') & (df_eng['entregue_no_prazo'] == 0), "Atraso Tolerável", 
    "Não Houve Atraso")))


# In[77]:


df_eng['performance_prioridade_envio'].value_counts()


# In[78]:


#Foi crianda uma nova variavel.

df_eng.sample(5)


# In[79]:


# Agrupamos os dados para análise
df_report1 = df_eng.groupby(['performance_prioridade_envio', 
                             'entregue_no_prazo']).agg({'prioridade_produto': ['count']}).reset_index()
df_report1.head()


# In[80]:


# Ajustamos os nomes das colunas
df_report1.columns = ['performance_prioridade_envio', 'entregue_no_prazo', 'contagem']
df_report1.head()


# In[81]:


# Pivot(transformar linhas em colunas ou colunas em linhas)
df_report1 = pd.pivot_table(df_report1,
                             index = 'performance_prioridade_envio',
                             columns = 'entregue_no_prazo',
                             values = 'contagem').reset_index()
df_report1.head()


# In[82]:


# Novo ajuste nos nomes das colunas
df_report1.columns = ['Status do Envio', 'Total Atraso', 'Total no Prazo']
df_report1.head()


# In[83]:


# Replace do valor nan por zero(colocar o número 0 nos NANs)
df_report1['Total Atraso'] = df_report1['Total Atraso'].replace(np.nan, 0)
df_report1['Total no Prazo'] = df_report1['Total no Prazo'].replace(np.nan, 0)
df_report1.head()


# In[84]:


# Concatena colunas criando uma terceira variável
df_report1["Total"] = df_report1["Total Atraso"] + df_report1["Total no Prazo"]
df_report1.head()


# In[85]:


# Não precisamos mais das colunas usadas na concatenação. Vamos removê-las.
df_report1.drop(df_report1.columns[[1, 2]], axis = 1, inplace = True)
df_report1.head()


# In[86]:


# Plot
df_report1.set_index("Status do Envio").plot(kind = 'bar', 
                title = 'Total de Envios dos Produtos Por Status',
                ylabel = 'Total', 
                xlabel = '\nStatus do Envio',
                colormap = 'jet',
                figsize=(12,8))
plt.xticks(rotation = 0)
plt.show()


# ### 2- Performance de Envio do Produto Por Prioridade de Envio e Modo de Envio
# 
# O modo de envio dos produtos associado à proridade de envio dos produtos, tem impacto no atraso dos produtos?
# 
# Criaremos uma nova variável que representa a performance do envio do produto com base na seguinte regra de negócio:
# 
# - Se a prioridade do produto era alta, o modo de envio era Navio e houve atraso no envio, o atraso é crítico por Navio.
# - Se a prioridade do produto era média, o modo de envio era Navio e houve atraso no envio, o atraso é problemático por Navio.
# - Se a prioridade do produto era baixa, o modo de envio era Navio e houve atraso no envio, o atraso é tolerável por Navio.
# - Se a prioridade do produto era alta, o modo de envio era Aviao e houve atraso no envio, o atraso é crítico por Aviao.
# - Se a prioridade do produto era média, o modo de envio era Aviao e houve atraso no envio, o atraso é problemático por Aviao.
# - Se a prioridade do produto era baixa, o modo de envio era Aviao e houve atraso no envio, o atraso é tolerável por Aviao.
# - Se a prioridade do produto era alta, o modo de envio era Caminhao e houve atraso no envio, o atraso é crítico por Caminhao.
# - Se a prioridade do produto era média, o modo de envio era Caminhao e houve atraso no envio, o atraso é problemático por Caminhao.
# - Se a prioridade do produto era baixa, o modo de envio era Caminhao e houve atraso no envio, o atraso é tolerável por Caminhao.
# - Outra opção significa que o envio foi feito no prazo e não apresenta problema.

# In[87]:


df_eng.columns


# In[88]:


df_eng.modo_envio.value_counts()


# In[90]:


# Solução
df_eng['performance_modo_envio'] = np.where(
        (df_eng['prioridade_produto'] == 'alta') & (df_eng['modo_envio'] == 'Navio') & (df_eng['entregue_no_prazo'] == 0), "Atraso Crítico na Entrega Por Navio", 
    np.where(
        (df_eng['prioridade_produto'] == 'media') & (df_eng['modo_envio'] == 'Navio') & (df_eng['entregue_no_prazo'] == 0), "Atraso Problemático na Entrega Por Navio",
    np.where(
        (df_eng['prioridade_produto'] == 'baixa') & (df_eng['modo_envio'] == 'Navio') & (df_eng['entregue_no_prazo'] == 0), "Atraso Tolerável na Entrega Por Navio", 
    np.where(
        (df_eng['prioridade_produto'] == 'alta') & (df_eng['modo_envio'] == 'Aviao') & (df_eng['entregue_no_prazo'] == 0), "Atraso Crítico na Entrega Por Aviao", 
    np.where(
        (df_eng['prioridade_produto'] == 'media') & (df_eng['modo_envio'] == 'Aviao') & (df_eng['entregue_no_prazo'] == 0), "Atraso Problemático na Entrega Por Aviao",
    np.where(
        (df_eng['prioridade_produto'] == 'baixa') & (df_eng['modo_envio'] == 'Aviao') & (df_eng['entregue_no_prazo'] == 0), "Atraso Tolerável na Entrega Por Aviao", 
    np.where(
        (df_eng['prioridade_produto'] == 'alta') & (df_eng['modo_envio'] == 'Caminhao') & (df_eng['entregue_no_prazo'] == 0), "Atraso Crítico na Entrega Por Caminhao", 
    np.where(
        (df_eng['prioridade_produto'] == 'media') & (df_eng['modo_envio'] == 'Caminhao') & (df_eng['entregue_no_prazo'] == 0), "Atraso Problemático na Entrega Por Caminhao",
    np.where(
        (df_eng['prioridade_produto'] == 'baixa') & (df_eng['modo_envio'] == 'Caminhao') & (df_eng['entregue_no_prazo'] == 0), "Atraso Tolerável na Entrega Por Caminhao", 
    "Não Houve Atraso")))))))))


# In[70]:


df_eng.sample(5)


# In[92]:


df_eng.performance_modo_envio.value_counts()


# In[93]:


# Agrupamos os dados para análise
df_report2 = df_eng.groupby(['performance_modo_envio', 
                             'entregue_no_prazo']).agg({'prioridade_produto': ['count']}).reset_index()
df_report2.head(10)


# In[94]:


df_report2.columns = ['performance_modo_envio', 'entregue_no_prazo', 'contagem']
df_report2.head(10)


# In[95]:


# Pivot
df_report2 = pd.pivot_table(df_report2,
                            index = 'performance_modo_envio',
                            columns = 'entregue_no_prazo',
                            values = 'contagem').reset_index()
df_report2.head(10)


# In[96]:


df_report2.columns = ['Status do Envio', 'Total Atraso', 'Total no Prazo']
df_report2.head(10)


# In[97]:


# Replace do valor nan por zero
df_report2['Total Atraso'] = df_report2['Total Atraso'].replace(np.nan, 0)
df_report2['Total no Prazo'] = df_report2['Total no Prazo'].replace(np.nan, 0)
df_report2.head(10)


# In[98]:


# Concatena colunas criando uma terceira variável
df_report2["Total"] = df_report2["Total Atraso"] + df_report2["Total no Prazo"]
df_report2.head(10)


# In[99]:


# Não precisamos mais dessas colunas. Vamos removê-las.
df_report2.drop(df_report2.columns[[1, 2]], axis = 1, inplace = True)
df_report2.head(10)


# In[100]:


# Plot
df_report2.set_index("Status do Envio").plot(kind = 'bar', 
                title = 'Total de Envios dos Produtos Por Status',
                ylabel = 'Total', 
                xlabel = '\nStatus do Envio',
                colormap = 'viridis',
                figsize = (20,8))
plt.xticks(rotation = 80)
plt.show()


# ### 3- Performance de Envio dos Produtos Considerando os Descontos
# 
# Há diferença na performance de envio dos produtos quando o produto recebe algum tipo de desconto?
# 
# Criaremos duas novas variáveis com base na seguinte regra de negócio:
# 
# **Variável 1 - faixa_desconto**
# 
# - Desconto acima ou igual à média
# - Desconto abaixo da média
# 
# **Variável 2 - performance_faixa_desconto**
# 
# - Se a faixa de desconto foi acima ou igual à média e houve atraso na entrega = "Atraso na Entrega com Desconto Acima da Média"
# 
# - Se a faixa de desconto foi acima ou igual à e não houve atraso na entrega = "Entrega no Prazo com Desconto Acima da Média"
# 
# - Se a faixa de desconto foi abaixo da média e houve atraso na entrega = "Atraso na Entrega com Desconto Abaixo da Média"
# 
# - Se a faixa de desconto foi abaixo da média e não houve atraso na entrega = "Entrega no Prazo com Desconto Abaixo da Média"

# In[101]:


df_eng.sample(5)


# In[102]:


df_eng.columns


# In[103]:


df_eng.desconto.describe()


# In[107]:


# Variável 1
df_eng['faixa_desconto'] = np.where(df_eng.desconto >= 12, "Desconto Acima da Media", "Desconto Abaixo da Media") 


# In[108]:


df_eng['faixa_desconto'].value_counts()


# In[109]:


df_eng.sample(5)


# In[110]:


# Variável 2
df_eng['performance_faixa_desconto'] = np.where(
        (df_eng['faixa_desconto'] == 'Desconto Acima da Media') & (df_eng['entregue_no_prazo'] == 0), "Atraso na Entrega com Desconto Acima da Media", 
    np.where(
        (df_eng['faixa_desconto'] == 'Desconto Abaixo da Media') & (df_eng['entregue_no_prazo'] == 0), "Atraso na Entrega com Desconto Abaixo da Media",
    np.where(
        (df_eng['faixa_desconto'] == 'Desconto Acima da Media') & (df_eng['entregue_no_prazo'] == 1), "Entrega no Prazo com Desconto Acima da Media",
    np.where(
        (df_eng['faixa_desconto'] == 'Desconto Abaixo da Media') & (df_eng['entregue_no_prazo'] == 1), "Entrega no Prazo com Desconto Abaixo da Media",
   "NA"))))


# In[111]:


df_eng.sample(5)


# In[112]:


df_eng['performance_faixa_desconto'].value_counts()


# In[113]:


# Agrupamos os dados para análise
df_report3 = df_eng.groupby(['performance_faixa_desconto', 
                             'entregue_no_prazo']).agg({'ID': ['count']}).reset_index()
df_report3.head()


# In[114]:


df_report3.columns = ['performance_faixa_desconto', 'entregue_no_prazo', 'contagem']
df_report3.head()


# In[115]:


# Pivot
df_report3 = pd.pivot_table(df_report3,
                             index = 'performance_faixa_desconto',
                             columns = 'entregue_no_prazo',
                             values = 'contagem').reset_index()
df_report3.head()


# In[116]:


df_report3.columns = ['Status do Envio', 'Total Atraso', 'Total no Prazo']
df_report3.head()


# In[117]:


# Replace do valor nan por zero
df_report3['Total Atraso'] = df_report3['Total Atraso'].replace(np.nan, 0)
df_report3['Total no Prazo'] = df_report3['Total no Prazo'].replace(np.nan, 0)
df_report3.head()


# In[118]:


# Concatena colunas criando uma terceira variável
df_report3["Total"] = df_report3["Total Atraso"] + df_report3["Total no Prazo"]
df_report3.head()


# In[119]:


# Não precisamos mais dessas colunas. Vamos removê-las.
df_report3.drop(df_report3.columns[[1, 2]], axis = 1, inplace = True)
df_report3.head()


# In[120]:


# Plot
df_report3.set_index("Status do Envio").plot(kind = 'bar', 
                title = 'Total de Envios dos Produtos Por Status',
                ylabel = 'Total', 
                xlabel = '\nStatus do Envio',
                colormap = 'plasma',
                figsize = (20,8))
plt.xticks(rotation = 0)
plt.show()


# In[121]:


df_eng.sample(10)


# In[123]:


# Salva o dataframe
df_eng.to_csv('dados/df_eng.csv', sep = ',', encoding = 'utf-8')

#Salvar o Dataset limpo.


# # Fim

# In[ ]:




