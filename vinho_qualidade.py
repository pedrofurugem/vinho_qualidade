import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

#Construa uma árvore de decisão para regressão. Considere que quality é avariável dependente.
# Exiba um gráfico de duas dimensões de "alcohol" versus "quality".
# Exiba um gráfico de três dimensões de "alcohol" versus "densit" versus "quality".

#importando dados
dataset = pd.read_csv('winequality-red.csv')#extensão csv


#seperando as colunas alcohol, densit e quality
##print(registers.shape)

#ordenando as variáveis 
X = dataset.iloc[:,[7, 10]]
Y = dataset.iloc[:, 11] 

x = X.values.reshape(-1, 2)
y = Y.values.reshape(-1, 1)

#Realizando o treinamento utilizando aclasse DecisionTreeRegressor
decisionTreeRegressor = DecisionTreeRegressor(random_state = 0) 
decisionTreeRegressor.fit(x, y)

# visualizando os dados graficamente
#densit vs quality
plt.scatter(x[:, 0], decisionTreeRegressor.predict(x),color="red")
plt.xlabel('densit')
plt.ylabel('quality')
plt.title('densit vs quality')
plt.show()

#alcohol vs quality
plt.scatter(x[:, 1], decisionTreeRegressor.predict(x),color="blue")
plt.xlabel('alcohol')
plt.ylabel('quality')
plt.title('alcohol vs quality')
plt.show()

#densit vs alcohol vs quality
fig = plt.figure()
subplot = fig.add_subplot(111, projection='3d')
subplot.scatter(x[:, 0],  x[:, 1],decisionTreeRegressor.predict(x),color="green")
plt.show()

