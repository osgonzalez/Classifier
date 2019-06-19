# importamos librerías
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#Load DataSet
from sklearn.datasets import load_boston
boston_dataset = load_boston()

dataframe = pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
dataframe['target'] = pd.Series(boston_dataset.target)
print(dataframe.tail(10))

# normalizamos los datos
scaler = StandardScaler()
dataframe = dataframe.drop(['target'], axis=1)  # quito la variable dependiente "Y"
scaler.fit(dataframe)  # calculo la media para poder hacer la transformacion
X_scaled = scaler.transform(dataframe)  # Ahora si, escalo los datos y los normalizo

# Instanciamos objeto PCA y aplicamos
pca = PCA(
    n_components=9)  # Otra opción es instanciar pca sólo con dimensiones nuevas hasta obtener un mínimo "explicado" ej.: pca=PCA(.85)
pca.fit(X_scaled)  # obtener los componentes principales
X_pca = pca.transform(X_scaled)  # convertimos nuestros datos con las nuevas dimensiones de PCA

print("shape of X_pca", X_pca.shape)
expl = pca.explained_variance_ratio_
print(expl)
print('suma:', sum(expl[0:5]))
# Vemos que con 5 componentes tenemos algo mas del 85% de varianza explicada


