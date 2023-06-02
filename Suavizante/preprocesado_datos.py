import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv('./Dataset_suavizante.csv')
# print(df.shape)

# Varibles independientes
x = df.iloc[:, :-1].values
# print(x[0].shape)

y = df.iloc[:, 39].values
# print(y.shape)

# test_size toma el 20% de los datos para testing
# random_state es un número cualquiera para que siempre me genere el mismo resultado
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Escalo los datos entre 0 y 1
x_min_max_train = preprocessing.MinMaxScaler().fit_transform(x_train)
x_min_max_test = preprocessing.MinMaxScaler().fit_transform(x_test)
# print(x_min_max_train, x_min_max_test)

y_min_max_train = preprocessing.MinMaxScaler().fit_transform(y_train.reshape(-1, 1)) # Se le da la forma para que se pueda escalar
y_min_max_test = preprocessing.MinMaxScaler().fit_transform(y_test.reshape(-1, 1))
# print(y_min_max_train, y_min_max_test)

# Crear los dataset
x_train1 = pd.DataFrame(x_min_max_train)
x_test1 = pd.DataFrame(x_min_max_test)
x_train1.columns = x_train.columns
x_test1.columns = x_test.columns
print(x_train1)