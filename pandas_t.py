import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# carichiamo il dataset

iris = pd.read_csv("/Users/tanialosole/Desktop/Repo_ML_Deeplearning/2 - Datasets e data preprocessing/data/iris.csv")

# la dicitura .head(5) ci mostra le prime 5 entry del dataset
print(iris.head(5))

# la dicitura .tail() ci mostra le ultime entry
print(iris.tail(10))

#se non avessimo i nomi delle tabelle è possibile inserire i nomi manualmente
iris_noheader = pd.read_csv \
    ("/Users/tanialosole/Desktop/Repo_ML_Deeplearning/2 - Datasets e data preprocessing/data/iris_noheader.csv", header= None,
     names=["sepal_length",  "sepal_width",  "petal_length",  "petal_width", "species"])
print(iris_noheader.tail(10))

print(iris.columns)
print(iris.info())

Y = iris["species"]
print(Y.head())

X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
print(X.head())

x = iris.drop("species", axis=1)
print(x.head())

iris_sampled = iris.copy()

iris_sampled = iris.sample(frac=1)

print(iris_sampled.head())

print(iris_sampled.iloc[3])
print(iris_sampled.loc[29])
print(iris_sampled.loc[29, "species"])
print(iris_sampled.iloc[:10, 1])
print(iris.shape)
print(iris.describe())
print(iris.max())
print(iris["sepal_length"].max())
print(iris["species"].unique())

#creare una mask, maschera in grado di selezionare solo le osservazioni che hanno
# la lunghezza del petalo maggiore della lunghezza media

long_petal_mask = iris["petal_length"] > iris["petal_length"].mean()
print(long_petal_mask)

#applicare una maschera
iris_long_petals = iris[long_petal_mask]
print(iris_long_petals.head())

# si crei una maschera che ci permetta di cambiare il valore della specie da “setosa” a “undefined”
iris_copy = iris.copy()
iris_copy[iris_copy["species"] == "setosa"] = "undefined"
print(iris_copy["species"].unique())

#nomalizzazione delle features
z = iris.drop("species", axis=1)
z_norm = (z - z.min())/(z.max()-z.min())
print(z_norm.head())

print(iris.sort_values("petal_length").head())

grouped_iris = iris.groupby("species")
print(grouped_iris.mean())

print(iris.apply(np.count_nonzero, axis=0).head())

a = iris.drop("species", axis=1)
a = a.applymap(lambda val: int(round(val, 0)))
print(a.head())

iris_nan = iris.copy()
samples = np.random.randint(iris.shape[0], size =(10))
iris_nan.loc[samples, "petal_length"] = None
print("******")
print(iris_nan["petal_length"].isnull().sum())

#calcoliamo il valore medio
mean_petal_length = iris_nan["petal_length"].mean()
iris_nan["petal_length"].fillna(mean_petal_length, inplace= True)
print(iris_nan["petal_length"].isnull().sum())

iris.plot(x="sepal_length", y="sepal_width", kind="hexbin")
#plt.show()


#ESERCITAZIONE MAGLIETTE

shirts = pd.read_csv("/Users/tanialosole/Desktop/Repo_ML_Deeplearning/2 - Datasets e data preprocessing/data/shirts.csv"
                     , index_col=0)
print(shirts.head())

#si va a creare un array dei valori all'inderno del dataset
X = shirts.values
print(X[:10])

#creiamo un dizionario per ordinare i label delle taglie

size_mapping = {"S": 0, "M": 1, "L": 2, "XL": 3}
shirts["taglia"] = shirts["taglia"].map(size_mapping)
print(shirts.head())

#quando hai un array np


fmap = np.vectorize(lambda t: size_mapping[t])
X[:, 0] = fmap(X[:, 0])
print(X[:5])

#PER LA COLONNA COLORE che è nominale
shirts_dummies = shirts.copy()
shirts_dummies = pd.get_dummies(shirts, columns=["colore"])
print(shirts.head())

#OneHotEncoder per creare le variabili di comodo
X = shirts.values
ct = ColumnTransformer([("colore", OneHotEncoder(), [1])], remainder="passthrough")
X = ct.fit_transform(X)
print(X[:5])

#QUANDO IN UN DATASET MANCANO DATI

iris_nan = pd.read_csv("/Users/tanialosole/Desktop/Repo_ML_Deeplearning/2 - Datasets e data preprocessing/"
                       "data/iris_nan.csv")
Y = iris_nan["class"].values
X = iris_nan.drop("class", axis=1).values
print(iris_nan)
#poco consigliato in quanto si andrebbero a perdere info
iris_drop =iris_nan.dropna()
print(iris_drop)
iris_drop =iris_nan.dropna(axis=1)
print(iris_drop)

#IMPUTAZIONE VALORI MANCANTI CON MEDIE
replace_with = iris_nan.mean()
iris_imp = iris_nan.fillna(replace_with)
print(iris_imp)

#metodo con la mediana
replace_with = iris_nan.median()
iris_imp = iris_nan.fillna(replace_with)
print("stampo la mediana", iris_imp)

#metodo con la moda ovvero con l'elemento più frequente
replace_with = iris_nan.mode().iloc[0]
iris_imp = iris_nan.fillna(replace_with)
print("stampo la moda", iris_imp)

#con np
imp = SimpleImputer(strategy="mean", missing_values=np.nan)
X_imp = imp.fit_transform(X)
print(X_imp)

nan_count = np.count_nonzero(np.isnan(X_imp))
print("Il dataset ha "+str(nan_count)+" valori mancanti")
