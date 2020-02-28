import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
plt.show()