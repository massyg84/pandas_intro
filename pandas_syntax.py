import pandas as pd
import numpy as np

# questo progetto è atto a dare una panoramica sulla libreria
# PANDAS per la gestione dei dati in PYTHON

# la libreria pandas da l'opportunità di leggere un file (in vari formati)
# per estrarre i dati molte volte espressi in tabelle o raggruppati in vari formati
iris = pd.read_csv("/Users/massimilianoguida/Repo_Corso_DeepLearning/2 - Datasets e data preprocessing/data/iris.csv")

# head prende le prime n righe della tabella considerata
print(iris.head())

# tail prende le ultime n righe della tabella considerata
print(iris.tail(10))

# Se avessimo una tabella senza header, si potrebbero definire i nomi delle
# colonne manualmente
iris_noheader = \
    pd.read_csv("/Users/massimilianoguida/Repo_Corso_DeepLearning/2 - Datasets e data preprocessing/data/iris_noheader.csv"
                , header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
print(iris_noheader.tail(10))

# per accedere alle chiavi delle varie colonne, cioè ai nomi delle features, si usa l'atrtributo columns
print(iris.columns)

# per accedere alle meta informazioni generiche sul dataframe che abbiamo creato si usa il metodo info()
print(iris.info())

# si può accedere ad una singola colonna usando la sua chiave
# in questo caso Y non sarà più un dataframe ma è un altra struttura dati pandas chiamata SERIES
# in sostanza una series è un array monodimensionale ed un dataframe è un insieme di series
# in questo modo possiamo dar luogo a strutture sottoinsiemi dell'insieme completo del dataframe
Y = iris['species']
print(Y.head())

# un esempio di sottoinsieme di dataframes con un insieme arbitrario di series
X = iris[["sepal_length", "sepal_width", "petal_length"]]
print(X.head())

# per creare un dataframe filtrando le series da visualizzare si usa il metodo drop
Z = iris.drop("species", axis=1)
print(Z.head())
# il parametro axis compare in tutti quei metodi che operano per righe o per colonne
# quando l'operazione del metodo deve svolgersi sulle righe axis deve corrispondere a 1
# come in questo caso in cui vogliamo rimuovere la colonna species per ogni riga
# quando invece l'operazione deve essere svolta per colonne Axis deve corrispondere a 0


#MIN 5:38 SLICING
# ci sono 2 modi di effettuare lo slicing
#   1. loc  -> seleziona per label
#   2. iloc -> seleziona per indice
# facciamo per prima cosa una copia del dataset
iris_sampled = iris.copy()
# mescoliamola utilizzando il metodo sample e l'attributo frac = 1 per tornare il dataset per intero
iris_sampled = iris_sampled.sample(frac=1)
print(iris_sampled.head())

# iloc -> seleziona per indice
print("\n", iris_sampled.iloc[3])

# loc  -> seleziona per label
print("\n", iris_sampled.loc[28])

# per selezionare, con loc il valore di una specifica riga relativo anche ad una specifica colonna
# sarà necessario inserire un secondo parametro relativo alla colonna d'interesse
print("\n", iris_sampled.loc[28, "species"])

# si può selezionare, con iloc, le prime 10 righe e la colonna selezionata mediante l'indice di colonna
print("\n", iris_sampled.iloc[:10, 4])


# PANORAMICA SULLA GESTIONE DI GENERICI DATI STATISTICI

# shape è un attributo che coniste in una tupla di 2 elementi, il primo che sono le osservazioni (le righe) ed il
# secondo che sono le features (le colonne)
print("\n", iris.shape)

#describa è un metodo che restituisce una serie di informazioni statistiche d'interesse per ogni dato nel dataset
print("\n", iris.describe())
print("\n", iris["sepal_length"].describe())
print("\n", iris.max())
print("\n", iris["sepal_length"].max())

# unique estrapola la classe dei valori specifici di una features
print("\n", iris['species'].unique())

#MIN 10:00 applicare dei filtri usando le maschere (utilizzo delle maschere)

# Creare una maschera per selezionare le osservazioni che hanno la lunghezza del petalo
# maggiore della lunghezza media
long_petal_mask = iris["petal_length"] > iris["petal_length"].mean()
print("\n", long_petal_mask)

# per applicare la maschera basta creare un dataframe applicando la maschera al dataframe di origine (iris)
iris_long_petal = iris[long_petal_mask]
print(iris_long_petal.head(150))

# si crei una maschera che ci permetta di cambiare il valore della specie da “setosa” a “undefined”
iris_copy = iris.copy()
# si noti che viene applicata la maschera allo stesso dataframe
iris_copy[iris_copy["species"] == "setosa"] = "undefined"
print(iris_copy["species"].unique())

# OPERAZIONI ARITMETICHE

# effettuiamo una normalizzazione delle features numeriche
# per prima cosa escludiamo la colonna delle specie che non è numerica
X = iris.drop("species", axis=1)
# per eseguire la normalizzazione sarà necessario sottrarre ogni elemento per il valore minimo e dividere il tutto
# per la differenza tra valore massimo e valore minimo.
###############################
#              X - Xmin       #
#    Xnorm = -----------      #
#            Xmax - Xmin      #
###############################
Xnorm = (X - X.min()) / (X.max() - X.min())
print(Xnorm.head())

# ORDINAMENTO tramite la sort_values(feature)
print("\n", iris.sort_values("petal_length").head(10))

# Raggruppamento mediante la groupby e stampa della media degli altri valori per ogni specie
grouped_species = iris.groupby("species")
print("\n", grouped_species.mean())

# APPLICAZIONE di funzioni ad un dataframe
# ad esempio applichiamo la funzione numpy per contare valori differenti da zero per riga
# il risultato è una series quindi possiamo ricavarne i primi valori grazie alla funzione head
print("\n", iris.apply(np.count_nonzero, axis=1).head(10))

print("\n", iris.apply(np.count_nonzero, axis=0).head(10))

# applichiamo la funzione valore per valore ad esempio
# proviamo ad arrotondare i valori del dataframe all'intero più vicino
# per prima cosa leviamo species in quanto non è numerico
X = iris.drop("species", axis=1)
# poi utilizziamo applymap usando una funzione lamda (funzione usa e getta per essere utilizzata una volta
# solo quando il cursore ci passa sopra (rivedere le funzioni lamda)
X = X.applymap(lambda val: int(round(val, 0)))
print("\n", X.head())

# GESTIONE ERRORE VALORI INVALIDI NEL DATASET
# per testare fillna si vanno ad inserire valori invalidi
# per prima cosa si usa una copia del dataframe
iris_nan = iris.copy()
# creiamo un vettore di 10 valori casuali interi
sample = np.random.randint(iris.shape[0], size=10)
# usiamo loc per modificare in None la la feature in corrispondenza di queste 10 osservazioni
iris_nan.loc[sample, "petal_length"] = None
# per verificare se la modifica è andata a buon fine andiamo a contare il numero di
# valori null all'interno di iris_nan in corrispondenza della feature petal_length
print("\n", iris_nan["petal_length"].isnull().sum())
# ora possiamo applicare FILLNA
# sostituiamo questi valori invalidi con il valore medio della lunghezza del petalo
# calcolo valore medio
mean_petal_length = iris_nan["petal_length"].mean()
print("\n", mean_petal_length)
# possiamo applicare fillna. Questa istruzione fa tornare una nuova series
# e non verrà modificato il valore della colonna petal_length
# per farlo o eseguiamo un assegnazione
#         iris_nan["petal_length"] = iris_nan["petal_length"].fillna(mean_petal_length)
# oppure ancora potremmo usare il parametro inplace
iris_nan["petal_length"].fillna(mean_petal_length, inplace=True)

# eseguendo il tutto andiamo a controllare i valori nulli dopo il fillna (dovrebbero essere 0)
print("\n", iris_nan["petal_length"].isnull().sum())


import matplotlib.pyplot as plt

iris.plot(x="sepal_length", y="sepal_width", kind="scatter")
# show su plt mostra il plot creato
plt.show()
