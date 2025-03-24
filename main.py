import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('dataset.csv')
d1 = df.iloc[:, 0].tolist()
d2 = df.iloc[:, 1:].values.tolist()
d3 = df.columns[1:].tolist()
data = np.array(d2)
labels = np.array(d1)
x_train,x_test,y_train,y_test = train_test_split(data , labels , test_size = 0.2, random_state= 42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
kn=KNeighborsClassifier(n_neighbors=4)
kn.fit(x_train,y_train)
acc=kn.score(x_test,y_test)
print(f"Accuracy of model : {acc:.2f}")
l=[]
for i in d3:
    n = int(input(f"Enter marks for {i} subject : "))
    l.append(n)
l=np.array([l])
dist ,ind = kn.kneighbors(l)
r = kn.predict(l)
print(f"\nYou should choose {r[0]} stream, All the best for future!")
sc=list(set(labels[ind[0]]))
for i in range(len(sc)):
    if sc[i] != r[0]:
        print(f"Next best choice : {sc[i]}")
