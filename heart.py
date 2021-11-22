import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import random
data = pd.read_csv("heart.csv")
data["trestbps"]=np.log(data["trestbps"])

data=data.drop(["fbs"],axis=1)
data=data.drop(["ca"],axis=1)
data["chol"]=np.log(data["chol"])
target=data["target"]
print(data.shape[1])

np.random.shuffle(data.values)
data=data.drop(["target"],axis=1)
print(data.columns)
sc= StandardScaler()
data=sc.fit_transform(data)
X=data
y=target
from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())
print(knn_scores)
knn_classifier = KNeighborsClassifier(n_neighbors = 12)
knn_classifier.fit(X,y)

joblib.dump(knn_classifier,"model2")

