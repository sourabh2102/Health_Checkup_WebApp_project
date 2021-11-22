import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
# DATA FOR PRED
data=pd.read_csv("diabetes.csv")
print(data.head())


logreg=LogisticRegression()



X=data.iloc[:,:8]
print(X.shape[1])
fill_values = SimpleImputer(missing_values=0, strategy="mean")

X = fill_values.fit_transform(X)
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
y=data[["Outcome"]]
classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(X,y)
#logreg.fit(X,y.reshape(-1,))
random_search.best_estimator_
joblib.dump(random_search,"model1")

