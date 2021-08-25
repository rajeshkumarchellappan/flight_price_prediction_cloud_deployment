import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
df=pd.read_csv('flight_pred_df')
df.drop('Unnamed: 0',axis=1,inplace=True)
df.drop('Additional_Info',axis=1,inplace=True)
X=df.drop('Price',axis=1)
y=df['Price']
print(X.columns)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=20)
#Preparing ExtraTrees Regression
extr_model=ExtraTreesRegressor()
extr_model.fit(X_train,y_train)
y_pred_extr=extr_model.predict(X_test)
print(len(X_test.columns))
print(len(X_test.columns))
import pickle
# # Saving model to disk
pickle.dump(extr_model, open('model.pkl','wb'),protocol=pickle.HIGHEST_PROTOCOL)
model=pickle.load(open('model.pkl','rb'))
print(y_pred_extr)