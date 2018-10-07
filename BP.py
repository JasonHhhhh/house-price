from keras.models import Sequential
from keras.layers import Input,Dense,Activation,Dropout
from keras.optimizers import Adam,RMSprop,SGD
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc


# load dataset
with open('case2_training.csv') as f:
    data=pd.read_csv(f,header=0)
print(data)

X, y = data.iloc[:, 1 :-1], data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)



model=Sequential()

model.add(Dense(2000, activation='sigmoid',input_dim=8))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0001))

model.fit(X_train,Y_train,epochs=10,batch_size=128)
Y_test_pred=model.predict(X_test)
fpr, tpr, threshold = roc_curve(Y_test, Y_test_pred, pos_label=1)
auc_score = auc(fpr, tpr)
print(auc_score)


