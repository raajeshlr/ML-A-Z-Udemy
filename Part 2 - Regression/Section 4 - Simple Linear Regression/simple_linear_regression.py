import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

plt.scatter(X,y)
plt.show()

from sklearn.linear_model import LinearRegression
estimator = LinearRegression()
estimator.fit(X_train,y_train)

y_pred = estimator.predict(X_test)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,estimator.predict(X_train),color='blue')
plt.show()

plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,estimator.predict(X_train),color='blue')
plt.show()

from sklearn.metrics import r2_score
r2value = r2_score(y_test,y_pred)

from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_test,y_pred)

from sklearn.model_selection import learning_curve
np.random.seed(10)

#not working
#def randomize(X,Y):
 #   permutation = np.random.permutation(Y.shape[0])
  #  X2 = X[permutation,:]
   # Y2 = Y[permutation,:]
    #return X2,Y2

#X2,Y2 = randomize(X,y)

def draw_learning_curves(X,y,estimator,num_trainings):
    train_sizes,train_scores,test_scores = learning_curve(
            estimator,X,y,cv=None,n_jobs=1,train_sizes=np.linspace(0.1,11.0,num_trainings))
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    
    plt.grid()
    plt.title('Learning Curves')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    
    plt.plot(train_scores_mean,'o-',color='g',
             label='Training score')
    plt.plot(test_scores,'o-',color='y',
             label='cross_validation score')
    plt.legend(loc='best')
    plt.show()

from sklearn.metrics import mean_squared_error

trainerrorvalues = []
testerrorvalues = []

for samples in range(1,21):
    estimator.fit(X_train[0:samples],y_train[0:samples])
    ypredtrain = estimator.predict(X_train[0:samples])
    ypredtest = estimator.predict(X_test)
    trainerror = mean_squared_error(y_train[0:samples],ypredtrain)
    testerror = mean_squared_error(y_test,ypredtest)
    trainerrorvalues.append(trainerror)
    testerrorvalues.append(testerror)
    
    
plt.plot(range(1,21),trainerrorvalues,color='red')
plt.plot(range(1,21),testerrorvalues,color='blue')
plt.show()
    
    

    

























