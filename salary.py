#prediction of salaries based on Years of Experience
#import required libraries
import pandas as pd
import matplotlib.pyplot as plt

#import data using pandas library
df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values

#split the data into train and test datasets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

#fit the model to training data
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)

#predict the test set from model
pred_y = model.predict(X_test)
model.score(X_train,Y_train)
model.score(X_test,Y_test)

#visualize the output using the matplotlib library
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,model.predict(X_train),color='green')
plt.title('Salary preduction based on experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()