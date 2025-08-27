import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import seaborn as sn 

data = pd.read_excel("data.xlsx")
print(data.shape)
print(type(data))

plt.hist(data['time'], bins=100)
plt.xlabel("Nanopore formation time (s)")
plt.ylabel("Frequency (-)")
plt.show()


plt.scatter(data['time'], data['probability'])
plt.xlabel('Nanopore formation time (s)')
plt.ylabel('Nanopore probability (-)')
plt.show()


X = data.drop(columns=['time', 'probability'])
y1 = data['time']
y2 = data['probability']
reg1 =  LinearRegression().fit(X, y1)
reg1_score = reg1.score(X, y1)
print(reg1_score)
print(reg1.coef_)
print(reg1.intercept_)
ypred1 = reg1.predict(X)
print(ypred1)
print(sklearn.metrics.mean_absolute_error(y1, ypred1))
print(np.sqrt(sklearn.metrics.mean_squared_error(y1, ypred1)))
print(sklearn.metrics.r2_score(y1, ypred1))


X1_train, X1_test, y1_train, y1_test= sklearn.model_selection.train_test_split(X, y1, test_size=0.25, random_state=42)
reg1_train = LinearRegression().fit(X1_train, y1_train)
reg1_train_score = reg1_train.score(X1_train, y1_train)
print(reg1_train_score)
print(reg1_train.coef_)
print(reg1_train.intercept_)

y1_train_pred = reg1_train.predict(X1_train)
y1_test_pred = reg1_train.predict(X1_test)
print(y1_train_pred)
print(y1_test_pred)
print(sklearn.metrics.r2_score(y1_test, y1_test_pred))
print(sklearn.metrics.mean_absolute_error(y1_test, y1_test_pred))




# print(sklearn.metrics.mean_absolute_error(y1, pred1))
# print(np.sqrt(sklearn.metrics.mean_squared_error(y1, pred1)))
# print(sklearn.metrics.r2_score(y1, ypred1))

plt.scatter(y1_train, y1_train_pred, c="blue", alpha=0.5)
plt.scatter(y1_test, y1_test_pred, c="green", alpha=0.5)
plt.xlabel("True formation time (s)")
plt.ylabel("Predicted formation time (s)")
parity_x = np.linspace(90, 230, 100)
plt.plot(parity_x, parity_x, '--')
plt.legend(["train", "test", "y=x"])
plt.show()

train_mae = sklearn.metrics.mean_absolute_error(y1_train, y1_train_pred)
train_r2 = sklearn.metrics.r2_score(y1_train, y1_train_pred)
test_mae = sklearn.metrics.mean_absolute_error(y1_test, y1_test_pred)
test_r2 = sklearn.metrics.r2_score(y1_test, y1_test_pred)


print("Train MAE = ", train_mae)
print("Train R^2 = ", train_r2)
print("Test MAE = ", test_mae)
print("Test R^2 = ", test_r2)
box_style = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(200, 100, "Test $R^2$: {:.2f}".format(round(test_r2, 2),{'color':'blue'}))

array = np.arange(1, 23)
# Plot the coefficients
plt.barh(array, reg1_train.coef_, color='blue')
plt.xlabel("Feature weight")
plt.ylabel("Feature name")
plt.title("Linear Regression Coefficients")
plt.yticks(array, list(X))
plt.show()


reg1_ridge = Ridge(alpha=1.0)
reg1_ridge.fit(X1_train, y1_train)
print(Ridge())

y1_ridge_train_pred = reg1_ridge.predict(X1_train)
y1_ridge_test_pred = reg1_ridge.predict(X1_test)

plt.scatter(y1_train, y1_ridge_train_pred, c="blue", alpha=0.5)
plt.scatter(y1_test, y1_ridge_test_pred, c="green", alpha=0.5)
plt.xlabel("True formation time (s)")
plt.ylabel("Predicted formation time (s)")
parity_x = np.linspace(90, 230, 100)
plt.plot(parity_x, parity_x, '--')
plt.legend(["train", "test", "y=x"])
plt.show()

train_mae = sklearn.metrics.mean_absolute_error(y1_train, y1_ridge_train_pred)
train_r2 = sklearn.metrics.r2_score(y1_train, y1_ridge_train_pred)
test_mae = sklearn.metrics.mean_absolute_error(y1_test, y1_ridge_test_pred)
test_r2 = sklearn.metrics.r2_score(y1_test, y1_ridge_test_pred)

print("Train MAE = ", train_mae)
print("Train R^2 = ", train_r2)
print("Test MAE = ", test_mae)
print("Test R^2 = ", test_r2)

reg2_ridge = Ridge(alpha=1.0)
reg2_ridge.fit(X1_train, y1_train)
print(Ridge())

y1_ridge_train_pred = reg1_ridge.predict(X1_train)
y1_ridge_test_pred = reg1_ridge.predict(X1_test)

plt.scatter(y1_train, y1_ridge_train_pred, c="blue", alpha=0.5)
plt.scatter(y1_test, y1_ridge_test_pred, c="green", alpha=0.5)
plt.xlabel("True formation time (s)")
plt.ylabel("Predicted formation time (s)")
parity_x = np.linspace(90, 230, 100)
plt.plot(parity_x, parity_x, '--')
plt.legend(["train", "test", "y=x"])
plt.show()

train_mae = sklearn.metrics.mean_absolute_error(y1_train, y1_ridge_train_pred)
train_r2 = sklearn.metrics.r2_score(y1_train, y1_ridge_train_pred)
test_mae = sklearn.metrics.mean_absolute_error(y1_test, y1_ridge_test_pred)
test_r2 = sklearn.metrics.r2_score(y1_test, y1_ridge_test_pred)

print("Train MAE = ", train_mae)
print("Train R^2 = ", train_r2)
print("Test MAE = ", test_mae)
print("Test R^2 = ", test_r2)


g = sn.PairGrid(X)
g.map(sn.scatterplot)
plt.show()
print(data.corr())
corr_matrix = data.corr()
plt.figure(figsize = (32,20))
f=sn.heatmap(corr_matrix, annot=True)
plt.show()