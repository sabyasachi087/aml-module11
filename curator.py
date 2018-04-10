from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def getTrainData():
    global X_train_scaled,y_train
    return X_train_scaled,y_train

def getTestData():
    global X_train_scaled,y_train
    return X_test_scaled,y_test