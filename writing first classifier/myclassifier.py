from scipy.spatial import distance
def euc(a,b):
    return distance.euclidean(a,b)

class scrappyKNN:
    def fit(self, x_train, y_train):
        self.x_train=x_train
        self.y_train=y_train
        
    def predict(self, x_test):
        predictions=[]
        for row in x_test:
            label=self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self,row):
        best_dist=euc(row, self.x_train[0])
        best_index=0
        for i in range(1,len(self.x_train)):
            dist=euc(row, self.x_train[i])
            if dist<best_dist:
                best_dist=dist
                best_index=i
        return self.y_train[best_index]
    
# importing dataset
from sklearn.datasets import load_iris
iris=load_iris()

x=iris.data    # features
y=iris.target  # labels
# print(x,y)

# spliting the data from training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33)

# # feature and label for training data
# print("X_train:", x_train, "X_test:", x_test)

# # feature and label for testing data
# print("y_train:", y_train, "y_test:", y_test)

# creating classifiers
from sklearn import tree
my_classifier=scrappyKNN()

# usin different classifiers for same task
# from sklearn.neighbors import KNeighborsClassifier
# my_classifier=KNeighborsClassifier()

# training our classifiers using training data
my_classifier.fit(x_train, y_train)

# call predict method and use it to classify our test data
predictions=my_classifier.predict(x_test)
# print(predictions)

# to calculate accuracy we compare the predicted label to the true label and talley up to the score
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, predictions)
print(accuracy)