# IMPORTING DATASET

# importing iris dataset
from sklearn.datasets import load_iris
iris=load_iris()

# the dataset include both the table and the metadata
print("feature_names: ",iris.feature_names,"\n")
print("target names: ",iris.target_names,"\n")

print("sample data point with feature names for 1st entry: ")
for feature_name, value in zip(iris.feature_names,iris.data[0]):
    print(f"{feature_name}:{value}")

print("\n printing 1st entry: ",iris.data[0])

print("\nCorresponding target (label):")
print(f"target index: {iris.target[0]}")
print(f"target name: {iris.target_names[iris.target[0]]} \n")

# TRAIN A CLASSIFIER
from sklearn import tree
import numpy as np
test_idx=[0,50,100]

# trainin data
train_target=np.delete(iris.target, test_idx)
train_data=np.delete(iris.data, test_idx, axis=0)

# testing data
test_target=iris.target[test_idx]
test_data=iris.data[test_idx]

# creating decision tree classifier & train it our training data
clf=tree.DecisionTreeClassifier()
clf=clf.fit(train_data,train_target)     # now we can predict data as per choice of training data

# PEREDICT THE LABEL FOR THE NEW FALOWER
# use the tree to classify our testing data
print(test_target)
# now we see what the tree predict on this testing data
print(clf.predict(test_data))

# VISUALIZE THE TREE
"""
visulaize code.....

"""
# taking example from our testing data
print(test_data[1], test_target[1])
print(train_data[0], train_target[0])

# We can find the meta data by looking at the metd data
print(iris.feature_names,iris.target_names)
