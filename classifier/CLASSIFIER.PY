# importing decision tree classifier
from sklearn import tree
feature= [[140,0],[130,0],[150,1],[170,1]]
labels=["apple","apple","orange","orange"]

clf=tree.DecisionTreeClassifier()
clf=clf.fit(feature,labels)
print(clf.predict([[140,0]]))

from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
tree.plot_tree(clf, feature_names=["weight", "texture"], class_names=["apple", "orange"], filled=True)
plt.show()
