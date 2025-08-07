# Machine Learning Recipes
- This repository contains code examples and key concepts of ML. The series aims to teach machine learning from scratch, focusing on practical implementation with open-source libraries like scikit-learn and TensorFlow.
## Introduction
The "Machine Learning Recipes" series is designed to guide you through the fundamentals of Machine Learning, starting from basic concepts to more advanced topics. It demystifies machine learning by showing how algorithms learn from examples and experience, rather than relying on hard-coded rules. We explore various types of classifiers, feature engineering techniques, and how to build and evaluate machine learning models.

## Introduction to Machine Learning
Machine Learning is a subfield of artificial intelligence (AI) focusing on algorithms that learn from examples and experience, rather than relying on explicit, hard-coded rules. This allows programs to solve numerous problems without needing to be rewritten for each specific task. A core concept in ML is the classifier, which can be thought of as a function that takes input data and assigns a label as output.
## Core Concepts
**Supervised Learning:** 
This technique for automatically writing classifiers begins with examples of the problem you want to solve. It's about learning a function, or a mapping from input (features) to output (labels), by adjusting a model's parameters based on training data.

**Features and Labels:**
    - Features (X): These are the input measurements or attributes that describe the data. They represent your knowledge about the world for the classifier.
    
    - Labels (Y): This is the output we want to predict or the class of the data.
    
    - Good Features: A good feature makes it easy to discriminate between different types of output. They should be informative, independent (not highly correlated with other features), and easy to understand. Classifiers are only as good as the features provided.
    
**Training and Testing Data:** To verify a model's performance on unseen data, a dataset is typically partitioned into two parts:

    - Training Data (X_train, y_train): Used to train the model, allowing the algorithm to find patterns. The more training data, the better the classifier.
    
    - Testing Data (X_test, y_test): Kept separate and used to evaluate how accurate the trained model is on new, unseen examples. This helps ensure the model works well before deployment.
    
**Model and Parameters:** A model is a prototype or set of rules that define the body of a function. It typically has parameters that can be adjusted using training data. Learning involves iteratively adjusting these parameters to make the model more accurate.

## Classifiers and Techniques
**Decision Trees**
Decision Trees are a type of classifier known for being interpretable and easy to understand, allowing you to see exactly why a decision is made.
- Learning Algorithm (CART): The CART (Classification and Regression Trees) algorithm is used to build decision trees from data. It provides a procedure to decide which true/false questions to ask and when.
  
- Node Splitting: Each node asks a true/false question about a feature, partitioning the data into two subsets (true rows and false rows). The goal is to unmix the labels as you proceed down the tree, producing the purest possible distribution of labels at each node.
  
- Metrics for Purity:
  
    - Gini Impurity: A metric ranging from 0 to 1 (lower is better) that quantifies the amount of uncertainty or mixing at a node. A Gini impurity of 0 means no mixing (e.g., all apples in a set).
      
    - Information Gain: Quantifies how much a question reduces uncertainty. It's calculated as the uncertainty of the starting set minus the weighted impurity of the child nodes after a split. The question that produces the most gain is selected as the best one for that node.
      
- Structure: The tree continues dividing data until no further questions can be asked (information gain is zero), at which point a leaf node is added, providing a prediction based on the ratio of labels in the data that reached it.
  
**K-Nearest Neighbors (KNN)**
K-Nearest Neighbors is a simple classifier where predictions are made by finding the 'k' closest training points to a new testing point and predicting the majority class among those neighbors.

- Euclidean Distance: The straight-line distance between two points, analogous to the Pythagorean Theorem, is used to measure closeness. This formula works regardless of the number of features/dimensions.
  
- Pros: Relatively easy to understand and works reasonably well for some problems.
- Cons: Can be slow as it iterates over every training point for each prediction, and it doesn't have an easy way to represent feature importance.
  
**Linear Classifiers**
Linear classifiers, such as those used for the MNIST handwritten digit classification, are trained to predict which of several classes an input belongs to.
- Mechanism: They work by adding up "evidence" for each possible digit. Each pixel (feature) flows into an input node, travels along edges, and is multiplied by a weight on that edge. Output nodes gather this evidence, and the digit with the most evidence is predicted.
- Weights: The "important part is the weights". They start randomly and are gradually adjusted during training (in the fit method) to achieve accurate classifications.
- Visualization of Weights: Visualising weights can provide intuition into how the classifier works. Positive weights (e.g., red) indicate evidence for a digit, while negative weights (e.g., blue) indicate evidence against it. For instance, looking at the weights for the digit '1', one can see an outline of the digit formed by positive weights in the central column where a '1' is typically drawn.
  
**Neural Networks and Deep Learning**
Deep Learning is a branch of machine learning that has led to significant advancements, especially in domains like image classification. Neural networks are a type of classifier used in deep learning that can learn more complex functions.

- Feature Extraction: A major advantage of deep learning for images is that you don't need to manually extract features like textures or shapes; instead, you can use the raw pixels of the image as features, and the classifier handles the rest.
  
- Image Classification Example (MNIST): Classifying handwritten digits from the MNIST dataset is considered the "Hello World" of computer vision. Images are low-resolution (28x28 pixels in grayscale) and properly segmented (each contains exactly one digit). A 28x28 image has 784 pixels, which means 784 features (when flattened into a 1D array).
  
- TensorFlow for Poets: This code lab simplifies image classification by using retraining on an existing, highly accurate model called Inception, trained on millions of images. This allows for the creation of a new, high-accuracy classifier with significantly less training data and time.
  
- Training Data Quality: For good image classifiers, diversity and quantity of training data are key. Diverse images (different angles, lighting, colours) and a large quantity of images improve accuracy.
  
**Feature Engineering**
Feature engineering involves transforming raw data into a more useful representation for the classifier. It's one of the most important contributions to an ML experiment.

- Bucketing: Transforms a numeric feature (e.g., age) into several categorical ones based on ranges. This allows linear models to capture non-linear relationships by learning different weights for each bucket.

- Categorical Features: For features with a small number of discrete values (e.g., education level), using the raw value directly is effective.
  
- Feature Crossing: Creates new features by combining existing ones. This can be particularly helpful for linear classifiers that cannot naturally model interactions between features.
  
- Hashed Feature Columns: An efficient way to represent categorical features with a large vocabulary, especially when the vocabulary isn't known in advance. A hash function computes the bit automatically. This can save programming time and limit memory usage.
  
- Embeddings: A powerful technique for categorical data in deep learning. An embedding is a vector that represents the "meaning" of a categorical value (e.g., a word). They are learned automatically during DNN training and can compress representations, allowing the classifier to learn general concepts rather than memorising specific values. For example, job titles like "programmer" and "software engineer" could have similar embeddings.
  
**Tools and Libraries**
- TensorFlow: An open-source machine learning library, especially useful for deep learning.
- TF.Learn: A high-level machine learning library built on top of TensorFlow, offering a syntax similar to scikit-learn.
- Scikit-learn: An open-source library for machine learning, used for various tasks including data importing, splitting, and classifier implementation.
- Docker: Used for configuring the TensorFlow environment by providing pre-configured TensorFlow images.
- IPython/Jupyter Notebook: An interactive computing environment often used for experimenting with ML code.
- Matplotlib: A Python plotting library used for displaying images and visualising data.
- Facets: A tool for visualising what feature transformations do, particularly helpful with census data.
- TensorFlow Embedding Projector: An online tool to visualise datasets of word embeddings.
- Scipy: A Python library used for scientific computing, including functions like Euclidean Distance
