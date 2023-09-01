# Importing necessary libraries
import os
import pandas as pd
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from joblib import dump
from sklearn.preprocessing import MinMaxScaler

# Reading and inspecting the data
df = pd.read_csv('Crop_recommendation.csv')

# Extracting features and labels
x = df.drop(columns=['label'])
y = df['label']

# Splitting the data into training and test sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, shuffle=True, random_state=0)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define models
models = [
     ('Logistic Regression', LogisticRegression()),
    ('Random Forest', RandomForestClassifier()),
    ('SVM_RBF', SVC(probability=True)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('KNN_Euclidean', KNeighborsClassifier(n_neighbors=19, metric='euclidean')),
    ('Naive Bayes', GaussianNB()),
    ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
    ('AdaBoost', AdaBoostClassifier()),
    ('MLP Neural Network', MLPClassifier(max_iter=1000, early_stopping=True))
]
if not os.path.exists('trained_models'):
    os.makedirs('trained_models')

# Train and save models
for name, model in models:
    try:
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        dump(model, f"trained_models/{name.replace(' ', '_').lower()}_model.joblib")
    except Exception as e:
        print(f"Could not train {name}. Error: {e}")