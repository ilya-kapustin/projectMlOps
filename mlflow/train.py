import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import boto3
import os
from hyperopt import fmin, tpe, hp, Trials


mlflow.set_tracking_uri("http://localhost:5000")

# Load Titanic dataset
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    endpoint_url='http://minio:9000',
    )
s3.download_file('datasets', 'train.csv', 'train.csv')
df = pd.read_csv("train.csv")

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())


# Split the dataset into training and test sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Define the features and target variable
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
target = "Survived"

# Preprocess the data
train_X = pd.get_dummies(train[features])
train_y = train[target]
test_X = pd.get_dummies(test[features])
test_y = test[target]

def objective(params):
    clf = RandomForestClassifier(**params)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    score = accuracy_score(test_y, y_pred)
    return 1 - score

space = {
    'n_estimators': hp.choice('n_estimators', range(10, 200)),
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'max_features': hp.choice('max_features', range(1, 5))
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

model = RandomForestClassifier(**best)
model.fit(train_X, train_y)
y_pred = model.predict(test_X)

# Evaluate the model on the test set
preds = model.predict(test_X)
accuracy = accuracy_score(test_y, preds)
precision = precision_score(test_y, preds)
recall = recall_score(test_y, preds)
f1 = f1_score(test_y, preds)


# Log the model parameters and metrics
with mlflow.start_run(experiment_id=1):
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("data", "titanic")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")
