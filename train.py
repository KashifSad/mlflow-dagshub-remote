import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt


import dagshub
dagshub.init(repo_owner='KashifSad', repo_name='mlflow-dagshub-remote', mlflow=True)



mlflow.set_tracking_uri("https://dagshub.com/KashifSad/mlflow-dagshub-remote.mlflow")



iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

max_depth = 15
n_estimators = 10

mlflow.set_experiment("kash-exp")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy",accuracy)


    
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion matrix")


    plt.savefig("Confusion-matrix.png")
    mlflow.log_artifact('confusion-matrix.png')


    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(rf,"Random forest")


    mlflow.set_tag("author","kashif")
    mlflow.set_tag('model',"random forest")

    print("accuracy",accuracy)