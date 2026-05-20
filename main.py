import os
import sys
import warnings
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse

logging.basicConfig(level = logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual,pred):
    r2 =r2_score(actual,pred)
    mae = mean_absolute_error(actual,pred)
    mse = mean_squared_error(actual,pred)
    rmse = np.sqrt(mse)
    return r2,mae,rmse

def load_data():
    csv_url = ("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv")

    try:
        data =pd.read_csv(csv_url,sep =";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    return data

def split_data(data):
    # Split the data into training and test sets. (0.75, 0.25) split.
    train,test = train_test_split(data)
    return train,test

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    data = load_data()

    # train,test split
    train,test = split_data(data)

    # The predicted column is "quality" which is a scalar from
    X_train =train.drop(["quality"],axis =1)
    X_test = test.drop(["quality"],axis =1)
    y_train = train[["quality"]]
    y_test = test[["quality"]]

    #hypertuning using positional argument
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # mlflow start and model training
    with mlflow.start_run():
        model = ElasticNet(alpha =alpha,
                           l1_ratio =l1_ratio,
                           random_state =42)
        
        model.fit(X_train,y_train)

        y_predicted_qualities =model.predict(X_test)

        r2,rae,rmse =eval_metrics(y_test,y_predicted_qualities)

        print("ElasticNet Model alpha:{:f},l1_ratio:{:f}".format(alpha,l1_ratio))
        print("R2 : ",r2)
        print("MAE : ",rae)
        print("RMSE : ",rmse)

        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)

        mlflow.log_metric("r2",r2)
        mlflow.log_metric("mae",rae)
        mlflow.log_metric("rmse",rmse)

        predictions = model.predict(X_train)
        signature = infer_signature(X_train, predictions)

        ## For Remote server only(DAGShub)

        # remote_server_uri="https://dagshub.com/krishnaik06/mlflowexperiments.mlflow"
        # mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                model, "model", registered_model_name="ElasticnetWineModel"
            )
        else:
            mlflow.sklearn.log_model(model, "model")