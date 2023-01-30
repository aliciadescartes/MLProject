import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    try:
        data = pd.read_csv("/Users/aliciabeniddir/home-credit-default-risk/application_train.csv")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train_x = train.drop('TARGET', axis=1)
    test_x = test.drop('TARGET', axis=1)
    train_y = train["TARGET"]
    test_y = test["TARGET"]


    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

    with mlflow.start_run():
        lr = RandomForestClassifier(cpp_alpha=alpha)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        print("RandomForestClassifier model (alpha={:f}):".format(alpha))

        mlflow.log_param("alpha", alpha)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="RandomForestClassifierModel")
        else:
            mlflow.sklearn.log_model(lr, "model")
