import os
import sys
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_paths = {
        f"target_{i + 1}": os.path.join("artifacts", f"model_target_{i + 1}.pkl") for i in range(3)
    }
    metrics_path = os.path.join("artifacts", "metrics.json")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_data, test_data):
        try:
            # Load data
            logging.info(f"Loading data from {train_data} and {test_data}")
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            # Define features and targets
            feature_columns = ["CO2", "CO2_1", "CO2_2", "CO2_3", "CO2_4"]
            target_columns = ["target_1", "target_2", "target_3"]

            # Split features and targets
            X_train = train_df[feature_columns]
            y_train = train_df[target_columns]
            X_test = test_df[feature_columns]
            y_test = test_df[target_columns]
            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            # Initialize models
            target_size = 3
            regs = [LinearRegression() for _ in range(target_size)]
            metrics = {target: {"r2": None, "mse": None, "mae": None} for target in target_columns}
            predictions = {target: None for target in target_columns}

            # Train and evaluate models
            for i, reg in enumerate(regs):
                target = f"target_{i + 1}"
                logging.info(f"Training model for {target}")

                # Train model
                reg.fit(X_train, y_train[target])
                save_object(self.model_trainer_config.trained_model_paths[target], reg)

                # Predict and evaluate
                y_predict = reg.predict(X_test)
                predictions[target] = y_predict.tolist()  # Convert to list for JSON serialization
                metrics[target]["mae"] = mean_absolute_error(y_test[target], y_predict)
                metrics[target]["mse"] = mean_squared_error(y_test[target], y_predict)
                metrics[target]["r2"] = r2_score(y_test[target], y_predict)
                logging.info(
                    f"{target} - R2: {metrics[target]['r2']:.4f}, MSE: {metrics[target]['mse']:.4f}, MAE: {metrics[target]['mae']:.4f}")

            # Save metrics
            with open(self.model_trainer_config.metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logging.info(f"Saved metrics to {self.model_trainer_config.metrics_path}")

            # Print summary
            print("\nModel Evaluation Metrics:")
            for target in target_columns:
                print(
                    f"{target}: R2={metrics[target]['r2']:.4f}, MSE={metrics[target]['mse']:.4f}, MAE={metrics[target]['mae']:.4f}")

            # Return metrics and predictions
            # result = {"metrics": metrics, "predictions": predictions}
            return metrics

        except Exception as e:
            raise CustomException(e, sys)
