import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

def create_ts_data(data, window_size=5, target_size=3):
    try:
        i = 1
        while i < window_size:
            data[f"CO2_{i}"] = data["CO2"].shift(-i)
            i += 1
        i = 0
        while i < target_size:
            data[f"target_{i+1}"] = data["CO2"].shift(-i-window_size)
            i += 1
        data = data.dropna(axis=0)
        logging.info(f"Created time-series data with window_size={window_size}, target_size={target_size}")
        return data
    except Exception as e:
        raise CustomException(e, sys)

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # Read dataset
            df = pd.read_csv('data/co2.csv')
            logging.info("Read the dataset as dataframe")

            # Convert Date to datetime
            df["Date"] = pd.to_datetime(df["Date"])
            logging.info("Converted Date column to datetime")

            # Interpolate missing CO2 values
            df["CO2"] = df["CO2"].interpolate()
            logging.info("Interpolated missing CO2 values")

            # Ensure data is sorted by Date
            df = df.sort_values(by="Date").reset_index(drop=True)
            logging.info("Sorted data by Date")

            # Create time-series data
            window_size = 5
            target_size = 3
            df = create_ts_data(df, window_size, target_size)
            logging.info(f"Transformed data with window_size={window_size}, target_size={target_size}")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw processed data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Saved raw data to {self.ingestion_config.raw_data_path}")

            # Define targets
            targets = [f"target_{i+1}" for i in range(target_size)]
            logging.info(f"Target columns: {targets}")

            # Split features and targets
            x = df.drop(["Date"] + targets, axis=1)
            y = df[targets]
            logging.info(f"Features shape: {x.shape}, Targets shape: {y.shape}")

            # Chronological train-test split
            train_ratio = 0.8
            num_samples = len(x)
            train_idx = int(num_samples * train_ratio)

            x_train = x.iloc[:train_idx].copy()
            y_train = y.iloc[:train_idx].copy()
            x_test = x.iloc[train_idx:].copy()
            y_test = y.iloc[train_idx:].copy()
            logging.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

            # Concatenate features and targets, include Date
            train_set = pd.concat([df[["Date"]].iloc[:train_idx], x_train, y_train], axis=1)
            test_set = pd.concat([df[["Date"]].iloc[train_idx:], x_test, y_test], axis=1)
            logging.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Saved train data to {self.ingestion_config.train_data_path}")
            logging.info(f"Saved test data to {self.ingestion_config.test_data_path}")

            logging.info("Data ingestion completed")
            return (
                df,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    # res, _, _ = obj.initiate_data_ingestion()
    # print(res.head(10))
    # train_data,test_data = obj.initiate_data_ingestion()

    # data_transformation = DataTransformation()
    # # data_transformation.initiate_data_transformation(train_data, test_data)
    # train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    #
    # modeltrainer = ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr, test_arr))