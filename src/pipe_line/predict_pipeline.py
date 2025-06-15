import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Define paths for three models
            model_paths = {
                "target_1": os.path.join("artifacts", "model_target_1.pkl"),
                "target_2": os.path.join("artifacts", "model_target_2.pkl"),
                "target_3": os.path.join("artifacts", "model_target_3.pkl")
            }

            # Load three models
            models = {}
            for target, model_path in model_paths.items():
                models[target] = load_object(file_path=model_path)

            # Validate input features
            required_columns = ["CO2", "CO2_1", "CO2_2", "CO2_3", "CO2_4"]
            if not all(col in features.columns for col in required_columns):
                raise CustomException(f"Input data missing required columns: {required_columns}", sys)

            # Prepare features
            X = features[required_columns]

            # Make predictions
            preds = {}
            for target, model in models.items():
                preds[target] = model.predict(X).tolist()

            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 CO2: float,
                 CO2_1: float,
                 CO2_2: float,
                 CO2_3: float,
                 CO2_4: float):
        self.CO2 = CO2
        self.CO2_1 = CO2_1
        self.CO2_2 = CO2_2
        self.CO2_3 = CO2_3
        self.CO2_4 = CO2_4

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CO2": [self.CO2],
                "CO2_1": [self.CO2_1],
                "CO2_2": [self.CO2_2],
                "CO2_3": [self.CO2_3],
                "CO2_4": [self.CO2_4]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == "__main__":
#     try:
#         # Example input from test set (1958-10-13)
#         custom_data = CustomData(
#             CO2=317.29,
#             CO2_1=317.49,
#             CO2_2=317.62,
#             CO2_3=317.93,
#             CO2_4=318.00
#         )
#         data_df = custom_data.get_data_as_data_frame()
#         print("\nInput Features:")
#         print(data_df)
#
#         # Initialize pipeline
#         pipeline = PredictPipeline()
#         predictions = pipeline.predict(data_df)
#         print("\nPredictions:")
#         print({
#             "target_1": predictions["target_1"][0],
#             "target_2": predictions["target_2"][0],
#             "target_3": predictions["target_3"][0]
#         })
#
#     except Exception as e:
#         print(f"Error: {e}")