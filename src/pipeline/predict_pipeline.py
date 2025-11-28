import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            # Loading model & preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transforming input data
            data_scaled = preprocessor.transform(features)

            # Predicting
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)
class CustomData:
    def __init__(
        self,
        Age: int,
        Gender: str,
        Daily_Screen_Time_hrs: float,
        Sleep_Quality: int,
        Stress_Level: int,
        Days_Without_Social_Media: int,
        Exercise_Frequency_week: int,
        Social_Media_Platform: str
    ):

        self.Age = Age
        self.Gender = Gender
        self.Daily_Screen_Time_hrs = Daily_Screen_Time_hrs
        self.Sleep_Quality = Sleep_Quality
        self.Stress_Level = Stress_Level
        self.Days_Without_Social_Media = Days_Without_Social_Media
        self.Exercise_Frequency_week = Exercise_Frequency_week
        self.Social_Media_Platform = Social_Media_Platform

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Daily_Screen_Time(hrs)": [self.Daily_Screen_Time_hrs],
                "Sleep_Quality(1-10)": [self.Sleep_Quality],
                "Stress_Level(1-10)": [self.Stress_Level],
                "Days_Without_Social_Media": [self.Days_Without_Social_Media],
                "Exercise_Frequency(week)": [self.Exercise_Frequency_week],
                "Social_Media_Platform": [self.Social_Media_Platform],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
