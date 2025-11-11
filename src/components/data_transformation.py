import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTrandformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_trandformation_config=DataTrandformationConfig()

    def get_data_transformer_object(self):
#        '''
#        This function is responsible for data transformation
#        '''
        try:
            numerical_columns = ['Age', 'Daily_Screen_Time(hrs)',
                'Sleep_Quality(1-10)', 'Stress_Level(1-10)',
                'Days_Without_Social_Media', 'Exercise_Frequency(week)']
            categorical_columns = ['Gender', 'Social_Media_Platform']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
            
                ]


            )

            logging.info("numarical columns: {numerical_columns}")
            logging.info("categorical columns: {categorical_columns}")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("The read train & test data")

            logging.info("obtaning preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name='Happiness_Index(1-10)'
            numerical_columns = ['Age', 'Daily_Screen_Time(hrs)',
                'Sleep_Quality(1-10)', 'Stress_Level(1-10)',
                'Days_Without_Social_Media', 'Exercise_Frequency(week)']
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Appling preprocesssing object on traning datafram & testing dataframe")
     
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Save preprocesing object.")

            save_object(
                file_path=self.data_trandformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_trandformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)

        





            
       