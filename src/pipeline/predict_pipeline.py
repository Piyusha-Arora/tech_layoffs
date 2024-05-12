import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)    



class CustomData:
    def __init__(self, Company:str,	Location_HQ:str,	Country:str, 	Company_Size_before_Layoffs:int, 	Industry:str, 	Stage:str):
        self.Company=Company
        self.Location_HQ=Location_HQ
        self.Country=Country
        self.Company_Size_before_Layoffs=Company_Size_before_Layoffs
        self.Industry=Industry
        self.Stage=Stage

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Company":[self.Company],
                "Location_HQ":[self.Location_HQ],
                "Country":[self.Country],
                "Company_Size_before_Layoffs":[self.Company_Size_before_Layoffs],
                "Industry":[self.Industry],
                "Stage":[self.Stage]
            }

            return pd.DataFrame(custom_data_input_dict)

        except:
            raise CustomException(e,sys)