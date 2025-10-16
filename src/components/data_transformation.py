import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self,df):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            # Identify column types
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            # Build OneHotEncoder in a way that's compatible with multiple sklearn versions
            try:
                # sklearn 1.2+ uses 'sparse_output'
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                # older sklearn versions use 'sparse'
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", ohe),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {cat_cols}")
            logging.info(f"Numerical columns: {num_cols}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_cols),
                ("cat_pipelines",cat_pipeline,cat_cols)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path, raw_df=None):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Strip whitespace from column names to avoid accidental mismatch
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Auto-detect target column: prefer 'RAG' if present (seen in dataset), else fallback to last column
            if 'RAG' in train_df.columns:
                target_column_name = 'RAG'
            else:
                # fallback: use last column as target
                target_column_name = train_df.columns[-1]

            logging.info(f"Using '{target_column_name}' as target column")

            # Drop target column from input features before creating preprocessor
            if target_column_name not in train_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in training data", sys)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Small cleaning: if target is 'RAG' and contains percent-like strings
            # (e.g., '75%', '299%'), convert those to the categorical RAG using
            # simple thresholds. This avoids putting percent strings into a
            # categorical encoder.
            def _perc_to_rag(val):
                try:
                    if pd.isna(val):
                        return val
                    s = str(val).strip()
                    if s.endswith('%'):
                        v = float(s.rstrip('%'))
                        # thresholds: tune as needed
                        if v >= 70:
                            return 'Green'
                        if v >= 40:
                            return 'Amber'
                        return 'Red'
                    return s
                except Exception:
                    return val

            if target_column_name == 'RAG':
                try:
                    target_feature_train_df = target_feature_train_df.apply(_perc_to_rag)
                    target_feature_test_df = target_feature_test_df.apply(_perc_to_rag)
                    logging.info('Converted percent-like RAG values to categories')
                except Exception:
                    logging.info('RAG conversion skipped')

            # Frequency-encode high-cardinality categorical columns to reduce dimensionality
            try:
                high_card_threshold = 20
                freq_encoded_cols = []
                # find categorical columns in the input features
                cat_cols_in_input = input_feature_train_df.select_dtypes(include=['object', 'category']).columns.tolist()
                for col in cat_cols_in_input:
                    n_unique = input_feature_train_df[col].nunique()
                    if n_unique > high_card_threshold:
                        # map to frequency (proportion) observed in train
                        freq = input_feature_train_df[col].value_counts(normalize=True)
                        input_feature_train_df[col] = input_feature_train_df[col].map(freq).fillna(0)
                        input_feature_test_df[col] = input_feature_test_df[col].map(freq).fillna(0)
                        freq_encoded_cols.append(col)

                if freq_encoded_cols:
                    logging.info(f"Applied frequency encoding to high-cardinality columns: {freq_encoded_cols}")
            except Exception:
                # if anything fails here, continue with original categorical values
                logging.info("Frequency encoding step failed or skipped")

            # Build preprocessor using only input features (no target column)
            preprocessing_obj = self.get_data_transformer_object(input_feature_train_df)

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # If transformer returns sparse matrices, convert to dense arrays
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            # Ensure target arrays are column vectors with shape (n_samples, 1)
            target_feature_train_array = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_array = np.array(target_feature_test_df).reshape(-1, 1)

            # Validate matching number of rows before concatenation
            if input_feature_train_arr.shape[0] != target_feature_train_array.shape[0]:
                raise CustomException(
                    f"Mismatch in training rows: features {input_feature_train_arr.shape[0]} vs target {target_feature_train_array.shape[0]}",
                    sys,
                )
            if input_feature_test_arr.shape[0] != target_feature_test_array.shape[0]:
                raise CustomException(
                    f"Mismatch in test rows: features {input_feature_test_arr.shape[0]} vs target {target_feature_test_array.shape[0]}",
                    sys,
                )

            # Concatenate features and target (both 2D)
            train_arr = np.c_[input_feature_train_arr, target_feature_train_array]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_array]

            logging.info(f"Saving preprocessing object to {self.data_transformation_config.preprocessor_obj_file_path}")
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)