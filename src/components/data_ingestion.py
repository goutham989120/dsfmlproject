import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
<<<<<<< HEAD
            df = pd.read_csv('notebook/data/2025-07-15 Project Progress Report.csv')
=======
            # Flexible input discovery: prefer date-prefixed files in notebook/data,
            # otherwise newest CSV/XLSX in notebook/data, then uploads, then './data.csv'
            input_path = None
            notebook_data_dir = os.path.join('notebook', 'data')
            candidates = []
            if os.path.isdir(notebook_data_dir):
                for fname in os.listdir(notebook_data_dir):
                    if fname.lower().endswith(('.csv', '.xlsx', '.xls')):
                        candidates.append(os.path.join(notebook_data_dir, fname))

            if candidates:
                import re
                date_pref = [p for p in candidates if re.match(r"^\d{4}-\d{2}-\d{2}", os.path.basename(p))]
                if date_pref:
                    input_path = max(date_pref, key=os.path.getmtime)
                else:
                    input_path = max(candidates, key=os.path.getmtime)
            else:
                # check uploads
                uploads_dir = os.path.join('uploads')
                uploads_candidates = []
                if os.path.isdir(uploads_dir):
                    for fname in os.listdir(uploads_dir):
                        if fname.lower().endswith(('.csv', '.xlsx', '.xls')):
                            uploads_candidates.append(os.path.join(uploads_dir, fname))
                if uploads_candidates:
                    input_path = max(uploads_candidates, key=os.path.getmtime)
                else:
                    if os.path.exists('data.csv'):
                        input_path = 'data.csv'

            if not input_path:
                raise FileNotFoundError("No input data file found in notebook/data, uploads, or project root (data.csv)")

            # If the chosen file came from notebook/data, log and print that selection
            try:
                if os.path.commonpath([os.path.abspath(input_path), os.path.abspath(notebook_data_dir)]) == os.path.abspath(notebook_data_dir):
                    picked_name = os.path.basename(input_path)
                    logging.info(f"Picked notebook/data file for ingestion: {picked_name}")
                    print(f"Picked notebook/data file for ingestion: {picked_name}")
            except Exception:
                # ignore commonpath errors on weird paths
                pass

            # read according to extension
            lower = input_path.lower()
            if lower.endswith('.csv'):
                df = pd.read_csv(input_path)
            elif lower.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(input_path)
            else:
                df = pd.read_csv(input_path)

>>>>>>> b1f066f345e977974f646e1d90d2d9df29ae5944
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data is saved")

            # If RAG exists and is intended as the target, stratify the split so
            # that class proportions are preserved in train and test.
            if 'RAG' in df.columns:
                try:
                    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['RAG'])
                except Exception:
                    # Fall back to simple split if stratify fails (e.g., very small classes)
                    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            else:
                train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                df
            )
        except Exception as e:
            logging.info("Exception occurred at data ingestion stage")
            raise CustomException(e, sys)
        
        


if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data,df = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data, test_data,df)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

    # After training completes and model artifacts are saved, run prediction pipeline
    try:
        from src.pipeline.predict_pipeline import predict_with_reasons

        # Use the raw data saved during ingestion as the input for predictions
        raw_input_path = obj.ingestion_config.raw_data_path
        print(f"Running prediction pipeline : {raw_input_path}")
        predict_with_reasons(input_csv=raw_input_path)
    except Exception as e:
        logging.info(f"Prediction pipeline skipped or failed: {e}")
    
    # After predictions are written, run the comparison script to generate compare files
    try:
        import subprocess
        # run compare_preds.py using the same Python executable
        compare_script = os.path.join(os.getcwd(), 'compare_preds.py')
        if os.path.exists(compare_script):
            logging.info(f"Running compare script: {compare_script}")
            completed = subprocess.run([sys.executable, compare_script], check=False, capture_output=True, text=True)
            logging.info(f"compare_preds.py exit code: {completed.returncode}")
            if completed.stdout:
                logging.info('compare_preds.py output:\n' + completed.stdout)
            if completed.stderr:
                logging.warning('compare_preds.py errors:\n' + completed.stderr)
        else:
            logging.info(f"compare_preds.py not found at {compare_script}, skipping comparison step")
    except Exception as e:
        logging.warning(f"Failed to run compare_preds.py: {e}")
    
    # Also run mismatch analysis script to generate per-ID mismatch details
    try:
        analyze_script = os.path.join(os.getcwd(), 'analyze_mismatches.py')
        if os.path.exists(analyze_script):
            logging.info(f"Running analyze script: {analyze_script}")
            completed2 = subprocess.run([sys.executable, analyze_script], check=False, capture_output=True, text=True)
            logging.info(f"analyze_mismatches.py exit code: {completed2.returncode}")
            if completed2.stdout:
                logging.info('analyze_mismatches.py output:\n' + completed2.stdout)
            if completed2.stderr:
                logging.warning('analyze_mismatches.py errors:\n' + completed2.stderr)
        else:
            logging.info(f"analyze_mismatches.py not found at {analyze_script}, skipping mismatch analysis")
    except Exception as e:
        logging.warning(f"Failed to run analyze_mismatches.py: {e}")

    # Generate prediction-derived confusion matrix PNG
    try:
        gen_script = os.path.join(os.getcwd(), 'gen_pred_confusion_matrix.py')
        if os.path.exists(gen_script):
            logging.info(f"Running prediction CM generation script: {gen_script}")
            completed3 = subprocess.run([sys.executable, gen_script], check=False, capture_output=True, text=True)
            logging.info(f"gen_pred_confusion_matrix.py exit code: {completed3.returncode}")
            if completed3.stdout:
                logging.info('gen_pred_confusion_matrix.py output:\n' + completed3.stdout)
            if completed3.stderr:
                logging.warning('gen_pred_confusion_matrix.py errors:\n' + completed3.stderr)
        else:
            logging.info(f"gen_pred_confusion_matrix.py not found at {gen_script}, skipping PNG generation")
    except Exception as e:
        logging.warning(f"Failed to run gen_pred_confusion_matrix.py: {e}")