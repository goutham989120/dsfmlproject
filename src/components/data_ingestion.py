import os
import sys
import glob
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

    def initiate_data_ingestion(self, input_path: str = None):
        """
        Initiate data ingestion.

        If input_path is provided it will be used. Otherwise this method will
        search a set of sensible locations for any CSV and pick the first match.
        The discovered file path will be stored in self.ingestion_config.raw_data_path
        so downstream code can reuse it.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Build candidate list: explicit arg, env var, then common locations
            candidates = []
            if input_path:
                candidates.append(input_path)

            env_path = os.environ.get('DSFML_INPUT') or os.environ.get('DSFML_INPUT_CSV')
            if env_path:
                candidates.append(env_path)

            # explicit common files / folders to search
            candidates.extend([
                os.path.join('uploads', '*.csv'),
                os.path.join('notebook', 'data', '*.csv'),
                os.path.join('.', 'data.csv'),
            ])

            chosen = None

            # Try direct candidate paths first
            for c in candidates:
                if not c:
                    continue
                # If candidate contains a glob pattern, search for matches
                if any(ch in c for ch in ['*', '?', '[']):
                    matches = glob.glob(c)
                    if matches:
                        # Prefer filenames that contain Project_Progress_Report_Status if available
                        preferred = [m for m in matches if 'Project_Progress_Report_Status' in os.path.basename(m)]
                        chosen = preferred[0] if preferred else matches[0]
                        break
                else:
                    if os.path.exists(c):
                        chosen = c
                        break

            # As a last resort, search the repo recursively for any CSV
            if chosen is None:
                matches = glob.glob(os.path.join('**', '*.csv'), recursive=True)
                # filter out files in .git or venv directories
                matches = [m for m in matches if '.git' not in m and 'venv' not in m]
                if matches:
                    preferred = [m for m in matches if 'Project_Progress_Report_Status' in os.path.basename(m)]
                    chosen = preferred[0] if preferred else matches[0]

            if chosen is None:
                raise FileNotFoundError(
                    'No input CSV found. Tried explicit input, DSFML_INPUT/DSFML_INPUT_CSV env, uploads/, notebook/data/, repo root.'
                )

            # Normalize chosen to a relative path
            chosen = os.path.normpath(chosen)
            logging.info(f"Reading dataset from: {chosen}")
            df = pd.read_csv(chosen)
            # update configured raw data path so downstream code can reference it
            self.ingestion_config.raw_data_path = chosen
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