import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # detect if target is categorical (non-numeric)
            is_classification = False
            try:
                # y_train is the last column of train_array
                sample_target = train_array[:, -1]
                # if dtype is object or contains strings, treat as classification
                if sample_target.dtype == object:
                    is_classification = True
            except Exception:
                pass

            if is_classification:
                # lightweight classifiers for categorical target
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.tree import DecisionTreeClassifier

                models = {
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                }
            else:
                models = {
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "K-Neighbors Regressor": KNeighborsRegressor(),
                    "XGB Regressor": XGBRegressor(),
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                }

            model_report, X_test_proc, y_test_proc, is_classification, label_encoder = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models
            )

            if not model_report:
                logging.warning("No models produced a report. Skipping model selection.")
                return None

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            if best_model_score < 0.6:
                logging.warning(f"No model reached the quality threshold. Proceeding with best model {best_model_name} with score {best_model_score}")
            else:
                logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Use the best model to predict on the processed X_test used during evaluation
            predicted = best_model.predict(X_test_proc)

            # Compute same metric used in evaluate_models
            if is_classification:
                final_score = accuracy_score(y_test_proc, predicted)
            else:
                final_score = r2_score(y_test_proc, predicted)

            # Print final score to stdout (keeps parity with earlier behavior)
            print(final_score)
            return final_score

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)