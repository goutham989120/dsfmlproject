import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    RandomForestClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Optional LightGBM support
try:
    from lightgbm import LGBMRegressor
    lightgbm_available = True
except ImportError:
    lightgbm_available = False

# Lightweight rule-based explainer reused from prediction pipeline
try:
    from src.pipeline.predict_pipeline import reason_from_row
except Exception:
    def reason_from_row(row: pd.Series) -> str:
        try:
            reasons = []
            for col in ['RAG Reason + Observations', 'RAG Reason', 'RAG']:
                if col in row and pd.notna(row[col]):
                    reasons.append(str(row[col]))
            return "; ".join(reasons) if reasons else ''
        except Exception:
            return ''


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    comparison_plot_path = os.path.join("artifacts", "model_comparison.png")
    model_scores_csv = os.path.join("artifacts", "model_scores.csv")
    train_csv = os.path.join("artifacts", "train.csv")
    test_csv = os.path.join("artifacts", "test.csv")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def prepare_train_test(self, df: pd.DataFrame, target_col='RAG', test_size=0.2, random_state=42):
        """
        Splits dataframe into train/test and preserves actual RAG reason
        """
        try:
            if target_col not in df.columns:
                raise CustomException(f"Target column '{target_col}' not found in dataframe", sys)

            # Ensure actual_RAG_reason is always present
            if 'RAG Reason + Observations' in df.columns:
                df['actual_RAG_reason'] = df['RAG Reason + Observations']
            else:
                df['actual_RAG_reason'] = ''

            # Stratify split if target has multiple classes
            stratify_col = df[target_col] if df[target_col].nunique() > 1 else None
            train_df, test_df = train_test_split(df, test_size=test_size, stratify=stratify_col, random_state=random_state)

            # Save train/test for pipeline use
            os.makedirs(os.path.dirname(self.model_trainer_config.train_csv), exist_ok=True)
            train_df.to_csv(self.model_trainer_config.train_csv, index=False)
            test_df.to_csv(self.model_trainer_config.test_csv, index=False)

            logging.info(f"Train CSV saved to {self.model_trainer_config.train_csv} ({len(train_df)} rows)")
            logging.info(f"Test CSV saved to {self.model_trainer_config.test_csv} ({len(test_df)} rows)")

            # Return arrays for model training
            train_array = train_df.drop(columns=['actual_RAG_reason']).values
            test_array = test_df.drop(columns=['actual_RAG_reason']).values
            return train_array, test_array

        except Exception as e:
            logging.error("Error during train/test preparation")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array, tune_hyperparams=True):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Determine problem type
            try:
                pd.to_numeric(y_train, errors='raise')
                is_classification = False
            except Exception:
                is_classification = True

            # Drop ID-like first column if detected
            try:
                first_col_vals = train_array[:, 0]
                if first_col_vals.dtype == object:
                    sample = str(first_col_vals[0])
                    if sample.startswith('DSF') or 'DSFFLOW' in sample:
                        logging.info("Dropping identifier-like first column (assumed DSF Project ID)")
                        train_array = np.delete(train_array, 0, axis=1)
                        test_array = np.delete(test_array, 0, axis=1)
                        X_train, y_train, X_test, y_test = (
                            train_array[:, :-1],
                            train_array[:, -1],
                            test_array[:, :-1],
                            test_array[:, -1],
                        )
            except Exception:
                pass

            # ==================== MODEL DEFINITIONS ====================
            if is_classification:
                y_train = y_train.astype(str)
                y_test = y_test.astype(str)
                le = LabelEncoder()
                y_train_enc = le.fit_transform(y_train)
                try:
                    y_test_enc = le.transform(y_test)
                except ValueError as e:
                    raise CustomException(f"y_test contains unseen labels: {e}", sys)

                models = {
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "XGBoost": XGBClassifier(eval_metric='logloss'),
                    "CatBoost": CatBoostClassifier(verbose=False),
                }

                param_grids = {
                    "Random Forest": {"n_estimators": [200, 500], "max_depth": [None, 10, 20]},
                    "Decision Tree": {"max_depth": [None, 5, 10, 20]},
                    "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
                    "XGBoost": {"n_estimators": [200, 400], "learning_rate": [0.03, 0.1], "max_depth": [4, 6]},
                    "CatBoost": {"depth": [6, 8], "learning_rate": [0.03, 0.1], "iterations": [300]},
                }

            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "XGBoost": XGBRegressor(),
                    "CatBoost": CatBoostRegressor(verbose=False),
                    "AdaBoost": AdaBoostRegressor(),
                    "KNN": KNeighborsRegressor(),
                }

                if lightgbm_available:
                    models["LightGBM"] = LGBMRegressor()

                # Add Stacking Ensemble
                base_estimators = [
                    ('cat', CatBoostRegressor(iterations=500, depth=8, learning_rate=0.03, verbose=0)),
                    ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)),
                    ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
                ]
                models["Stacking Ensemble"] = StackingRegressor(estimators=base_estimators, final_estimator=RidgeCV())

                param_grids = {
                    "Random Forest": {"n_estimators": [200, 400], "max_depth": [None, 10, 20]},
                    "Decision Tree": {"max_depth": [None, 5, 10]},
                    "Gradient Boosting": {"n_estimators": [200, 400], "learning_rate": [0.03, 0.05, 0.1]},
                    "XGBoost": {"n_estimators": [300, 600], "learning_rate": [0.03, 0.05], "max_depth": [5, 8]},
                    "CatBoost": {"depth": [6, 8], "learning_rate": [0.03, 0.05], "iterations": [500]},
                    "LightGBM": {"n_estimators": [500, 1000], "learning_rate": [0.02, 0.05], "num_leaves": [31, 64]} if lightgbm_available else {},
                    "KNN": {"n_neighbors": [3, 5, 7]},
                    "AdaBoost": {"n_estimators": [100, 200], "learning_rate": [0.03, 0.1]},
                }

            # ==================== TRAINING LOOP ====================
            best_models = {}
            test_scores = {}

            for name, model in models.items():
                logging.info(f"Training model: {name}")
                if tune_hyperparams and param_grids.get(name):
                    try:
                        if is_classification:
                            from collections import Counter
                            class_counts = Counter(y_train_enc)
                            min_count = min(class_counts.values()) if class_counts else 0
                            if min_count < 3:
                                logging.info(f"Skipping GridSearchCV for {name}: too few samples (min_count={min_count})")
                                model.fit(X_train, y_train_enc)
                            else:
                                n_splits = 2 if min_count < 6 else 3
                                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                                grid = GridSearchCV(model, param_grids[name], cv=skf, scoring='accuracy', n_jobs=-1)
                                grid.fit(X_train, y_train_enc)
                                model = grid.best_estimator_
                        else:
                            grid = GridSearchCV(model, param_grids[name], cv=3, scoring='r2', n_jobs=-1)
                            grid.fit(X_train, y_train)
                            model = grid.best_estimator_
                    except Exception as ex:
                        logging.warning(f"GridSearchCV failed for {name}: {ex}")
                        model.fit(X_train, y_train_enc if is_classification else y_train)
                else:
                    model.fit(X_train, y_train_enc if is_classification else y_train)

                best_models[name] = model

                # Evaluate on test set
                if is_classification:
                    y_pred_test = model.predict(X_test)
                    score = accuracy_score(y_test_enc, y_pred_test)
                else:
                    y_pred_test = model.predict(X_test)
                    score = r2_score(y_test, y_pred_test)

                test_scores[name] = score

            # ==================== MODEL SELECTION ====================
            best_model_name = max(test_scores, key=test_scores.get)
            best_model = best_models[best_model_name]
            logging.info(f"✅ Best model: {best_model_name} (test score: {test_scores[best_model_name]:.4f})")

            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            # ==================== SAVE MODEL SCORES ====================
            os.makedirs(os.path.dirname(self.model_trainer_config.model_scores_csv), exist_ok=True)
            scores_df = pd.DataFrame(list(test_scores.items()), columns=["Model", "Score"])
            scores_df.to_csv(self.model_trainer_config.model_scores_csv, index=False)
            logging.info(f"Saved model scores to {self.model_trainer_config.model_scores_csv}")

            # ==================== PLOT COMPARISON ====================
            try:
                plt.figure(figsize=(10,6))
                sorted_scores = dict(sorted(test_scores.items(), key=lambda item: item[1], reverse=True))
                plt.bar(sorted_scores.keys(), sorted_scores.values(), color='skyblue')
                plt.xticks(rotation=45, ha='right')
                ylabel = 'Accuracy' if is_classification else 'R² Score'
                plt.ylabel(ylabel)
                plt.title(f'Model Comparison ({ylabel})')
                plt.tight_layout()
                os.makedirs(os.path.dirname(self.model_trainer_config.comparison_plot_path), exist_ok=True)
                plt.savefig(self.model_trainer_config.comparison_plot_path)
                plt.close()
                logging.info(f"Saved model comparison plot to {self.model_trainer_config.comparison_plot_path}")
            except Exception as e:
                logging.warning(f"Failed to save model comparison plot: {e}")

            return best_model_name, test_scores[best_model_name]

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)
