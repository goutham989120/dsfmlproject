import os
import sys
from dataclasses import dataclass
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# classifiers for classification branch
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, tune_hyperparams=True):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Detect if target is categorical (optional)
            is_classification = False
            if train_array[:, -1].dtype == object:
                is_classification = True

            # Optionally drop identifier-like column (e.g., DSF Project ID) if present
            # Heuristic: if first column contains strings starting with 'DSF' or 'DSFFLOW', drop it
            try:
                first_col_vals = train_array[:, 0]
                if first_col_vals.dtype == object:
                    sample = str(first_col_vals[0])
                    if sample.startswith('DSF') or 'DSFFLOW' in sample:
                        logging.info("Dropping identifier-like first column from features (assumed DSF Project ID)")
                        # remove first column from train and test arrays
                        train_array = np.delete(train_array, 0, axis=1)
                        test_array = np.delete(test_array, 0, axis=1)
                        # recompute splits
                        X_train, y_train, X_test, y_test = (
                            train_array[:, :-1],
                            train_array[:, -1],
                            test_array[:, :-1],
                            test_array[:, -1],
                        )
            except Exception:
                pass

            # Define models and hyperparameter grids
            if is_classification:
                # Encode target labels
                le = LabelEncoder()
                y_train_enc = le.fit_transform(y_train)
                y_test_enc = le.transform(y_test)

                # Classifiers and grids - add higher-capacity models
                models = {
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                    "CatBoost": CatBoostClassifier(verbose=False),
                }

                param_grids = {
                    "Random Forest": {"n_estimators": [200, 500], "max_depth": [None, 10, 20]},
                    "Decision Tree": {"max_depth": [None, 5, 10, 20]},
                    "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
                    "XGBoost": {"n_estimators": [100, 300], "learning_rate": [0.01, 0.1], "max_depth": [3, 6]},
                    "CatBoost": {"depth": [4, 6], "learning_rate": [0.03, 0.1], "iterations": [200]},
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

                param_grids = {
                    "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
                    "Decision Tree": {"max_depth": [None, 5, 10]},
                    "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]},
                    "Linear Regression": {},  # No hyperparameters
                    "K-Neighbors Regressor": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
                    "XGB Regressor": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]},
                    "CatBoosting Regressor": {"depth": [4, 6, 8], "learning_rate": [0.01, 0.05, 0.1], "iterations": [100, 200]},
                    "AdaBoost Regressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.05, 0.1]},
                }

            best_models = {}
            best_scores = {}

            # Loop through all models for hyperparameter tuning
            for name, model in models.items():
                logging.info(f"Training and tuning model: {name}")

                if tune_hyperparams and param_grids.get(name):
                    grid = GridSearchCV(
                        estimator=model,
                        param_grid=param_grids[name],
                        cv=3,
                        scoring='accuracy' if is_classification else 'r2',
                        n_jobs=-1,
                    )
                    # For classification, ensure there are enough samples per class for cv
                    if is_classification:
                        from collections import Counter
                        class_counts = Counter(y_train_enc)
                        min_count = min(class_counts.values()) if class_counts else 0
                        # determine cv strategy
                        if min_count < 3:
                            # too few samples for stratified 3-fold CV; skip GridSearch and do direct fit
                            logging.info(f"Skipping GridSearchCV for {name} due to small class sizes (min_count={min_count}); performing direct fit")
                            model.fit(X_train, y_train_enc)
                            best_model = model
                            best_score = accuracy_score(y_train_enc, model.predict(X_train))
                        else:
                            # use StratifiedKFold for classification
                            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                            grid = GridSearchCV(
                                estimator=model,
                                param_grid=param_grids[name],
                                cv=skf,
                                scoring='accuracy',
                                n_jobs=-1,
                            )
                            grid.fit(X_train, y_train_enc)
                            best_model = grid.best_estimator_
                            best_score = grid.best_score_
                    else:
                        grid.fit(X_train, y_train)
                        best_model = grid.best_estimator_
                        best_score = grid.best_score_
                    # Log best params only if GridSearchCV was run and found results
                    if hasattr(grid, 'best_params_'):
                        logging.info(f"Best params for {name}: {grid.best_params_}")
                else:
                    if is_classification:
                        model.fit(X_train, y_train_enc)
                        best_model = model
                        best_score = accuracy_score(y_train_enc, model.predict(X_train))
                    else:
                        model.fit(X_train, y_train)
                        best_model = model
                        best_score = r2_score(y_train, model.predict(X_train))

                best_models[name] = best_model
                best_scores[name] = best_score

            # Select the best model
            best_model_name = max(best_scores, key=best_scores.get)
            best_model_score = best_scores[best_model_name]
            best_model = best_models[best_model_name]
            logging.info(f"Best model selected: {best_model_name} with R2 score {best_model_score}")

            # Save the best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Evaluate on test data
            if is_classification:
                y_pred = best_model.predict(X_test)
                final_score = accuracy_score(y_test_enc, y_pred)
                logging.info(f"Final accuracy on test data: {final_score}")
                print(f"Final Accuracy: {final_score}")

                # Save predictions vs actual to CSV
                try:
                    # If we have a label encoder, inverse transform for readability
                    try:
                        y_pred_readable = le.inverse_transform(y_pred)
                        y_test_readable = le.inverse_transform(y_test_enc)
                    except Exception:
                        y_pred_readable = y_pred
                        y_test_readable = y_test_enc

                    preds_df = pd.DataFrame({
                        'predicted': y_pred_readable,
                        'actual': y_test_readable,
                    })
                    preds_csv_path = os.path.join('artifacts', 'predictions.csv')
                    preds_df.to_csv(preds_csv_path, index=False)
                    logging.info(f"Saved predictions to {preds_csv_path}")
                except Exception:
                    logging.info("Failed to save predictions CSV")

                # Save confusion matrix plot
                try:
                    cm = confusion_matrix(y_test_enc, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
                    for (i, j), val in np.ndenumerate(cm):
                        ax.text(j, i, int(val), ha='center', va='center')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    cm_path = os.path.join('artifacts', 'confusion_matrix.png')
                    fig.savefig(cm_path)
                    plt.close(fig)
                    logging.info(f"Saved confusion matrix to {cm_path}")
                except Exception:
                    logging.info("Failed to save confusion matrix plot")

                # Save model bundle (model + label encoder) for future inference
                try:
                    model_bundle = {
                        'model': best_model,
                        'label_encoder': le,
                    }
                    bundle_path = os.path.join('artifacts', 'model_bundle.pkl')
                    save_object(file_path=bundle_path, obj=model_bundle)
                    logging.info(f"Saved model bundle (model + label encoder) to {bundle_path}")
                except Exception:
                    logging.info("Failed to save model bundle with label encoder")

                # Produce classification report and save
                try:
                    report = classification_report(y_test_enc, y_pred, target_names=(le.classes_ if hasattr(le, 'classes_') else None))
                    report_path = os.path.join('artifacts', 'classification_report.txt')
                    with open(report_path, 'w') as f:
                        f.write(report)
                    logging.info(f"Saved classification report to {report_path}")
                except Exception:
                    logging.info("Failed to save classification report")

                # Compute and save ROC curves for multiclass
                try:
                    classes = np.unique(y_test_enc)
                    n_classes = len(classes)
                    # Binarize the output
                    y_test_bin = label_binarize(y_test_enc, classes=classes)
                    # get prediction scores/probabilities
                    try:
                        y_score = best_model.predict_proba(X_test)
                    except Exception:
                        # fall back to decision function
                        y_score = best_model.decision_function(X_test)
                    # if binary, ensure shape
                    if n_classes == 2 and y_score.ndim == 1:
                        y_score = np.vstack([1 - y_score, y_score]).T

                    # Compute ROC curve and AUC for each class
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])

                    # Plot all ROC curves
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
                    for i in range(n_classes):
                        label = f"Class {le.classes_[i]} (AUC = {roc_auc[i]:.2f})" if hasattr(le, 'classes_') else f"Class {i} (AUC = {roc_auc[i]:.2f})"
                        ax.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2, label=label)
                    ax.plot([0, 1], [0, 1], 'k--', lw=2)
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic (ROC)')
                    ax.legend(loc='lower right')
                    roc_path = os.path.join('artifacts', 'roc_curves.png')
                    fig.savefig(roc_path)
                    plt.close(fig)
                    logging.info(f"Saved ROC curves to {roc_path}")
                except Exception:
                    logging.info("Failed to compute or save ROC curves")
            else:
                y_pred = best_model.predict(X_test)
                final_r2 = r2_score(y_test, y_pred)
                logging.info(f"Final R2 score on test data: {final_r2}")
                print(f"Final R2 Score: {final_r2}")

            return final_score if is_classification else final_r2

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)
