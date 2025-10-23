import os
import sys
from dataclasses import dataclass
import warnings
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
)
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.exceptions import UndefinedMetricWarning
# Globally ignore sklearn's UndefinedMetricWarning (common when no positive samples exist for a class)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
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
    roc_plot_path = os.path.join("artifacts", "roc_curve.png")
    confusion_matrix_path = os.path.join("artifacts", "confusion_matrix.png")
    classification_report_csv = os.path.join("artifacts", "classification_report.csv")
    classification_metrics_csv = os.path.join("artifacts", "classification_metrics.csv")
    model_bundle_path = os.path.join("artifacts", "model_bundle.pkl")
    confusion_matrix_csv = os.path.join("artifacts", "confusion_matrix.csv")
    summary_slide_path = os.path.join("artifacts", "summary_slide.png")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def prepare_train_test(self, df: pd.DataFrame, target_col='RAG', test_size=0.2, random_state=42):
        """
        Splits dataframe into train/test and preserves actual RAG reason.
        Guarantees that every class appears at least once in test set.
        """
        try:
            if target_col not in df.columns:
                raise CustomException(f"Target column '{target_col}' not found in dataframe", sys)

            df['actual_RAG_reason'] = df.get('RAG Reason + Observations', '')

            stratify_col = df[target_col] if df[target_col].nunique() > 1 else None

            if stratify_col is not None:
                class_counts = df[target_col].value_counts()
                if class_counts.min() < 2:
                    logging.warning("Some classes have only 1 sample, stratified split not possible. Using non-stratified split.")
                    stratify_col = None

            train_df, test_df = train_test_split(
                df, test_size=test_size, stratify=stratify_col, random_state=random_state
            )

            # Ensure all classes appear in test set
            missing_classes = set(df[target_col].unique()) - set(test_df[target_col].unique())
            if missing_classes:
                logging.info(f"Adjusting test set to include missing classes: {missing_classes}")
                for cls in missing_classes:
                    row_to_move = train_df[train_df[target_col] == cls].iloc[0]
                    test_df = pd.concat([test_df, row_to_move.to_frame().T])
                    train_df = train_df.drop(row_to_move.name)

            os.makedirs(os.path.dirname(self.model_trainer_config.train_csv), exist_ok=True)
            train_df.to_csv(self.model_trainer_config.train_csv, index=False)
            test_df.to_csv(self.model_trainer_config.test_csv, index=False)
            logging.info(f"Train CSV saved to {self.model_trainer_config.train_csv} ({len(train_df)} rows)")
            logging.info(f"Test CSV saved to {self.model_trainer_config.test_csv} ({len(test_df)} rows)")

            train_array = train_df.drop(columns=['actual_RAG_reason']).values
            test_array = test_df.drop(columns=['actual_RAG_reason']).values
            return train_array, test_array

        except Exception as e:
            logging.error("Error during train/test preparation")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array, tune_hyperparams=True, console_summary: bool = False):
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
            best_score = test_scores[best_model_name]
            logging.info(f" Best model: {best_model_name} (test score: {best_score:.4f})")
            # Also print to console (so VS Code/terminal shows the result)
            try:
                if isinstance(best_score, float):
                    score_display = f"{best_score:.4f}"
                else:
                    score_display = str(best_score)
                print(f"BEST MODEL => {best_model_name} | Score: {score_display}")
            except Exception:
                # Fallback: still ensure logging recorded it
                pass

            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            # Also save a bundle with model + label encoder (if classification)
            try:
                bundle = {'model': best_model}
                if is_classification and 'le' in locals():
                    bundle['label_encoder'] = le
                os.makedirs(os.path.dirname(self.model_trainer_config.model_bundle_path), exist_ok=True)
                save_object(self.model_trainer_config.model_bundle_path, bundle)
                logging.info(f"Saved model bundle to {self.model_trainer_config.model_bundle_path}")
            except Exception as e:
                logging.warning(f"Failed to save model bundle: {e}")

            # ==================== SAVE MODEL SCORES ====================
            os.makedirs(os.path.dirname(self.model_trainer_config.model_scores_csv), exist_ok=True)
            scores_df = pd.DataFrame(list(test_scores.items()), columns=["Model", "Score"])
            scores_df.to_csv(self.model_trainer_config.model_scores_csv, index=False)
            logging.info(f"Saved model scores to {self.model_trainer_config.model_scores_csv}")

            # ==================== CONFUSION MATRIX & ROC ====================
            try:
                if is_classification:
                    best = best_model
                    y_pred = best.predict(X_test)

                    # Suppress UndefinedMetricWarning
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

                        # Safe LabelEncoder for confusion matrix
                        try:
                            le_cm = LabelEncoder()
                            le_cm.fit(np.concatenate([y_test_enc, y_pred.astype(str)]))
                            y_test_enc_safe = le_cm.transform(y_test_enc)
                            try:
                                y_pred_safe = le_cm.transform(y_pred.astype(str))
                            except Exception:
                                y_pred_safe = y_pred
                            labels = le_cm.classes_
                        except Exception:
                            y_test_enc_safe = y_test_enc
                            y_pred_safe = y_pred
                            labels = np.unique(np.concatenate([y_test_enc_safe, y_pred_safe]))

                        # Confusion matrix
                        cm = confusion_matrix(y_test_enc_safe, y_pred_safe, labels=range(len(labels)))
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
                        fig_cm, ax_cm = plt.subplots(figsize=(8,6))
                        disp.plot(ax=ax_cm, cmap='Blues', colorbar=False)
                        plt.title('Confusion Matrix')
                        # Label axes for clarity
                        ax_cm.set_xlabel('Predicted RAG')
                        ax_cm.set_ylabel('Actual RAG')
                        # Rotate tick labels for readability
                        if labels is not None and len(labels) > 0:
                            ax_cm.set_xticklabels(labels, rotation=45, ha='right')
                            ax_cm.set_yticklabels(labels, rotation=0)
                        # Annotate mapping: encoded integer -> RAG label
                        try:
                            mapping = {}
                            if 'le' in locals():
                                # use the original label encoder if available
                                mapping = {i: str(lbl) for i, lbl in enumerate(list(le.classes_))}
                            else:
                                mapping = {i: str(lbl) for i, lbl in enumerate(labels)}
                            mapping_lines = [f"{k}: {v}" for k, v in mapping.items()]
                            mapping_str = "\n".join(mapping_lines)
                            # place the mapping in a small textbox on the figure
                            fig_cm.text(0.99, 0.01, mapping_str, ha='right', va='bottom', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), transform=fig_cm.transFigure)
                        except Exception:
                            pass
                        plt.tight_layout()
                        os.makedirs(os.path.dirname(self.model_trainer_config.confusion_matrix_path), exist_ok=True)
                        plt.savefig(self.model_trainer_config.confusion_matrix_path)
                        plt.close(fig_cm)
                        logging.info(f"Saved confusion matrix to {self.model_trainer_config.confusion_matrix_path}")

                        # Save labeled confusion matrix to CSV (rows=actual, cols=predicted)
                        try:
                            # Prefer original label encoder to get textual labels
                            n = cm.shape[0]
                            try:
                                if is_classification and 'le' in locals():
                                    text_labels = list(le.inverse_transform(list(range(n))))
                                else:
                                    text_labels = [str(x) for x in labels]
                            except Exception:
                                text_labels = [str(x) for x in labels]
                            cm_df = pd.DataFrame(cm, index=text_labels, columns=text_labels)
                            os.makedirs(os.path.dirname(self.model_trainer_config.confusion_matrix_csv), exist_ok=True)
                            cm_df.to_csv(self.model_trainer_config.confusion_matrix_csv)
                            logging.info(f"Saved confusion matrix CSV to {self.model_trainer_config.confusion_matrix_csv}")
                        except Exception as e:
                            logging.warning(f"Failed to save confusion matrix CSV: {e}")

                        # Classification metrics: precision, recall, f1 (per-class and averages)
                        try:
                            # compute per-class precision/recall/f1
                            labels_for_metrics = list(range(len(labels))) if labels is not None else None
                            p, r, f1, support = precision_recall_fscore_support(y_test_enc_safe, y_pred_safe, labels=labels_for_metrics, zero_division=0)
                            report = classification_report(y_test_enc_safe, y_pred_safe, labels=labels_for_metrics, target_names=[str(x) for x in labels], zero_division=0)
                            # Save report text
                            os.makedirs(os.path.dirname(self.model_trainer_config.classification_report_csv), exist_ok=True)
                            with open(self.model_trainer_config.classification_report_csv, 'w', encoding='utf-8') as fh:
                                fh.write(report)
                            # Save numeric metrics to CSV
                            metrics_df = pd.DataFrame({
                                'label': [str(x) for x in labels],
                                'precision': p,
                                'recall': r,
                                'f1': f1,
                                'support': support,
                            })
                            metrics_df.loc['macro_avg'] = [ 'macro_avg', metrics_df['precision'].mean(), metrics_df['recall'].mean(), metrics_df['f1'].mean(), metrics_df['support'].sum() ]
                            # Suppress pandas FutureWarning about concat behavior when assigning rows with different dtypes
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', category=FutureWarning)
                                metrics_df.loc['micro_avg'] = [ 'micro_avg', None, None, None, metrics_df['support'].sum() ]
                            metrics_df.to_csv(self.model_trainer_config.classification_metrics_csv, index=False)
                            logging.info(f"Saved classification report to {self.model_trainer_config.classification_report_csv}")
                            logging.info(f"Saved classification metrics to {self.model_trainer_config.classification_metrics_csv}")
                        except Exception as e:
                            logging.warning(f"Failed to compute/save classification metrics: {e}")

                        # ROC curve (binary/multiclass)
                        try:
                            if len(np.unique(y_test_enc_safe)) > 1:
                                y_score = None
                                if hasattr(best, 'predict_proba'):
                                    y_score = best.predict_proba(X_test)
                                elif hasattr(best, 'decision_function'):
                                    y_score = best.decision_function(X_test)

                                if y_score is not None:
                                    n_classes = y_score.shape[1] if len(y_score.shape) > 1 else 1
                                    if n_classes == 1:
                                        fpr, tpr, _ = roc_curve(y_test_enc_safe, y_score[:, 1] if y_score.ndim>1 else y_score)
                                        roc_auc = auc(fpr, tpr)
                                        plt.figure(figsize=(8,6))
                                        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                                        plt.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
                                        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
                                        plt.title('Receiver Operating Characteristic')
                                        plt.legend(loc='lower right')
                                        plt.tight_layout()
                                        os.makedirs(os.path.dirname(self.model_trainer_config.roc_plot_path), exist_ok=True)
                                        plt.savefig(self.model_trainer_config.roc_plot_path)
                                        plt.close()
                                        logging.info(f"Saved ROC curve to {self.model_trainer_config.roc_plot_path}")
                                    else:
                                        y_test_b = label_binarize(y_test_enc_safe, classes=range(n_classes))
                                        fpr = {}; tpr = {}; roc_auc = {}
                                        for i in range(n_classes):
                                            fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score[:, i])
                                            roc_auc[i] = auc(fpr[i], tpr[i])
                                        fpr['micro'], tpr['micro'], _ = roc_curve(y_test_b.ravel(), y_score.ravel())
                                        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
                                        plt.figure(figsize=(8,6))
                                        plt.plot(fpr['micro'], tpr['micro'], label=f'micro-average ROC (area = {roc_auc["micro"]:.2f})', color='deeppink', linestyle=':', linewidth=4)
                                        for i in range(n_classes):
                                            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')
                                        plt.plot([0,1],[0,1],'k--',lw=1)
                                        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
                                        plt.title('Multiclass ROC')
                                        plt.legend(loc='lower right')
                                        plt.tight_layout()
                                        os.makedirs(os.path.dirname(self.model_trainer_config.roc_plot_path), exist_ok=True)
                                        plt.savefig(self.model_trainer_config.roc_plot_path)
                                        plt.close()
                                        logging.info(f"Saved ROC curve to {self.model_trainer_config.roc_plot_path}")
                        except Exception as e:
                            logging.warning(f"Failed to generate ROC curve: {e}")
            except Exception as e:
                logging.warning(f"Failed to generate confusion matrix / ROC: {e}")

            # ==================== MODEL COMPARISON PLOT ====================
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

            # ==================== SUMMARY SLIDE (single image) ====================
            try:
                # Prepare subplots: 2x2
                fig, axes = plt.subplots(2,2, figsize=(14,10))

                # Top-left: model comparison
                ax0 = axes[0,0]
                models_list = list(sorted_scores.keys())
                scores_list = list(sorted_scores.values())
                ax0.bar(models_list, scores_list, color='skyblue')
                ax0.set_title('Model Comparison')
                ax0.set_ylabel(ylabel)
                ax0.tick_params(axis='x', rotation=45)

                # Top-right: ROC (reuse existing roc if available)
                ax1 = axes[0,1]
                try:
                    if is_classification and 'y_score' in locals() and y_score is not None:
                        if (hasattr(y_score, 'shape') and (len(y_score.shape) > 1 and y_score.shape[1] > 1)):
                            # multiclass: plot micro + per-class
                            y_test_b = label_binarize(y_test_enc_safe, classes=range(y_score.shape[1]))
                            fpr = dict(); tpr = dict(); roc_auc = dict()
                            for i in range(y_score.shape[1]):
                                fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score[:, i])
                                roc_auc[i] = auc(fpr[i], tpr[i])
                            fpr['micro'], tpr['micro'], _ = roc_curve(y_test_b.ravel(), y_score.ravel())
                            roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
                            ax1.plot(fpr['micro'], tpr['micro'], label=f'micro (AUC={roc_auc["micro"]:.2f})', color='deeppink', linestyle=':')
                            for i in range(y_score.shape[1]):
                                ax1.plot(fpr[i], tpr[i], lw=1.5, label=f'Class {i} (AUC={roc_auc[i]:.2f})')
                        else:
                            fpr, tpr, _ = roc_curve(y_test_enc_safe, y_score[:,1] if y_score.ndim>1 else y_score)
                            roc_auc = auc(fpr, tpr)
                            ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc:.2f}')
                        ax1.plot([0,1],[0,1],'k--', lw=1)
                        ax1.set_title('ROC Curve')
                        ax1.set_xlabel('False Positive Rate')
                        ax1.set_ylabel('True Positive Rate')
                        ax1.legend(loc='lower right', fontsize='small')
                    else:
                        ax1.text(0.5,0.5,'ROC not available', ha='center', va='center')
                        ax1.set_title('ROC Curve')
                except Exception:
                    ax1.text(0.5,0.5,'ROC not available', ha='center', va='center')
                    ax1.set_title('ROC Curve')

                # Bottom-left: confusion matrix
                ax2 = axes[1,0]
                try:
                    disp.plot(ax=ax2, cmap='Blues', colorbar=False)
                    ax2.set_title('Confusion Matrix')
                    ax2.set_xlabel('Predicted RAG')
                    ax2.set_ylabel('Actual RAG')
                    for tick in ax2.get_xticklabels():
                        tick.set_rotation(45)
                except Exception:
                    ax2.text(0.5,0.5,'Confusion matrix not available', ha='center', va='center')
                    ax2.set_title('Confusion Matrix')

                # Bottom-right: note pointing to metrics CSV
                ax3 = axes[1,1]
                ax3.axis('off')
                try:
                    note = f"Detailed metrics saved to:\n{self.model_trainer_config.classification_metrics_csv}"
                    ax3.text(0.5,0.5, note, ha='center', va='center', fontsize=10)
                    ax3.set_title('Classification Metrics (see CSV)')
                except Exception:
                    ax3.text(0.5,0.5,'Detailed metrics saved to classification_metrics.csv', ha='center', va='center')
                    ax3.set_title('Classification Metrics (see CSV)')

                plt.tight_layout()
                os.makedirs(os.path.dirname(self.model_trainer_config.summary_slide_path), exist_ok=True)
                fig.savefig(self.model_trainer_config.summary_slide_path, dpi=150)
                plt.close(fig)
                logging.info(f"Saved summary slide to {self.model_trainer_config.summary_slide_path}")
            except Exception as e:
                logging.warning(f"Failed to create summary slide: {e}")

            # If console_summary requested, print only the model comparison, confusion matrix and ROC AUC
            if console_summary:
                try:
                    print("\n=== Model comparison (test scores) ===")
                    sorted_scores = dict(sorted(test_scores.items(), key=lambda item: item[1], reverse=True))
                    for name, sc in sorted_scores.items():
                        print(f"{name}: {sc:.4f}")

                    print("\n=== Confusion Matrix ===")
                    try:
                        best = best_model
                        y_pred = best.predict(X_test)
                        try:
                            labels_list = list(le.classes_)
                            try:
                                y_pred_enc = le.transform(y_pred.astype(str))
                            except Exception:
                                y_pred_enc = y_pred
                            cm_local = confusion_matrix(y_test_enc, y_pred_enc)
                            print("Label mapping:")
                            for i, lbl in enumerate(labels_list):
                                print(f"{i}: {lbl}")
                            print(cm_local)
                        except Exception:
                            cm_local = confusion_matrix(y_test, y_pred)
                            print(cm_local)
                    except Exception as e:
                        print(f"Could not compute confusion matrix: {e}")

                    print("\n=== ROC AUC ===")
                    try:
                        y_score_local = None
                        if hasattr(best_model, 'predict_proba'):
                            y_score_local = best_model.predict_proba(X_test)
                        elif hasattr(best_model, 'decision_function'):
                            y_score_local = best_model.decision_function(X_test)
                        if y_score_local is not None:
                            n_classes = y_score_local.shape[1] if len(getattr(y_score_local, 'shape', [])) > 1 else 1
                            if n_classes == 1:
                                fpr, tpr, _ = roc_curve(y_test_enc if 'y_test_enc' in locals() else y_test, y_score_local[:,1] if getattr(y_score_local, 'ndim', 1)>1 else y_score_local)
                                roc_auc_local = auc(fpr, tpr)
                                print(f"AUC: {roc_auc_local:.4f}")
                            else:
                                y_test_b_local = label_binarize(y_test_enc if 'y_test_enc' in locals() else y_test, classes=range(n_classes))
                                for i in range(n_classes):
                                    fpr, tpr, _ = roc_curve(y_test_b_local[:, i], y_score_local[:, i])
                                    print(f"Class {i} AUC: {auc(fpr, tpr):.4f}")
                        else:
                            print("ROC not available (model provides no probability/decision scores)")
                    except Exception as e:
                        print(f"Could not compute ROC AUC: {e}")
                except Exception:
                    pass

            return best_model_name, test_scores[best_model_name]

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)
