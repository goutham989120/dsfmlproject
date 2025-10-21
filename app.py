import os
import sys
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


try:
    import dill
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "dill"])
    import dill


app = FastAPI(title="Model Trainer API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

@app.post("/train/")
async def train(file: UploadFile = File(...), tune_hyperparams: bool = True):
    try:
        # Save uploaded file
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())

        # Read file
        if file.filename.endswith(".xlsx"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

        data = df.values
        train_array, test_array = train_test_split(data, test_size=0.2, random_state=42)

        trainer = ModelTrainer()
        scores_df = trainer.initiate_model_trainer(train_array, test_array, tune_hyperparams)

        return {
            "trained_model": ModelTrainerConfig.trained_model_file_path,
            "scores_csv": ModelTrainerConfig.scores_csv_path,
            "comparison_plot": ModelTrainerConfig.comparison_plot_path,
            "scores": scores_df.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/download/scores")
def download_scores():
    return FileResponse(ModelTrainerConfig.scores_csv_path)


@app.get("/download/plot")
def download_plot():
    return FileResponse(ModelTrainerConfig.comparison_plot_path)


@app.get("/download/model")
def download_model():
    return FileResponse(ModelTrainerConfig.trained_model_file_path)
