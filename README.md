## Machine Learning project for DSF Hackathon

"Generate an ML algorithm (preferably python based) to predict how a DSF project will end, given the current delivery parameters. It should be a predictive model generating a RAG – Red, Amber & Green status (with reason). The teams will be supplied with delivery parameters (as captured in JIRA) in excel format of anonymized projects (training data). Evaluation will be done on a different set of project list with same parameters."

This project automates the end-to-end machine learning workflow from data ingestion to transformation to model training and predictions.

**1. Overview :**
    The goal is to predict project progress or RAG status(Red, Amber and Green) based on the project metadata that is fetched from JIRA.

**2. Pipleline Architecture :**
    Raw CSV → Data Ingestion (src/components/data_ingestion.py) → Data Transformation (src/components/data_transformation.py) → Model Training (src/components/model_trainer.py) → Prediction (with Reasons) (src/pipeline/predict_pipeline.py)
    
**3. Folder Structure :**

📦 dsfmlproject/
 ┣ 📂 notebook/data/
 ┃ ┗ 📄 Project_Progress_Report_Status.csv
 ┣ 📂 src/
 ┃ ┣ 📂 components/
 ┃ ┃ ┣ 📄 data_ingestion.py
 ┃ ┃ ┣ 📄 data_transformation.py
 ┃ ┃ ┗ 📄 model_trainer.py
 ┃ ┣ 📂 pipeline/
 ┃ ┃ ┗ 📄 predict_pipeline.py
 ┃ ┣ 📄 utils.py
 ┃ ┣ 📄 exception.py
 ┃ ┗ 📄 logger.py
 ┣ 📂 artifacts/
 ┃ ┣ 📄 data.csv
 ┃ ┣ 📄 train.csv
 ┃ ┣ 📄 test.csv
 ┃ ┣ 📄 preprocessor.pkl
 ┃ ┗ 📄 model.pkl
 ┗ 📄 README.md

**4. How to Run :**
    # Step 1: Create environment 
    conda create -p venv python=3.8 -y 
    conda activate venv/ 
    
    # Step 2: Install dependencies 
    pip install -r requirements.txt 
    
    # Step 3: Run the pipeline 
    python src/components/data_ingestion.py

**5. Output Artifacts :**

data.csv -> Cleaned raw input data
train.csv, test.csv	-> Split datasets for ML
preprocessor.pkl ->	Saved preprocessing pipeline
model.pkl ->	Best-trained model
prediction_output.csv ->	Predicted RAG statuses with reasons

artifacts/
├── data.csv
├── train.csv
├── test.csv
├── preprocessor.pkl
├── model.pkl
├── model_scores.csv
├── model_comparison.png
└── prediction_output.csv

**6. Tech Stack :**
Python 3.8+
Scikit-learn, Pandas, NumPy
CatBoost, RandomForest
Dill, Logging, Dataclasses

