import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from prefect import flow, task, get_run_logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient
from supabase import create_client
from dotenv import load_dotenv


load_dotenv()
SUPABASE_KEY = os.getenv("anon_key")
SUPABASE_URL =os.getenv("SUPABASE_URL")


MODEL_NAME    = "titanic-survival-model"
EXPERIMENT    = "titanic-survival"




@task(name="load-data", retries=3, retry_delay_seconds=5)
def load_data():
    logger = get_run_logger()
    logger.info("Loading data from Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table("passengers").select("*").execute()
    all_records = []
    batch_size  = 100
    offset      = 0


    while True:
        response = supabase.table("passengers").select("*").range(offset, offset + batch_size - 1).execute()
        if not response.data:
            break
        all_records.extend(response.data)
        offset += batch_size
        if len(response.data) < batch_size:
            break

    df=pd.DataFrame(all_records)
    df=df.drop(columns=["id","created_at"],errors="ignore")

    assert len(df) > 0, "No data loaded from Supabase"
    assert "survived" in df.columns, "'survived' column is missing in the data"

    logger.info(f"Loaded {len(df)} records from Supabase")
    return df


@task(name="validate-data")
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Validating data...")
    required = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    for col in required:
        assert col in df.columns, f"Column '{col}' is missing in the data"
    missing = df[required].isnull().sum() / len(df) * 100
    for col, pct in missing.items():
        if pct > 50:
            raise ValueError(f"Column '{col}' has {pct:.2f}% missing values, which is above the 50% threshold")
    
    before=len(df)
    df=df.dropna()
    after=len(df)
    logger.info(f"Validation passed. Dropped {before - after} rows with missing values.")
    return df



@task(name="prepare-features")
def prepare_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    logger = get_run_logger()
    logger.info("Preparing features...")
    X = df.drop(columns=["survived"])
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    logger.info("Features prepared successfully")
    return X_train_scaled, X_test_scaled, y_train, y_test,scaler



@task(name="train-model")
def train_model(X_train,y_train, C:float=1.0) :
    logger = get_run_logger()
    logger.info(f"Training model with C={C}...")
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Model trained successfully")
    return model



@task(name="evaluate-model")
def evaluate_model(model, X_test, y_test) ->dict:
    logger = get_run_logger()
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    logger.info(f"Model evaluation completed. Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return {"accuracy": acc, "f1_score": f1}


@task(name="compare-with-production")
def compare_with_production(metrics: dict) -> bool:
    logger = get_run_logger()
    logger.info("Comparing with production model...")
    client = MlflowClient()
    try:
        version      = client.get_model_version_by_alias(MODEL_NAME, "production")
        run          = mlflow.get_run(version.run_id)
        current_acc  = float(run.data.metrics.get("accuracy", 0))
        new_acc      = metrics["accuracy"]
        logger.info(f"Current production model accuracy: {current_acc:.4f}")
        is_better = new_acc > current_acc
        logger.info(f"New model is {'better' if is_better else 'not better'} than production model")
        return is_better
    except  Exception as e:
        logger.warning(f"No production model found or error occurred: {e}")
        return True  # If no production model, consider new model as better

@task(name="register-model")
def register_and_promote(model,scaler,metrics:dict) ->str:
    logger = get_run_logger()
    client= MlflowClient()
    mlflow.set_experiment(EXPERIMENT)

    logger.info("Registering model in MLflow...")

    with mlflow.start_run():
        mlflow.log_params({"C": 1.0, "solver": "lbfgs", "source": "automated-pipeline"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact("scaler.pkl")
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest   = sorted(versions, key=lambda v: int(v.version))[-1]
    client.set_registered_model_alias(MODEL_NAME, "production", latest.version)

    # Save pkl for Docker
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model v{latest.version} promoted to production")
    return latest.version




@task(name="notify")
def notify(version: str, metrics: dict, promoted: bool):
    logger = get_run_logger()
    if promoted:
        logger.info(f"Pipeline complete — v{version} promoted to production")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
    else:
        logger.info("Pipeline complete — current model retained, no update needed")



# ── Main flow ─────────────────────────────────────────────
@flow(name="titanic-retraining-pipeline", log_prints=True)
def retraining_pipeline(C: float = 1.0):
    logger = get_run_logger()
    logger.info("Pipeline started")

    df                                        = load_data()
    df                                        = validate_data(df)
    X_train, X_test, y_train, y_test, scaler = prepare_features(df)
    model                                     = train_model(X_train, y_train, C)
    metrics                                   = evaluate_model(model, X_test, y_test)
    is_better                                 = compare_with_production(metrics)

    if is_better:
        version = register_and_promote(model, scaler, metrics)
        notify(version, metrics, promoted=True)
    else:
        notify("none", metrics, promoted=False)

if __name__ == "__main__":
    retraining_pipeline()