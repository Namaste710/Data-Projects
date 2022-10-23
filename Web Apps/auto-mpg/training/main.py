import training as training
import joblib
FILE_PATH = "data/auto-mpg.data"
MODEL_PATH = "artifacts/model.joblib"

if __name__ == "__main__":
    df = training.load_data(FILE_PATH)
    model, preproc_pl = training.model_training(df)
    joblib.dump((model, preproc_pl), MODEL_PATH)
    
    