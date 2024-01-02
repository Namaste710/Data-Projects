import io
import pickle

import joblib
import training as training
from minio import Minio

FILE_PATH = "data/auto-mpg.data"
MODEL_PATH = "artifacts/model.joblib"

if __name__ == "__main__":
    df = training.load_data(FILE_PATH)
    model, preproc_pl = training.model_training(df)
    joblib.dump((model, preproc_pl), MODEL_PATH)

    minio_client = Minio(endpoint="minio-storage:9000")

    bytes_model_pipeline = pickle.dumps((model, preproc_pl))

    minio_client.put_object(
        bucket_name="test-bucket",
        object_name="test",
        data=io.BytesIO(bytes_model_pipeline),
        length=len(bytes_model_pipeline),
    )

    # pickle.loads(client.get_object(bucket_name=bucket_name,
    #                            object_name=path_file).read())
