"""## 6. Predict Using Serving Function"""

import tensorflow as tf
from tensorflow import data
from datetime import datetime
import multiprocessing
import shutil
import os

print(tf.__version__)


MODEL_NAME = 'sms-class-model-01'
model_dir = 'trained_models/{}'.format(MODEL_NAME)
export_dir = model_dir + "/export/predict/"
print(export_dir)

saved_model_dir = os.listdir("trained_models/sms-class-model-01/export/predict/")[0]
saved_model_dir = "trained_models/sms-class-model-01/export/predict/"+saved_model_dir
print(saved_model_dir)
print("")

predictor_fn = tf.contrib.predictor.from_saved_model(
    export_dir=saved_model_dir,
    signature_def_key="prediction"
)

output = predictor_fn(
    {
        'sms': [
            'girls waiting call chat',
            # 'win 1000 cash free of charge promo hot deal sexy',
            'hot girls sexy tonight call girls waiting call chat',
            # 'hot girls sexy tonight call girls waiting call chat',
            'Ola Francisco Valadares'
        ]

    }
)


print(output["class"])
print(output)
print(output["class"])
print(output["probabilities"])