import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import math
import multiprocessing
from datetime import datetime
from tensorflow.python.feature_column import feature_column
print(tf.__version__)

tf.compat.v1.disable_eager_execution()

MODEL_NAME = 'reg-model-01'

TRAIN_DATA_FILE = '/home/shravan/tf/datasets/estimator_data/train.csv'
VALID_DATA_FILE = '/home/shravan/tf/datasets/estimator_data/val.csv'
TEST_DATA_FILE  = '/home/shravan/tf/datasets/estimator_data/test.csv'

RESUME_TRAINING = False
PROCESS_FEATURES = True
MULTI_THREADING = False


HEADER = ['location_number','year','month','date','minutes','noise_level']
HEADER_DEFAULTS = [[1], [2018], [1], [1], [1], [0.0]]

NUMERIC_FEATURE_NAMES = ['location_number','year','month','date','minutes']  


FEATURE_NAMES = NUMERIC_FEATURE_NAMES 

TARGET_NAME = 'noise_level'

UNUSED_FEATURE_NAMES = list(set(HEADER) - set(FEATURE_NAMES) - {TARGET_NAME})

print("Header: {}".format(HEADER))
print("Numeric Features: {}".format(NUMERIC_FEATURE_NAMES))
print("Target: {}".format(TARGET_NAME))
print("Unused Features: {}".format(UNUSED_FEATURE_NAMES))



#####


def generate_pandas_input_fn(file_name, mode=tf.estimator.ModeKeys.EVAL,
                             skip_header_lines=0,
                             num_epochs=1,
                             batch_size=100):

    df_dataset = pd.read_csv(file_name, names=HEADER, skiprows=skip_header_lines)
    print(df_dataset.head())
   
    x = df_dataset[FEATURE_NAMES].copy()
    y = df_dataset[TARGET_NAME]
  
    print("---x---")
    print(x.head())
        
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    
    num_threads=1
    
    if MULTI_THREADING:
        num_threads=multiprocessing.cpu_count()
        num_epochs = int(num_epochs/num_threads) if mode == tf.estimator.ModeKeys.TRAIN else num_epochs
    
    pandas_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        batch_size=batch_size,
        num_epochs= num_epochs,
        shuffle=shuffle,
        x=x,
        y=y,
        target_column=TARGET_NAME
    )
    
    print("")
    print("* data input_fn:")
    print("================")
    print("Input file: {}".format(file_name))
    print("Dataset size: {}".format(len(df_dataset)))
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Thread Count: {}".format(num_threads))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")
    
    return pandas_input_fn


features, target = generate_pandas_input_fn(file_name=TRAIN_DATA_FILE)()
print("Feature read from DataFrame: {}".format(list(features.keys())))
print("Target read from DataFrame: {}".format(target))


####


def get_feature_columns():
    
    
    all_numeric_feature_names = NUMERIC_FEATURE_NAMES
    
    numeric_columns = {feature_name: tf.feature_column.numeric_column(feature_name)
                       for feature_name in all_numeric_feature_names}

    feature_columns = {}

    if numeric_columns is not None:
        feature_columns.update(numeric_columns)

    return feature_columns

feature_columns = get_feature_columns()
print("Feature Columns: {}".format(feature_columns))




def create_estimator(run_config):
    
    feature_columns = list(get_feature_columns().values())
    dense_columns = feature_columns
    estimator_feature_columns = dense_columns 
    print("****", dense_columns) 
    
    estimator = tf.estimator.DNNRegressor(
        feature_columns=estimator_feature_columns,
        hidden_units= [8,4],
        optimizer= tf.keras.optimizers.Adam(), # tf.compat.v1.train.AdamOptimizer(),
        activation_fn= tf.nn.elu,
        dropout=0.0,
        config= run_config
    )
    
    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")
    
    return estimator


from tensorboard.plugins.hparams import api as hp

"""
hparams  = hp.HParam(
    'num_epochs' = 100,
    batch_size = 500,
    hidden_units=[8, 4], 
    dropout_prob = 0.0)
"""

model_dir = 'trained_models/{}'.format(MODEL_NAME)

run_config = tf.estimator.RunConfig().replace(model_dir=model_dir)
print("Model directory: {}".format(run_config.model_dir))
#print("Hyper-parameters: {}".format(hparams))


estimator = create_estimator(run_config)


train_input_fn = generate_pandas_input_fn(file_name= TRAIN_DATA_FILE, 
                                      mode=tf.estimator.ModeKeys.TRAIN,
                                      num_epochs=10,
                                      batch_size=512) 

if not RESUME_TRAINING:
    shutil.rmtree(model_dir, ignore_errors=True)
    
#tf.compat.v1.logging.set_verbosity(tf.logging.INFO)

time_start = datetime.utcnow() 
print("Estimator training started at {}".format(time_start.strftime("%H:%M:%S")))
print(".......................................")

estimator.train(input_fn = train_input_fn)

time_end = datetime.utcnow() 
print(".......................................")
print("Estimator training finished at {}".format(time_end.strftime("%H:%M:%S")))
print("")
time_elapsed = time_end - time_start
print("Estimator training elapsed time: {} seconds".format(time_elapsed.total_seconds()))
