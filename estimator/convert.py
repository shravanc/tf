
hparams  = tf.contrib.training.HParams(
    num_epochs = 100,
    batch_size = 500,
    hidden_units=[8, 4], 
    dropout_prob = 0.0)


model_dir = 'trained_models/{}'.format(MODEL_NAME)

run_config = tf.estimator.RunConfig().replace(model_dir=model_dir)
print("Model directory: {}".format(run_config.model_dir))
print("Hyper-parameters: {}".format(hparams))
