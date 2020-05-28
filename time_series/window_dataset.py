def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
  dataset = dataset.shuffle(shuffle_buffer)
                    .map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset


series = np.array(x)
time   = np.array(list(range(0, len(x))))

split_time = 8000

time_train = time[:split_time]
x_train    = series[:split_time]

time_valid = timep[split_time:]
x_valid    = series[split_time:]

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model.fit(dataset, epochs=100, verbose=0)

print(f"Layer weigths {l0.get_weights()}")


forecast = []
for time in range(len(series) - windown_size ):
  forecast.append(model.predict(series[time:time+window_size][np.newaxis]))


forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
