from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.keras as keras

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target.reshape(-1, 1),
                                                              random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)


def create_huber(threshold = 1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * (tf.abs(error) - 0.5 * threshold)
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn

input_shape = X_train.shape[1:]
net = keras.models.Sequential([
    keras.layers.Dense(30, activation=keras.activations.selu, kernel_initializer=keras.initializers.lecun_normal(),
                       input_shape=input_shape),
    keras.layers.Dense(1)
])

net.compile(loss=create_huber(2.0), optimizer=keras.optimizers.Nadam(), metrics=['mae'])
net.fit(X_train_scaled, y_train, validation_data=(X_valid_scaled, y_valid), epochs=2, workers=0, )


net.save('model_with_custom_loss_threshold.h5')

net = keras.models.load_model('model_with_custom_loss_threshold.h5', custom_objects={'huber_fn': create_huber(2.0)})

net.fit(X_train_scaled, y_train, validation_data=(X_valid_scaled, y_valid), epochs=2, workers=0 )