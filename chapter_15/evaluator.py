import tensorflow.keras as keras
from chapter_15.visualizer import plot_learning_curve


def evaluate(net, x_test, y_test, x_valid, y_valid, x_train, y_train, metrics = None):
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    loss = keras.losses.mean_squared_error
    if metrics:
        def last_time_step_mse(y_true, y_pred):
            return keras.metrics.mean_squared_error(y_true[:, -1], y_pred[:, -1])
        net.compile(optimizer=optimizer, loss="mse", metrics=[last_time_step_mse])
    else:
        net.compile(optimizer=optimizer, loss=loss, metrics=['mse'])
    history = net.fit(x_train, y_train, epochs=20, workers=8, validation_data=(x_valid, y_valid), verbose = 2)
    plot_learning_curve(history.history['loss'], history.history['val_loss'])
    return net.evaluate(x_test, y_test)
