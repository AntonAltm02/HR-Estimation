import tensorflow as tf


class ModelGenerator:
    """

    """
    def __init__(self, time_steps, input_height, input_width, input_channels):
        """

        :param time_steps: number of time steps
        :param input_height: height of input
        :param input_width: width of input
        :param input_channels: number of channels
        """
        self.input_shape = (time_steps, input_height, input_width, input_channels)
        self.intensity_input_shape = (time_steps, 1)

    def generate_model(self):
        """
        Implements the network's architecture
        :return: model
        """
        # Input layers
        intensity = tf.keras.layers.Input(shape=self.intensity_input_shape)
        input_layer = tf.keras.layers.Input(shape=self.input_shape)

        # TimeDistributed layers
        td_conv1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 37), strides=(4, 4), padding="same")
        )(input_layer)
        td_leaky_relu1 = tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU())(td_conv1)
        td_max_pooling1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(1, 2),
                                                                                       strides=(2, 2)))(td_leaky_relu1)
        td_dropout1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.3))(td_max_pooling1)
        td_conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=5,
                                                                          strides=1, padding="same"))(td_dropout1)
        td_leaky_relu2 = tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU())(td_conv2)
        td_max_pooling2 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(1, 2),
                                                                                       strides=(2, 2)))(td_leaky_relu2)
        td_dropout2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.3))(td_max_pooling2)
        td_flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(td_dropout2)
        td_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=512))(td_flatten)
        td_leaky_relu3 = tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU())(td_dense)
        concatenated = tf.keras.layers.Concatenate(axis=-1)([td_leaky_relu3, intensity])
        lstm1 = tf.keras.layers.LSTM(units=512, return_sequences=True, dropout=0.2)(concatenated)
        lstm2 = tf.keras.layers.LSTM(units=222, return_sequences=True, dropout=0.3)(lstm1)
        last_timestamp = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(lstm2)
        dense = tf.keras.layers.Dense(units=222)(last_timestamp)
        softmax = tf.keras.layers.Softmax(axis=-1)(dense)

        # Create the model
        model = tf.keras.models.Model(inputs=[input_layer, intensity], outputs=softmax)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            # loss=custom_loss,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        return model
