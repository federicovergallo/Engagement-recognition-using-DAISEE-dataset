

def network():
    model = tf.keras.Sequential()
    model.add(kl.InputLayer(input_shape=(224, 224, 3)))
    # First conv block
    model.add(kl.Conv2D(filters=96, kernel_size=7, padding='same', strides=2))
    model.add(tf.keras.layers.ReLU())
    model.add(kl.MaxPooling2D(pool_size=(3, 3)))
    # Second conv block
    model.add(kl.Conv2D(filters=256, kernel_size=5, padding='same', strides=1))
    model.add(tf.keras.layers.ReLU())
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    # Third-Fourth-Fifth conv block
    for i in range(3):
        model.add(kl.Conv2D(filters=512, kernel_size=3, padding='same', strides=1))
        model.add(tf.keras.layers.ReLU())
    model.add(kl.MaxPooling2D(pool_size=(3, 3)))
    # Flatten
    model.add(kl.Flatten())
    # First FC
    model.add(kl.Dense(4048))
    # Second Fc
    model.add(kl.Dense(4048))
    # Third FC
    model.add(kl.Dense(4))
    # Softmax at the end
    model.add(kl.Softmax())

    return model
