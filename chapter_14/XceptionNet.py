from functools import partial
import tensorflow_datasets as tfds
from CNNUtils import preprocess
import tensorflow as tf
from functional import seq

(test_set, valid_set, train_set), info = tfds.load("tf_flowers",
                                                  split=["train[:10%]","train[10%:25%]", "train[25%:]"], 
                                                  as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples  # 3670
class_names = info.features["label"].names  # ["dandelion", "daisy", ...]
n_classes = info.features["label"].num_classes  # 5

print(f"Dataset size: {dataset_size}, classes: {n_classes}, names: {class_names}")


batch_size = 32
train_set = train_set.shuffle(buffer_size=1000).repeat(1)
train_set = train_set.map(partial(preprocess, randomize=True)).batch(
    batch_size).prefetch(1)
valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)

base_model = tf.keras.applications.Xception(include_top=False,
                                            weights="imagenet", input_shape=(224, 224, 3))
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False
    
optimizer = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay = 0.01)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_set,
                    steps_per_epoch = int( 0.75 * dataset_size / batch_size), 
                    epochs=10, validation_data=valid_set, 
                    validation_steps=int(0.15 * dataset_size / batch_size))

for layer in base_model.layers:
    layer.trainable = True

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay = 0.01)
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_set,
                    steps_per_epoch=int(0.75 * dataset_size / batch_size),
                    epochs=10, validation_data=valid_set,
                    validation_steps=int(0.15 * dataset_size / batch_size))

model.evaluate(test_set, steps=int(0.10 * dataset_size / batch_size))