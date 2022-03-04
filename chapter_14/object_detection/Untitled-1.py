# %%
import os, random
import zipfile
import pandas as pd

import numpy as np
from skimage.io import imread  # reading images
from skimage.transform import resize  # resizing images
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import tensorflow_datasets as tfds
from coders import LabelEncoder, DecodePredictions
from retinamodels import RetinaNet
from losses import RetinaNetLoss
from RetinaNetUtils import get_backbone, preprocess_data, resize_and_pad_image

# %%
url = "http://agristats.eu/images.zip"
filename = os.path.join(os.getcwd(), "images.zip")
keras.utils.get_file(filename, url)

with zipfile.ZipFile("images.zip", "r") as z_fp:
    z_fp.extractall("./")
    


# %%


# %%

# %%



# %%
model_dir = "retinanet/"
label_encoder = LabelEncoder()

num_classes = 80
batch_size = 2

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

# %%
resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

# %%
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    )
]

import json
import io
url = "http://agristats.eu/images.json"
# filename = os.path.join(os.getcwd(), "images.json")
# keras.utils.get_file(filename, url)
with open('images.json') as json_file:
    train_data = json.load(json_file)





# %%

directory = os.fsencode("images")
for file in os.listdir(directory):
  filename = os.fsdecode(file)
  for item in train_data:
    if filename == item["image/filename"]: 
      image = load_img(os.path.join("./images/", filename))
      image = img_to_array(image)
      item["image"] = image
      item["label"] = random.randint(0,1)
myimages = pd.DataFrame.from_dict(train_data).to_dict("list")
myimages = tf.data.Dataset.from_tensor_slices(myimages)

crop_size = 300
upscale_factor = 3
input_size = crop_size // upscale_factor
batch_size = 8
#root_dir = '/content/images'

# %%


# %%
train_dataset = myimages.take(180)
val_dataset = myimages.skip(180) 

# %%

autotune = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
print('1Number of train batches are: ', tf.data.experimental.cardinality(train_dataset))
train_dataset = train_dataset.shuffle(8 * batch_size)
print('2Number of train batches are: ', tf.data.experimental.cardinality(train_dataset))
train_dataset = train_dataset.padded_batch(
    batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
print('3Number of train batches are: ', tf.data.experimental.cardinality(train_dataset))
train_dataset = train_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
print('4Number of train batches are: ', tf.data.experimental.cardinality(train_dataset))
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
print('5Number of train batches are: ', tf.data.experimental.cardinality(train_dataset))
train_dataset = train_dataset.prefetch(autotune)
print('6Number of train batches are: ', tf.data.experimental.cardinality(train_dataset))

print('1Number of validation batches are: ', tf.data.experimental.cardinality(val_dataset))
val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
print('2Number of validation batches are: ', tf.data.experimental.cardinality(val_dataset))
val_dataset = val_dataset.padded_batch(
    batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
print('3Number of validation batches are: ', tf.data.experimental.cardinality(val_dataset))
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
print('4Number of validation batches are: ', tf.data.experimental.cardinality(val_dataset))
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
print('5Number of validation batches are: ', tf.data.experimental.cardinality(val_dataset))
val_dataset = val_dataset.prefetch(autotune)
print('Final')
print('Number of train batches are: are: ', tf.data.experimental.cardinality(val_dataset))
print('Number of validation batches are: ', tf.data.experimental.cardinality(val_dataset))

# %%
# Uncomment the following lines, when training on full dataset
# train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
# val_steps_per_epoch = \
#     dataset_info.splits["validation"].num_examples // batch_size

# train_steps = 4 * 100000
# epochs = train_steps // train_steps_per_epoch

epochs = 1

# Running 100 training and 50 validation steps,
# remove `.take` when training on the full dataset
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)

# %%
# Change this to `model_dir` when not using the downloaded weights
weights_dir = model_dir

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

# %%
image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

# %%
def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
int2str = dataset_info.features["label"].int2str

image = load_img('test1.jpg')
image = img_to_array(image)

input_image, ratio = prepare_image(image)
detections = inference_model.predict(input_image)
num_detections = detections.valid_detections[0]
class_names = [
    int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
]
visualize_detections(
    image,
    detections.nmsed_boxes[0][:num_detections] / ratio,
    class_names,
    detections.nmsed_scores[0][:num_detections],
)

#for sample in val_dataset.take(4):
#    print(sample["image"])
#    image = tf.cast(sample["image"], dtype=tf.float32)
#    input_image, ratio = prepare_image(image)
#    detections = inference_model.predict(input_image)
#    num_detections = detections.valid_detections[0]
#    class_names = [
#        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
#    ]
#    visualize_detections(
#        image,
#        detections.nmsed_boxes[0][:num_detections] / ratio,
#        class_names,
#        detections.nmsed_scores[0][:num_detections],
#    )



