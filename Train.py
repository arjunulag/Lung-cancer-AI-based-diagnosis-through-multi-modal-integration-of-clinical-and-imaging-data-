from google.colab import drive
drive.mount("/content/drive")

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def undersample_majority(df, class_feature, seed):
  df_majority = df[df[class_feature] == 0]
  df_minority = df[df[class_feature] == 1]
  num_positive_cases = df_minority.shape[0]
  df_majority_downsampled = resample(df_majority, replace=False,  n_samples=num_positive_cases, random_state=seed)
  df_balanced = pd.concat([df_minority, df_majority_downsampled])
  df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
  return df_balanced

def resize_and_pad_image(image, target_size):
  initial_shape = tf.cast(tf.shape(image)[:2], tf.float32)
  ratio = tf.reduce_min([target_size[0] / initial_shape[0], target_size[0] / initial_shape[1]])
  new_shape = tf.cast(initial_shape * ratio, tf.int32)
  image = tf.image.resize(image, new_shape)
  delta_height = target_size[0] - new_shape[0]
  delta_width = target_size[1] - new_shape[1]
  pad_top = delta_height // 2
  pad_bottom = delta_height - pad_top
  pad_left = delta_width // 2
  pad_right = delta_width - pad_left
  image = tf.pad(image, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0)
  return image

def load_image_only(image_path, label, target_size):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=1)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = resize_and_pad_image(image, target_size)
  image = tf.image.grayscale_to_rgb(image)
  return image, label

def load_image_with_features(image_path, features, label, target_size):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=1)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = resize_and_pad_image(image, target_size)
  image = tf.image.grayscale_to_rgb(image)
  features = tf.cast(features, tf.float32)
  return (image, features), label

def load_features_only(features, label):
  features = tf.cast(features, tf.float32)
  return features, label

def create_dataset(df, features, seed, include_images, include_features, target_size, batch_size, training):
  dataset = tf.data.Dataset.from_tensor_slices((df['path'], df[features], df['lung_cancer']))
  if training:
    dataset = dataset.shuffle(buffer_size=1000, seed=seed)
  if include_images and include_features:
    dataset = dataset.map(lambda x, y, z: load_image_with_features(x, y, z, target_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif include_images and not include_features:
    dataset = dataset.map(lambda x, _, z: load_image_only(x, z, target_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif not include_images and include_features:
    dataset = dataset.map(lambda _, y, z: load_features_only(y, z), num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size)
  dataset = dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
  return dataset

def create_feature_model(feature_size):
  input_layer = tf.keras.layers.Input(feature_size)
  x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2())(input_layer)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2())(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2())(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
  model = tf.keras.Model(input_layer, x)
  return model

def create_image_model(image_size):
  base_model = tf.keras.applications.ResNet152V2(input_shape=(image_size[0], image_size[1], 3), input_tensor=None,
                                                 include_top=False, weights='imagenet', pooling='avg')
  base_model.trainable = False

  data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomContrast(0.1),
      tf.keras.layers.RandomRotation(0.1),
      tf.keras.layers.RandomBrightness(0.1),
      tf.keras.layers.RandomZoom(0.1)
  ])

  input_layer = tf.keras.layers.Input(shape=image_size)
  x = data_augmentation(input_layer)
  x = tf.keras.layers.experimental.preprocessing.Rescaling(255)(x)
  x = tf.keras.applications.resnet_v2.preprocess_input(x)
  x = base_model(x, training=False)
  x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2())(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2())(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
  model = tf.keras.Model(inputs=input_layer, outputs=x)
  return model

def create_multimodal_model(image_size, feature_size):
  base_model = tf.keras.applications.ResNet152V2(input_shape=(image_size[0], image_size[1], 3), input_tensor=None,
                                                 include_top=False, weights='imagenet', pooling='avg')
  base_model.trainable = False
  data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomContrast(0.1),
      tf.keras.layers.RandomRotation(0.1),
      tf.keras.layers.RandomBrightness(0.1),
      tf.keras.layers.RandomZoom(0.1)
  ])
  # Inputs
  input_image = tf.keras.layers.Input(shape=image_size, name='image_input')
  input_feature = tf.keras.layers.Input(shape=feature_size, name='feature_input')
  # Image stream
  x = data_augmentation(input_image)
  x = tf.keras.layers.experimental.preprocessing.Rescaling(255)(x)
  x = tf.keras.applications.resnet_v2.preprocess_input(x)
  x = base_model(x, training=False)
  x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2())(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2())(x)
  # Feature stream
  y = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2())(input_feature)
  y = tf.keras.layers.Dropout(0.2)(y)
  y = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2())(y)
  y = tf.keras.layers.Dropout(0.2)(y)
  y = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2())(y)
  # Late fusion
  combined = tf.keras.layers.concatenate([x, y])
  z = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2())(combined)
  z = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2())(z)
  z = tf.keras.layers.Dropout(0.2)(z)
  z = tf.keras.layers.Dense(1, activation='sigmoid')(z)
  model = tf.keras.Model(inputs=[input_image, input_feature], outputs=z)
  return model

def plot_training_curves(history):
  plt.figure(figsize=(15, 5))
  plt.subplot(1, 2, 1)
  plt.plot(history.history['loss'], label='training loss')
  plt.plot(history.history['val_loss'], label='validation loss')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend()
  plt.subplot(1, 2, 2)
  plt.plot(history.history['accuracy'], label='training accuracy')
  plt.plot(history.history['val_accuracy'], label='validation accuracy')
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  plt.legend()
  plt.show()

def get_y_true_and_y_pred(model, dataset):
  y_true_list = []
  y_pred_list = []
  for batch in dataset:
    x, y_true = batch
    y_pred = model.predict(x, verbose=0)
    y_true_list.append(y_true.numpy())
    y_pred_list.append(y_pred)
  y_true = np.concatenate(y_true_list)
  y_pred = np.concatenate(y_pred_list).squeeze()
  y_pred = (y_pred >= 0.5).astype('int64')
  return y_true, y_pred

def evaluate_model(model, dataset):
  Y_test, Y_pred = get_y_true_and_y_pred(model, dataset)
  classes = np.unique(Y_test)
  accuracy = np.mean(Y_test == Y_pred)*100
  cm = confusion_matrix(Y_test, Y_pred, labels=classes)
  cm_percentage = 100 * cm.astype('float') / cm.sum(axis=1, keepdims=True)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=classes)
  fig, ax = plt.subplots()
  disp.plot(cmap='Blues', ax=ax, values_format='.1f')
  ax.set_title(f'Accuracy: {accuracy:.2f}')
  plt.show()

csv_data_path = '/content/drive/MyDrive/Veritas AI/Veritas AI - Arjun/data/cleaned_data.csv'

df = pd.read_csv(csv_data_path)

print(df.shape)

features = ['age', 'sex', 'race7', 'bmi_curr', 'bmi_20', 'cigpd_f', 'cig_years', 'rsmoker_f', 'lung_fh', 'asp', 'ibup',
            'arthrit_f', 'bronchit_f', 'colon_comorbidity', 'diabetes_f', 'divertic_f', 'emphys_f', 'gallblad_f',
            'hearta_f', 'hyperten_f', 'liver_comorbidity', 'osteopor_f', 'polyps_f', 'stroke_f']

seed = 42
image_size = (256, 256, 3)
feature_size = len(features)

# Undersample majority class for balanced dataset
df = undersample_majority(df, 'lung_cancer', seed)

# Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df['lung_cancer'])

print(df.shape)
print(train_df.shape, val_df.shape)

include_images = False
include_features = True

batch_size = 16
epochs = 200

train_ds = create_dataset(train_df, features, seed, include_images, include_features, image_size, batch_size, training=True)
val_ds = create_dataset(val_df, features, seed, include_images, include_features, image_size, batch_size, training=False)

feature_model = create_feature_model(feature_size)
feature_model.summary()

loss = tf.keras.losses.BinaryCrossentropy()

feature_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=loss, metrics=['accuracy'])
feature_history = feature_model.fit(train_ds, validation_data=val_ds, epochs=epochs, batch_size=batch_size)

plot_training_curves(feature_history)

feature_model = tf.keras.models.load_model('/content/drive/MyDrive/Veritas AI/Veritas AI - Arjun/code/models/MLP_model.h5')
feature_model.summary()

evaluate_model(feature_model, val_ds)

include_images = True
include_features = False

train_ds = create_dataset(train_df, features, seed, include_images, include_features, image_size, batch_size, training=True)
val_ds = create_dataset(val_df, features, seed, include_images, include_features, image_size, batch_size, training=False)

a, b = next(iter(val_ds))

k = 3
plt.figure()
plt.imshow(a[k, :, :, :], cmap='gray')
plt.show()
print(b[k])

image_model = create_image_model(image_size)
image_model.summary()

image_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
image_history = image_model.fit(train_ds, validation_data=val_ds, epochs=epochs, batch_size=batch_size)

plot_training_curves(image_history)

image_model = tf.keras.models.load_model('/content/drive/MyDrive/Veritas AI/Veritas AI - Arjun/code/models/CNN_model.h5')
image_model.summary()

evaluate_model(image_model, val_ds)

include_images = True
include_features = True

train_ds = create_dataset(train_df, features, seed, include_images, include_features, image_size, batch_size, training=True)
val_ds = create_dataset(val_df, features, seed, include_images, include_features, image_size, batch_size, training=False)

multimodal_model = create_multimodal_model(image_size, feature_size)
multimodal_model.summary()

multimodal_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
multimodal_history = multimodal_model.fit(train_ds, validation_data=val_ds, epochs=epochs, batch_size=batch_size)

plot_training_curves(multimodal_history)

multimodal_model = tf.keras.models.load_model('/content/drive/MyDrive/Veritas AI/Veritas AI - Arjun/code/models/hybrid_model.h5')
multimodal_model.summary()

evaluate_model(multimodal_model, val_ds)
