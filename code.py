import matplotlib
import tensorflow as tf
import keras as K
import numpy as np
import pandas as pd
import IPython
from scipy.io import wavfile
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout, Conv1D, MaxPooling1D, Flatten
from keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from IPython import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


# plot an audio
def plot_time_series(data):
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()


# ------DATA AUGMENTATION-------

# add random noise, not too much
def add_noise_train(features, labels, scale=0.2, nr_original=8000):
    new_audios = np.random.normal(loc=0.0, scale=scale, size=(nr_original, features.shape[1], features.shape[2]))
    new_audios = train_features[:nr_original] + new_audios * train_features[:nr_original]
    features = np.concatenate((features, new_audios), axis=0)
    new_labels = labels[0:nr_original].copy()
    labels = np.concatenate((labels, new_labels))

    return features, labels


# shift sound, not too much
def shift_sound(features, labels, nr_original=8000):
    new_audios = features[:nr_original].copy()
    new_labels = labels[:nr_original].copy()

    for i in range(nr_original):
        shift = np.random.uniform(low=1600, high=4000)
        shift = int(shift)
        new_audios[i] = np.roll(new_audios[i], shift, axis=0)

    features = np.concatenate((features, new_audios), axis=0)
    labels = np.concatenate((labels, new_labels))

    return features, labels


# add noise AND shift sound
def noise_and_shift(features, labels, scale=0.2, nr_original=8000):
    new_audios = np.random.normal(loc=0.0, scale=scale, size=(nr_original, features.shape[1], features.shape[2]))
    new_audios = train_features[:nr_original] + new_audios * train_features[:nr_original]

    for i in range(nr_original):
        shift = np.random.uniform(low=1600, high=8000)
        shift = int(shift)
        new_audios[i] = np.roll(new_audios[i], shift, axis=0)

    features = np.concatenate((features, new_audios), axis=0)
    new_labels = labels[0:nr_original].copy()
    labels = np.concatenate((labels, new_labels))

    return features, labels


# check one training example
IPython.display.Audio("train/train/100001.wav")

# rate is the sample rate (how many samples are measured per second)
# data is a np.array with the values sampled from the audio file, it's length is rate * duration(in seconds)
rate, data = wavfile.read("train/train/102333.wav")
# print(data.shape)
# print(rate)
plot_time_series(data)

train_df = pd.read_csv("train.txt", header=None, names=["filename", "label"])
print(train_df)
val_df = pd.read_csv("validation.txt", header=None, names=["filename", "label"])
print(val_df)
test_df = pd.read_csv("test.txt", header=None, names=["filename"])
print(test_df)

# -------PROCESS TRAINING DATA--------

train_features = np.zeros((8000, 16000))
train_labels = np.zeros((8000))

# append each training example to train_features and train_labels
for i in range(train_df.shape[0]):
    if i % 1000 == 0:
        print(i)
    # extract file name
    filename = train_df.iloc[i]["filename"]

    # construct file path
    filepath = "train/train/" + filename

    # extract values sampled from the audio file
    rate, data = wavfile.read(filepath)

    # append the values corresponding to this example to the whole training features set
    train_features[i] = data

    # extract file label
    label = train_df.iloc[i]["label"]
    train_labels[i] = label

# add channels
train_features = np.reshape(train_features, (train_features.shape[0], train_features.shape[1], 1))
train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))

# visualize one training example
plot_time_series(train_features[6, :, 0])

# add new examples with artificial noise to the training set (this was used multiple times; in jupyter notebook I
# just rerun it as many times as I want)
train_features, train_labels = add_noise_train(train_features, train_labels)
print(train_features.shape, train_labels.shape)

# visualize the training example with noise added
plot_time_series(train_features[8006, :, 0])

# add new examples with sound shifted to the training set (this was used multiple times; in jupyter notebook I
# just rerun it as many times as I want)
train_features, train_labels = shift_sound(train_features, train_labels)
print(train_features.shape, train_labels.shape)

# visualize the training example with sound shifted
plot_time_series(train_features[16006, :, 0])

# add new features to training set with both noise and shift (this was used multiple times; in jupyter notebook I
# just rerun it as many times as I want)
train_features, train_labels = noise_and_shift(train_features, train_labels)
print(train_features.shape, train_labels.shape)

# visualize the training example with BOTH noise added and sound shifted
plot_time_series(train_features[24006, :, 0])

# -------PROCESS VALIDATION DATA--------
# extract file name
filename = val_df.iloc[0]["filename"]

# construct file path
filepath = "validation/validation/" + filename

# extract values sampled from the audio file
rate, data = wavfile.read(filepath)

# initialize validation features set with the first example
val_features = data
val_features = np.reshape(val_features, (1, val_features.shape[0]))

# extract file label
label = np.array([val_df.iloc[0]["label"]])

# initialize validation labels set with the first example
val_labels = label
val_labels = np.reshape(val_labels, (1, 1))

# append each validation example to val_features and val_labels
for i in range(1, val_df.shape[0]):
    if i % 100 == 0:
        print(i)
    # extract file name
    filename = val_df.iloc[i]["filename"]

    # construct file path
    filepath = "validation/validation/" + filename

    # extract values sampled from the audio file
    rate, data = wavfile.read(filepath)
    data = np.reshape(data, (1, data.shape[0]))

    # append the values coresponding to this example to the whole validation features set
    val_features = np.concatenate((val_features, data), axis=0)

    # extract file label
    label = np.array([val_df.iloc[i]["label"]])
    label = np.reshape(label, (1, 1))

    # append the label coresponding to this example to the whole validation labels set
    val_labels = np.concatenate((val_labels, label), axis=0)

# add channels
val_features = np.reshape(val_features, (val_features.shape[0], val_features.shape[1], 1))

# ----------PROCESS TEST DATA----------
# extract file name
filename = test_df.iloc[0]["filename"]

# construct file path
filepath = "test/test/" + filename

# extract values sampled from the audio file
rate, data = wavfile.read(filepath)

# initialize validation features set with the first example
test_features = data
test_features = np.reshape(test_features, (1, test_features.shape[0]))

# append each test example to test_features
for i in range(1, test_df.shape[0]):
    if i % 100 == 0:
        print(i)
    # extract file name
    filename = test_df.iloc[i]["filename"]

    # construct file path
    filepath = "test/test/" + filename

    # extract values sampled from the audio file
    rate, data = wavfile.read(filepath)
    data = np.reshape(data, (1, data.shape[0]))

    # append the values coresponding to this example to the whole test features set
    test_features = np.concatenate((test_features, data), axis=0)

# add channels
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

# -------SHUFFLE TRAINING SET(OPTIONAL)--------
# shuffle training set
perm = np.random.permutation(train_features.shape[0])
train_features = train_features[perm]
train_labels = train_labels[perm]


# -------MODEL ARCHITECTURE----------
# I have tested a lot of architectures, this is just one of the last, the others had mostly the same number
# of layers, but other hiperparameters were different
def best2():
    # create model
    model = Sequential()

    # add layers
    model.add(Conv1D(filters=16, kernel_size=15, strides=3, padding='same', kernel_regularizer=regularizers.l2(0.0045),
                     input_shape=train_features.shape[1:]))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))

    model.add(Conv1D(filters=16, kernel_size=15, strides=3, padding='same', kernel_regularizer=regularizers.l2(0.0045)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))

    model.add(Conv1D(filters=32, kernel_size=15, strides=3, padding='same', kernel_regularizer=regularizers.l2(0.0045)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))

    model.add(Conv1D(filters=32, kernel_size=15, strides=3, padding='same', kernel_regularizer=regularizers.l2(0.0045)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


# instantiate model
model = best2()

# compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train model
history = model.fit(x=train_features, y=train_labels, validation_data=(val_features, val_labels), batch_size=32,
                    epochs=100, shuffle=True)

# check models summary
print(model.summary())

# The code for the next two plots was copied from:
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# predict on validation set
preds_val = model.predict(val_features)
# convert predictions to 0/1
for i in range(preds_val.shape[0]):
    if preds_val[i, 0] >= 0.5:
        preds_val[i, 0] = 1
    else:
        preds_val[i, 0] = 0
preds_val = preds_val.astype('int32')
# print(preds_val)

# Confusion matrix for validation set, source:
# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
matrix = confusion_matrix(val_labels, preds_val)

print('Confusion Matrix :')
print(matrix)
print('Accuracy Score :', accuracy_score(val_labels, preds_val))
print('Report : ')
print(classification_report(val_labels, preds_val))

# predict test labels
preds = model.predict(test_features)
# save predictions
np.save("preds_array.npy", preds)

# convert predictions to 0/1
for i in range(preds.shape[0]):
    if preds[i][0] >= 0.5:
        preds[i][0] = 1
    else:
        preds[i][0] = 0
preds = preds.astype('int32')

# create dataframe with the predictions
preds_df = pd.DataFrame(preds, columns=["label"])
# print(preds_df[:20])

submission = pd.concat([test_df, preds_df], axis=1)
submission.columns = ["name", "label"]
#print(submission[0:50])

# create csv with the predictions
submission.to_csv(r'submission.txt', index=False, header=True)

# ----------GETTING FINAL RESULTS BY COMBINING RESULTS OF MULTIPLE PREVIOUS MODELS---------
# uploaded 6 predictions of 6 different CNN's to google drive and used them in my colab workspace
preds1 = np.load("/content/drive/My Drive/Colab Notebooks/ProiectML/preds_array.npy")
preds2 = np.load("/content/drive/My Drive/Colab Notebooks/ProiectML/preds_array 70.044%.npy")
preds3 = np.load("/content/drive/My Drive/Colab Notebooks/ProiectML/preds_array 69.66%.npy")
preds4 = np.load("/content/drive/My Drive/Colab Notebooks/ProiectML/preds_array 0.701.npy")
preds5 = np.load("/content/drive/My Drive/Colab Notebooks/ProiectML/preds_array 0.6911.npy")
preds6 = np.load("/content/drive/My Drive/Colab Notebooks/ProiectML/preds_array 69%.npy")

# combine the predictions
preds_final = preds1 + preds2 + preds3 + preds4 + preds5 + preds6
preds_final = 1 / 6 * preds_final

# convert the predictions to 0/1
for i in range(preds_final.shape[0]):
    if preds_final[i][0] >= 0.5:
        preds_final[i][0] = 1
    else:
        preds_final[i][0] = 0
preds_final = preds_final.astype('int32')

# create dataframe with the final predictions
preds_final_df = pd.DataFrame(preds_final, columns=["label"])
# print(preds_final_df[:20])

submission_final = pd.concat([test_df, preds_final_df], axis=1)
submission_final.columns = ["name", "label"]
# print(submission_final[0:50])

# create csv with the final predictions
submission_final.to_csv(r'submission_final.txt', index=False, header=True)
