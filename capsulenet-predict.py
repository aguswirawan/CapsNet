from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers,regularizers
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')

import pandas as pd
import time
import pickle
import numpy as np
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import callbacks

def deap_load(data_file,dimention,debaseline):
    rnn_suffix = ".mat_win_128_rnn_dataset.pkl"
    label_suffix_valence = ".mat_win_128_labels_valence.pkl"
    label_suffix_arousal = ".mat_win_128_labels_arousal.pkl"
    arousal_or_valence = dimention
    with_or_without = debaseline # 'yes','not'
    dataset_dir = "deap_shuffled_MWMF_fractional/" + with_or_without + "_" + arousal_or_valence + "/"

    ###load training set
    with open(dataset_dir + data_file + rnn_suffix, "rb") as fp:
        rnn_datasets = pickle.load(fp)
    with open(dataset_dir + data_file + label_suffix_valence, "rb") as fp:
        label_valence = pickle.load(fp)
        # labels = pickle.load(fp)
        # labels = np.transpose(labels[0])
    with open(dataset_dir + data_file + label_suffix_arousal, 'rb') as fp:
        label_arousal = pickle.load(fp)

    def preprocess_two_labels(label_a, label_b):
        if label_a == 1 and label_b == 1:
            return 'HAHV'
        elif label_a == 0 and label_b == 1:
            return 'LAHV'
        elif label_a == 1 and label_b == 0:
            return 'HALV'
        elif label_a == 0 and label_b == 0:
            return 'LALV'

    labels = list(map(preprocess_two_labels, label_arousal.tolist()[0], label_valence.tolist()[0]))
    # print(labels[:5])

    from sklearn.preprocessing import LabelEncoder
    from keras.utils import np_utils
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    print(encoder.classes_)
    # convert integers to dummy variables (i.e. one hot encoded)
    # print(encoded_labels[:5])
    labels = np_utils.to_categorical(encoded_labels)
    # print(labels[:5, :])

    # labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)


    # shuffle data
    # index = np.array(range(0, len(labels)))
    # np.random.shuffle(index)
    # rnn_datasets = rnn_datasets[index]  # .transpose(0,2,1)
    # labels = labels[index]

    print(rnn_datasets.shape)
    datasets = rnn_datasets.reshape(-1, 9, 9, 4).astype('float32')
    labels = labels.astype('float32')
    print(datasets.shape)
    print(labels.shape)

    return datasets , labels, encoder.classes_

# X = load_data
subject = 's01'
dimention = 'all'
debaseline = 'yes'
X, y, labels_map = deap_load(subject,dimention,debaseline)

def CapsNet(input_shape, n_class, routings, batch_size):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :param batch_size: size of batch
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape, batch_size=batch_size)
    # print(x.shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=64, kernel_size=4, strides=1, padding='same', activation='relu', name='conv1')(x)
    # print(conv1.shape)
    conv2 = layers.Conv2D(filters=128, kernel_size=4, strides=1, padding='same', activation='relu', name='conv2')(conv1)
    conv3 = layers.Conv2D(filters=256, kernel_size=4, strides=1, padding='same', activation='relu', name='conv3')(conv2)
    # conv4 = layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu', name='conv4')(conv3)
    # out_flat = layers.Flatten()(conv3)
    # conv5 = layers.Conv2D(filters=1024, kernel_size=1, strides=1, padding='same', activation='selu', name='conv5')(conv4)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    # primarycaps = PrimaryCap(conv3, dim_capsule=8, n_channels=32, kernel_size=4, strides=2, padding='same')
    primarycaps = layers.Reshape(target_shape=[-1, 8], name='primarycap_reshape')(conv3)
    # print('primarycaps shape:')
    # print(primarycaps.shape)

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)
    # out_caps = layers.Softmax(name='capsnet')(out_caps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    # masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    # masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    # decoder = models.Sequential(name='decoder')
    # decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    # decoder.add(layers.Dense(1024, activation='relu'))
    # decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    # decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    # train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    # eval_model = models.Model(x, [out_caps, decoder(masked)])
    train_model = models.Model([x, y], out_caps)
    eval_model = models.Model(x, out_caps)


    # manipulate model
    # noise = layers.Input(shape=(n_class, 16))
    # noised_digitcaps = layers.Add()([digitcaps, noise])
    # masked_noised_y = Mask()([noised_digitcaps, y])
    # manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    # return train_model, eval_model, manipulate_model
    return train_model, eval_model

dataset_name = 'deap'
model_version = 'v0'
epochs = 30
save_dir = 'result_MWMF_fractional_ver2/sub_dependent_'+ model_version +'/' + debaseline + '/' + subject + '_' + dimention + str(epochs)
fold = 9
# define model
model, eval_model = CapsNet(input_shape=X.shape[1:],
                                              n_class=4,
                                              routings=3,
                                              batch_size=120)
model.summary()
model.load_weights(f'{save_dir}/trained_model_fold{fold}.h5')
# y_pred = model.predict(X, batch_size=120)
y_pred = eval_model.predict(X, batch_size=120)
# print(y_pred)
# print(y_pred.shape)
y_decoded = [labels_map[value]for value in np.argmax(y, axis=1)]
y_pred_decoded = [labels_map[value]for value in np.argmax(y_pred, axis=1)]
pd.DataFrame({'second': range(len(y_decoded)),
              'label': y_decoded,
              'label_pred': y_pred_decoded}).to_excel(f'{save_dir}/predicted_label_fold{fold}.xlsx', index=False)
