import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout, ActivityRegularization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.regularizers import l2
from lifelines import utils
import h5py
from common.config import Config

###############
# 1.Data Load #
###############
# 1-1.Clinical 엑셀 데이터 로드
df_train = pd.read_excel(f'../SurvivalClassification/preprocess/output/{Config.img_shape}/TRAIN_CLINICAL_pre.xlsx')
df_valid = pd.read_excel(f'../SurvivalClassification/preprocess/output/{Config.img_shape}/VALID_CLINICAL_pre.xlsx')
df_test = pd.read_excel(f'../SurvivalClassification/preprocess/output/{Config.img_shape}/TEST_CLINICAL_pre.xlsx')

X_train = df_train.copy()
X_valid = df_valid.copy()
X_test = df_test.copy()
E_train = df_train["Deadstatus.event"]
E_valid = df_valid["Deadstatus.event"]
E_test = df_test["Deadstatus.event"]
Y_train = df_train["Survival.time"]
Y_valid = df_valid["Survival.time"]
Y_test = df_test["Survival.time"]

X_train.drop(['PatientID', '5year_survival', 'Survival.time', 'Deadstatus.event'], axis='columns', inplace=True)
X_valid.drop(['PatientID', '5year_survival', 'Survival.time', 'Deadstatus.event'], axis='columns', inplace=True)
X_test.drop(['PatientID', '5year_survival', 'Survival.time', 'Deadstatus.event'], axis='columns', inplace=True)

X_train = np.array(X_train).astype('float32')
X_valid = np.array(X_valid).astype('float32')
X_test = np.array(X_test).astype('float32')
E_train = np.array(E_train).astype('float32')
E_valid = np.array(E_valid).astype('float32')
E_test = np.array(E_test).astype('float32')
Y_train = np.array(Y_train).astype('float32')
Y_valid = np.array(Y_valid).astype('float32')
Y_test = np.array(Y_test).astype('float32')

n_features = X_train.shape[1]


def negative_log_likelihood(E):
    def loss(y_true, y_pred):
        hazard_ratio = tf.math.exp(y_pred)
        log_risk = tf.math.log(tf.math.cumsum(hazard_ratio))
        uncensored_likelihood = tf.transpose(y_pred) - log_risk
        censored_likelihood = uncensored_likelihood * E
        neg_likelihood_ = -tf.math.reduce_sum(censored_likelihood)
        num_observed_events = tf.constant(1, dtype=tf.float32)
        neg_likelihood = neg_likelihood_ / num_observed_events
        return neg_likelihood
    return loss


activation = 'relu'
n_nodes = 48
learning_rate = 0.067
l2_reg = 16.094
dropout = 0.147
lr_decay = 6.494e-4
momentum = 0.863


# Create model
ff_input = Input(shape=(n_features,))
x = Dense(units=n_features, activation=activation, kernel_initializer='glorot_uniform', input_shape=(n_features,))(ff_input)
x = Dropout(dropout)(x)
x = Dense(units=n_nodes, activation=activation, kernel_initializer='glorot_uniform')(x)
x = Dropout(dropout)(x)
x = Dense(units=n_nodes, activation=activation, kernel_initializer='glorot_uniform')(x)
x = Dropout(dropout)(x)
x = Dense(units=1, activation='linear', kernel_initializer='glorot_uniform', kernel_regularizer=l2(l2_reg))(x)
ff_output = ActivityRegularization(l2=l2_reg)(x)

model = Model([ff_input], ff_output)

# Nadam is Adam + Nesterov momentum
optimizer = Nadam(learning_rate=learning_rate, weight_decay=lr_decay)

# Compile the model and show a summary of it
model.compile(loss=negative_log_likelihood(E_train), optimizer=optimizer)
model.summary()

file_name = f"deepsurvk_central.h5"
callbacks = [tf.keras.callbacks.TerminateOnNaN(),
             tf.keras.callbacks.ModelCheckpoint((Path("./report") / file_name), monitor='loss', save_best_only=True, mode='min')]

history = model.fit(X_train, Y_train,
                    batch_size=64,  # 환자 수
                    epochs=100,
                    callbacks=callbacks,
                    shuffle=False)

model = load_model((Path("./report") / file_name), compile=False)

Y_pred_test = np.exp(-model.predict(X_test))
c_index_test = utils.concordance_index(Y_test, Y_pred_test, E_test)
print(f"c-index of testing dataset = {c_index_test}")
