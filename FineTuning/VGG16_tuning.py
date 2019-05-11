from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import pickle
import pandas as pd
import numpy as np
from time import time
import argparse

parser = argparse.ArgumentParser(description='Caltech CNN training test')

# Required number of epochs for training
parser.add_argument('epochs', type=int,
                    help='Number of epochs to be run')


parser.add_argument('freeze_layers', type=int,
                     help='Number of layers frozen')

parser.add_argument('lr', type=float,
                     help='learning rate')

parser.add_argument('optimizer', type=str,
                     help='optimizer')



args = parser.parse_args()


Num_epochs=args.epochs
flayers=args.freeze_layers
lrate=args.lr
optimizer=args.optimizer


# Plot the training and validation loss + accuracy
def plot_training(history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    #Accuracy plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.title('Training and validation accuracy')
    plt.savefig('fine_tuning_accuracy_org_'+str(flayers)+optimizer+str(lrate)+'.pdf')
    plt.close()
    #Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.title('Training and validation loss')
    plt.savefig('fine_tuning_loss_org_'+str(flayers)+optimizer+str(lrate)+'.pdf')



img_size, img_width, img_height = 128, 128, 128
#train_data_dir = "/gpfs/scratch/bsc28/hpai/storage/data/datasets/original/mit67/train"
#validation_data_dir = "/gpfs/scratch/bsc28/hpai/storage/data/datasets/original/mit67/test"
#nb_train_samples = 5359
#nb_validation_samples = 1339
batch_size = 64
epochs = Num_epochs
target_classes = 257

#####preparing input data

pickle_in = open("../../Caltech/Data/pickle_all_images_df1.pickle","rb")
all_images_df1 = pickle.load(pickle_in)

pickle_in = open("../../Caltech/Data/pickle_all_images_df2.pickle","rb")
all_images_df2 = pickle.load(pickle_in)

pickle_in = open("../../Caltech/Data/pickle_all_images_df3.pickle","rb")
all_images_df3 = pickle.load(pickle_in)

pickle_in = open("../../Caltech/Data/pickle_all_classes.pickle","rb")
all_classes = pickle.load(pickle_in)


all_images = np.concatenate((all_images_df1, all_images_df2,all_images_df3), axis=0)

del all_images_df1
del all_images_df2
del all_images_df3


all_classes = pd.get_dummies(all_classes)
all_images = np.array(all_images)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_images, all_classes, test_size=0.1)

del all_images
X_train = X_train.reshape(-1,img_size,img_size,3)
X_test = X_test.reshape(-1,img_size,img_size,3)

nb_train_samples = len(y_train)
nb_validation_samples = len(y_test)

model = applications.VGG16(weights=None, include_top=False, input_shape = (img_width, img_height, 3))
#model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:flayers]:
    layer.trainable = False
#for i, layer in enumerate(model.layers):
#    print(i, layer.name)

#Adding custom Layers 
#x = model.output
#x = Flatten()(x)
#x = Dense(512, activation="relu")(x)
#x = Dropout(0.5)(x)
#x = Dense(512, activation="relu")(x)
#predictions = Dense(257, activation="softmax")(x)


x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(257, activation='softmax')(x)






# creating the final model 
model_final = Model(inputs = model.input, output = predictions)

if optimizer=='adam':

	opt = optimizers.Adam(lr=lrate)

elif optimizer=='sgd': 
	opt = optimizers.SGD(lr=lrate, momentum=0.9)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"])


# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
        horizontal_flip = True,
        fill_mode = "nearest",
        zoom_range = 0.3,
        width_shift_range = 0.3,
        height_shift_range=0.3,
        rotation_range=30)

val_datagen = ImageDataGenerator(
        horizontal_flip = True,
        fill_mode = "nearest",
        zoom_range = 0.3,
        width_shift_range = 0.3,
        height_shift_range=0.3,
        rotation_range=30)

train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size = batch_size)

validation_generator = val_datagen.flow(
        X_test, y_test,
        batch_size = batch_size)

# Save the model according to the conditions  
#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')



tensorboard = TensorBoard(log_dir="logs/VGG16_noWeights"+str(flayers)+optimizer+str(lrate))

# Train the model 
history = model_final.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples//batch_size,
        epochs = epochs,
        validation_data = validation_generator,
        validation_steps=nb_validation_samples//batch_size,
        callbacks = [tensorboard])#[early])#,checkpoint])

plot_training(history)
