from __future__ import division
import keras
import pandas as pd
import numpy as np
import pickle
import argparse
from keras.optimizers import Adam
print( 'Using Keras version', keras.__version__)

########### Arguments parsing

# Instantiate the parser
parser = argparse.ArgumentParser(description='Caltech CNN training test')

# Required number of epochs for training
parser.add_argument('epochs', type=int,
                    help='Number of epochs to be run')


parser.add_argument('batch_size', type=int,
                     help='Number of images per step')

# # Optional argument
parser.add_argument('dense_size', type=int,
                     help='An optional integer argument')
#
# # Switch
# parser.add_argument('--switch', action='store_true',
#                     help='A boolean switch')


args = parser.parse_args()

#############

############## Variables initialization


Num_epochs=args.epochs
Batch_size=args.batch_size
img_size = 128
dense_size=args.dense_size


############# Data Augmentation

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2, 
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

#############

#from keras.datasets import mnist

#Load the MNIST dataset, already provided by Keras
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Check sizes of dataset
#print( 'Number of train examples', x_train.shape[0])
#print( 'Size of train examples', x_train.shape[1:])

#Adapt the data as an input of a fully-connected (flatten to 1D)
#x_train = x_train.reshape(60000, 784)
#x_test = x_test.reshape(10000, 784)

#Normalize data
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train = x_train / 255
#x_test = x_test / 255

############# Loading data from preprocessed files

pickle_in = open("Data/pickle_all_images_df1.pickle","rb")
all_images_df1 = pickle.load(pickle_in)

pickle_in = open("Data/pickle_all_images_df2.pickle","rb")
all_images_df2 = pickle.load(pickle_in)

pickle_in = open("Data/pickle_all_images_df3.pickle","rb")
all_images_df3 = pickle.load(pickle_in)

pickle_in = open("Data/pickle_all_classes.pickle","rb")
all_classes = pickle.load(pickle_in)


all_images = np.concatenate((all_images_df1, all_images_df2,all_images_df3), axis=0)

del all_images_df1
del all_images_df2
del all_images_df3


all_classes = pd.get_dummies(all_classes)
all_images = np.array(all_images)

############## Preparing train/test data


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_images, all_classes, test_size=0.1)

del all_images
X_train = X_train.reshape(-1,img_size,img_size,3)
X_test = X_test.reshape(-1,img_size,img_size,3)

#Adapt the labels to the one-hot vector syntax required by the softmax
#from keras.utils import np_utils
#y_train = np_utils.to_categorical(y_train, 257)
#y_test = np_utils.to_categorical(y_test, 257)


#Find which format to use (depends on the backend), and compute input_shape
from keras import backend as K

#Caltech resolution
img_rows, img_cols, channels = img_size, img_size, 3

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], channels, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, 3)

##############

############## Define the NN architecture

from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
#Two hidden layers
nn = Sequential()
nn.add(Conv2D(32, 3, 3, activation='relu', input_shape=input_shape))
#nn.add(BatchNormalization())
#nn.add(Dropout(0.2))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(64, 3, 3, activation='relu'))
#nn.add(BatchNormalization())
#nn.add(Dropout(0.2))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(128, 3, 3, activation='relu'))
#nn.add(BatchNormalization())
#nn.add(Dropout(0.2))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(256, 3, 3, activation='relu'))
#nn.add(Dropout(0.2))
nn.add(MaxPooling2D(pool_size=(2, 2)))
#nn.add(Conv2D(512, 3, 3, activation='relu'))
#nn.add(Dropout(0.2))
#nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Flatten())
nn.add(Dense(dense_size, activation='relu'))
nn.add(Dropout(0.4))
nn.add(Dense(257, activation='softmax'))


#Model visualization
#We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
#from keras.util import plot_model
#plot_model(nn, to_file='nn.png', show_shapes=true)

#Compile the NN
adam=Adam()
nn.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

print(nn.summary())

#############

############# Start training
#history = nn.fit(x_train,y_train,batch_size=128,epochs=20, validation_split=0.15)
history = nn.fit_generator(generator.flow(X_train, y_train, batch_size=Batch_size), epochs=Num_epochs, verbose=1,validation_data=(X_test, y_test))


############# Model Evaluation

#Evaluate the model with test set
score = nn.evaluate(X_test, y_test, verbose=0)
print('test loss:', score[0])
print('test accuracy:', score[1])


#Store Plots
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
plt.savefig('./model3/4L-ImgAug-dout-adam_cnn_accuracy_batch'+'_'+str(Num_epochs)+'_'+str(Batch_size)+'_'+str(dense_size)+'.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('./model3/4L-ImgAug-dout-adam_cnn_loss_batch'+'_'+str(Num_epochs)+'_'+str(Batch_size)+'_'+str(dense_size)+'.pdf')

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
#Compute probabilities
Y_pred = nn.predict(X_test)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#y_pred=Y_pred
y_test = pd.Series(y_test.columns[np.where(y_test!=0)[1]])

#print(y_test)

y_test = [int(label.split('.',1)[0]) for label in y_test]

#Plot statistics
print( 'Analysis of results' )
target_names = ['001.ak47', '002.american-flag', '003.backpack', '004.baseball-bat', '005.baseball-glove', '006.basketball-hoop', '007.bat', '008.bathtub', '009.bear', '010.beer-mug', '011.billiards', '012.binoculars', '013.birdbath', '014.blimp', '015.bonsai-101', '016.boom-box', '017.bowling-ball', '018.bowling-pin', '019.boxing-glove', '020.brain-101', '021.breadmaker', '022.buddha-101', '023.bulldozer', '024.butterfly', '025.cactus', '026.cake', '027.calculator', '028.camel', '029.cannon', '030.canoe', '031.car-tire', '032.cartman', '033.cd', '034.centipede', '035.cereal-box', '036.chandelier-101', '037.chess-board', '038.chimp', '039.chopsticks', '040.cockroach', '041.coffee-mug', '042.coffin', '043.coin', '044.comet', '045.computer-keyboard', '046.computer-monitor', '047.computer-mouse', '048.conch', '049.cormorant', '050.covered-wagon', '051.cowboy-hat', '052.crab-101', '053.desk-globe', '054.diamond-ring', '055.dice', '056.dog', '057.dolphin-101', '058.doorknob', '059.drinking-straw', '060.duck', '061.dumb-bell', '062.eiffel-tower', '063.electric-guitar-101', '064.elephant-101', '065.elk', '066.ewer-101', '067.eyeglasses', '068.fern', '069.fighter-jet', '070.fire-extinguisher', '071.fire-hydrant', '072.fire-truck', '073.fireworks', '074.flashlight', '075.floppy-disk', '076.football-helmet', '077.french-horn', '078.fried-egg', '079.frisbee', '080.frog', '081.frying-pan', '082.galaxy', '083.gas-pump', '084.giraffe', '085.goat', '086.golden-gate-bridge', '087.goldfish', '088.golf-ball', '089.goose', '090.gorilla', '091.grand-piano-101', '092.grapes', '093.grasshopper', '094.guitar-pick', '095.hamburger', '096.hammock', '097.harmonica', '098.harp', '099.harpsichord', '100.hawksbill-101', '101.head-phones', '102.helicopter-101', '103.hibiscus', '104.homer-simpson', '105.horse', '106.horseshoe-crab', '107.hot-air-balloon', '108.hot-dog', '109.hot-tub', '110.hourglass', '111.house-fly', '112.human-skeleton', '113.hummingbird', '114.ibis-101', '115.ice-cream-cone', '116.iguana', '117.ipod', '118.iris', '119.jesus-christ', '120.joy-stick', '121.kangaroo-101', '122.kayak', '123.ketch-101', '124.killer-whale', '125.knife', '126.ladder', '127.laptop-101', '128.lathe', '129.leopards-101', '130.license-plate', '131.lightbulb', '132.light-house', '133.lightning', '134.llama-101', '135.mailbox', '136.mandolin', '137.mars', '138.mattress', '139.megaphone', '140.menorah-101', '141.microscope', '142.microwave', '143.minaret', '144.minotaur', '145.motorbikes-101', '146.mountain-bike', '147.mushroom', '148.mussels', '149.necktie', '150.octopus', '151.ostrich', '152.owl', '153.palm-pilot', '154.palm-tree', '155.paperclip', '156.paper-shredder', '157.pci-card', '158.penguin', '159.people', '160.pez-dispenser', '161.photocopier', '162.picnic-table', '163.playing-card', '164.porcupine', '165.pram', '166.praying-mantis', '167.pyramid', '168.raccoon', '169.radio-telescope', '170.rainbow', '171.refrigerator', '172.revolver-101', '173.rifle', '174.rotary-phone', '175.roulette-wheel', '176.saddle', '177.saturn', '178.school-bus', '179.scorpion-101', '180.screwdriver', '181.segway', '182.self-propelled-lawn-mower', '183.sextant', '184.sheet-music', '185.skateboard', '186.skunk', '187.skyscraper', '188.smokestack', '189.snail', '190.snake', '191.sneaker', '192.snowmobile', '193.soccer-ball', '194.socks', '195.soda-can', '196.spaghetti', '197.speed-boat', '198.spider', '199.spoon', '200.stained-glass', '201.starfish-101', '202.steering-wheel', '203.stirrups', '204.sunflower-101', '205.superman', '206.sushi', '207.swan', '208.swiss-army-knife', '209.sword', '210.syringe', '211.tambourine', '212.teapot', '213.teddy-bear', '214.teepee', '215.telephone-box', '216.tennis-ball', '217.tennis-court', '218.tennis-racket', '219.theodolite', '220.toaster', '221.tomato', '222.tombstone', '223.top-hat', '224.touring-bike', '225.tower-pisa', '226.traffic-light', '227.treadmill', '228.triceratops', '229.tricycle', '230.trilobite-101', '231.tripod', '232.t-shirt', '233.tuning-fork', '234.tweezer', '235.umbrella-101', '236.unicorn', '237.vcr', '238.video-projector', '239.washing-machine', '240.watch-101', '241.waterfall', '242.watermelon', '243.welding-mask', '244.wheelbarrow', '245.windmill', '246.wine-bottle', '247.xylophone', '248.yarmulke', '249.yo-yo', '250.zebra', '251.airplanes-101', '252.car-side-101', '253.faces-easy-101', '254.greyhound', '255.tennis-shoes', '256.toad', '257.clutter']

print(len(target_names))
print(len(y_test))
print(y_pred.shape)

print()

#print(classification_report(y_test, y_pred,target_names=target_names))
#print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

#print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))
cm=confusion_matrix(y_test, y_pred)
cm = cm / cm.sum(axis=1)
import pylab as pl
pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.savefig('./model3/4L-ImgAug-dout-adam_cnn_confmat'+'_'+str(Num_epochs)+'_'+str(Batch_size)+'_'+str(dense_size)+'.pdf')

############## Saving model

#Saving model and weights
from keras.models import model_from_json
nn_json = nn.to_json()
with open('./models/4L-ImgAug-adam_cnn_confmat'+'_'+str(Num_epochs)+'_'+str(Batch_size)+'_'+str(dense_size)+'.json', 'w') as json_file:
        json_file.write(nn_json)
weights_file = "weights-MNIST_"+str(score[1])+".hdf5"
nn.save_weights(weights_file,overwrite=True)

#Loading model and weights
json_file = open('./models/4L-adam_cnn_confmat'+'_'+str(Num_epochs)+'_'+str(Batch_size)+'_'+str(dense_size)+'.json','r')
nn_json = json_file.read()
json_file.close()
nn = model_from_json(nn_json)
nn.load_weights(weights_file)
