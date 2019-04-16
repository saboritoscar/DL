""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from time import time
from music21 import converter, instrument, note, chord
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


times = [0.25, 0.5, 1, 2, 5]

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))
    print(n_vocab)
    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    load=0
    offset=0
    last_offset=0
    if load:
        for file in glob.glob("midi_data/*.mid"):

            print("Parsing %s" % file)

            midi = converter.parse(file)


            notes_to_parse = None

            try: # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            #print(notes_to_parse)
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    #print(str(element.offset))
                    #print(str(element.pitch))
                    offset = round(element.offset-last_offset,2)
                    offset= str(min(times, key=lambda x:abs(x-offset)))
                    print(offset)
                    last_offset=element.offset
                    sound = str(element.pitch)+':'+ offset
                    notes.append(sound)
                elif isinstance(element, chord.Chord):
                    offset = round(element.offset-last_offset,2)
                    offset= str(min(times, key=lambda x:abs(x-offset)))
                    print(offset)
                    last_offset=element.offset
                    sound = '.'.join(str(n) for n in element.normalOrder)
                    sound = sound +':' + offset
                    notes.append(sound)
                    #print(element.normalOrder)
                    #print(element.offset)
                    #notes.append('.'.join(str(n) for n in element.normalOrder))

            print(notes)
        with open('/gpfs/projects/bsc99/bsc99526/OTHERS/Classical-Piano-Composer-master/data/5tnotes', 'wb') as filepath:
            pickle.dump(notes, filepath)
    else:
        with open('/gpfs/projects/bsc99/bsc99526/OTHERS/Classical-Piano-Composer-master/data/5tnotes', 'rb') as filepath:
            notes = pickle.load(filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(512))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    callbacks_list.append(tensorboard)
    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
