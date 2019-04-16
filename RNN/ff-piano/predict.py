""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord, converter
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, GRU
from keras.layers import Activation
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            print('input')
            print(frac_str)
            print('out')
            print(float(num) / float(denom))
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))


def print_distribution(freqs):

    plt.bar(list(freqs.keys()), freqs.values(), color='g')
    #plt.axes.xaxis.set_ticklabels([])
    plt.xticks([])
    plt.xlabel('Different sounds')
    plt.ylabel('Number occurrences')
    plt.title('Data distibution')
    plt.show()



def generate(**kwargs):
    """ Generate a piano midi file """
    #load the notes used to train the model
    
    song = kwargs.get('song', None)

    with open('data/tnotes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))

    print(len(notes))
    freqs={}
    for i in notes:
     if i in freqs.keys():# if key is present in the list, just append the value
         freqs[i] = freqs[i]+1
     else:
         freqs[i] = 1 # else create a empty list as value for the key
    print('most common element:')
    print(freqs[max(freqs, key=freqs.get)])
    print(max(freqs, key=freqs.get))
    print('least common element')
    print(freqs[min(freqs, key=freqs.get)])
    print(min(freqs, key=freqs.get))
    print(len(freqs))
    print(freqs)
    #print_distribution(freqs)
    #exit(1)
    # Get all pitch names
    n_vocab = len(set(notes))

    print(n_vocab)
    if song == None:
        network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
        model = create_network(normalized_input, n_vocab)
        prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    else:
        network_input, normalized_input = prepare_sequences(song, pitchnames, n_vocab)
        print(network_input)
        model = create_network(normalized_input, n_vocab)
        prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)

    create_midi(prediction_output)

def generate_song(name):

    notes = []
    midi = converter.parse(name)
    offset=0
    last_offset=0

    notes_to_parse = None

    try: # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            #print(str(element.offset))
            #print(str(element.pitch))
            offset = str(round(element.offset-last_offset,2))
            last_offset=element.offset
            sound = str(element.pitch)+':'+ offset
            notes.append(sound)
        elif isinstance(element, chord.Chord):
            offset = str(round(element.offset-last_offset,2))
            last_offset=element.offset
            sound = '.'.join(str(n) for n in element.normalOrder)
            sound = sound +':' + offset
            notes.append(sound)
            #print(element.normalOrder)
            #print(element.offset)
            #notes.append('.'.join(str(n) for n in element.normalOrder))

        #print(notes)
    with open('data/song', 'wb') as filepath:
        pickle.dump(notes, filepath)

    generate(song=notes)




def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    print(note_to_int)
    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.5))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')

    # Load the weights to each node
    model.load_weights('/home/ohernand/PycharmProjects/DL-Caltech/Classical-Piano-Composer-master/8/weights-improvement-189-0.9793-bigger.hdf5')

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[1]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    float(offset)
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        split = pattern.split(':')
        print(split)
        pattern = split[0]
        time = split[1]
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += convert_to_float(time)

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='8/prelude-189-0.9793.mid')

if __name__ == '__main__':
    #generate()
    #generate_song('/home/ohernand/PycharmProjects/DL-Caltech/Classical-Piano-Composer-master/midi_data/FFX-ToZanarkand.mid')
    #generate_song('/home/ohernand/PycharmProjects/DL-Caltech/Classical-Piano-Composer-master/midi_data/ff7aerith.mid')
    generate_song('/home/ohernand/PycharmProjects/DL-Caltech/Classical-Piano-Composer-master/midi_data/ff1prelude.mid')
    #generate_song('/home/ohernand/PycharmProjects/DL-Caltech/Classical-Piano-Composer-master/midi_data/ff7choco.mid')
    #generate_song('/home/ohernand/PycharmProjects/DL-Caltech/Classical-Piano-Composer-master/midi_data/ff7goldsaucer.mid')
    #generate_song('/home/ohernand/PycharmProjects/DL-Caltech/Classical-Piano-Composer-master/midi_data/ff4chocobo.mid')
    #generate_song('/home/ohernand/PycharmProjects/DL-Caltech/Classical-Piano-Composer-master/midi_data/ff9melodiesoflife.mid')
