from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import LSTM

def create_model():
    #TUNE INPUT SHAPE TO THE SHAPE DESCRIBED IN EXCEPTION
    input_shape = 14748

    model = Sequential()
    model.add(Dense(2048, input_shape=(input_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    #model.add(LSTM(100))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()
    
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    return model
    


