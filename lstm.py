from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout

def create_model():
    model = Sequential()
    model.add(Dense(2048, input_shape=(7245,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
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
    


