import tensorflow as tf
import musdb
import numpy as np





def stft(x, n_FFT, hop_length):
    """inp:
    x: input audio signal
    n_FFT: number of FFT bins for STFT (2048 is common)
    hop_length: hop length for STFT (512 is common with 2048 n_FFT for 75% overlap of windows)
    out:
    X: STFT of the input audio signal
    """

    stft = tf.signal.stft(x, frame_length=n_FFT, frame_step=hop_length, fft_length=n_FFT)                       #tensorflow stft function
    magnitude = tf.abs(stft)                                                                                    #finding the magnitude of the complex stft output (what we are actually interested in)
    return magnitude







    
def load_data(set, chunk_len, n_FFT, hop_length):
    """inp: 
    set: musdb subset to load (train, test, validation)
    chunk_len: length of each chunk in seconds
    n_FFT: number of FFT bins for STFT (2048 is common)
    hop_length: hop length for STFT (512 is common with 2048 n_FFT for 75% overlap of windows)
    out:
    X: array of original audio chunks
    Y: array of corresponding drum audio chunks"""

    set = musdb.DB(root = "data/musdb18", subsets=set)

    chunk_size = 44100 * chunk_len                      # chunk_len seconds at 44100 Hz sampling rate

    X = []
    Y = []

    for track in set:
        mix = track.audio.mean(axis=1)                               #orignal audio
        drums = track.targets['drums'].audio.mean(axis=1)            #target drum audio

        num_chunks = mix.shape[0] // chunk_size         #finding number of chunks in the audio

        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size                    #defining the start and end of each chunk
            X.append(mix[start:end])
            Y.append(drums[start:end])                  #appending the chunks to the respective arrays


    return np.array(X), np.array(Y)




def conv_block(x, filters):
    """inp:
    x: input tensor
    filters: number of filters for the convolutional layers
    out:
    x: output tensor
    """
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    return x


def model_def():
    inputs = tf.keras.Input(shape=(128, 128, 1))

    #encoder
    c1 = conv_block(inputs, 64)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    #bottleneck
    c5 = conv_block(p4, 1024)

    #decoder
    u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.Concatenate()([u6, c4])
    c6 = conv_block(u6, 512)

    u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.Concatenate()([u7, c3])
    c7 = conv_block(u7, 256)

    u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.Concatenate()([u8, c2])
    c8 = conv_block(u8, 128)

    u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.Concatenate()([u9, c1])
    c9 = conv_block(u9, 64)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c9)
    return tf.keras.Model(inputs=inputs, outputs=outputs)






X, Y = load_data("train", chunk_len=5, n_FFT=2048, hop_length=512)






model = model_def()
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


