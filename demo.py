import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
import musdb

def load_audio(file_path, sr=44100):
    """load a mp3 file and return the audio signal and sample rate"""
    audio, sr = lr.load(file_path, sr=sr)
    return audio, sr


def STFT(audio, n_fft=2048, hop_length=512):
    """turn into spectrogram"""
    stft = lr.stft(audio, n_fft=n_fft, hop_length=hop_length)
    return stft


def plot_spectrogram(stft, sr=44100, hop_length=512):
    """plot the spectrogram of the audio signal"""

    magnitude = np.abs(stft)
    magnitude_db = lr.amplitude_to_db(magnitude, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(magnitude_db, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (bins)')
    plt.tight_layout()
    plt.show()











class conv:
    def __init__(self, C_in, C_out, H_f, W_f, stride):
        """parameters:
        C_in: number of input channels
        C_out: number of output channels
        H_f: height of the filter
        W_f: width of the filter
        stride: stride """

        self.C_in = C_in
        self.C_out = C_out
        self.H_f = H_f
        self.W_f = W_f
        self.stride = stride


        scale = np.sqrt(2.0 / (C_in * H_f * W_f))
        self.W = np.random.randn(C_out, C_in, H_f, W_f) * scale                   #He normal initialization of the weights
        self.b = np.zeros((C_out, 1))                                               #initializing the bias to zero


        
    def forward(self, X):
        """Inp:
        X: input tensor of shape (B_in, C_in, H_in, W_in)
        W: weight tensor of shape (C_out, C_in, H_f, W_f)
        s: stride
        out:
        X: output tensor of shape (B_in, C_out, H_out, W_out)"""

        B_in, C_in, H_in, W_in = X.shape
        C_out, C_in_w, H_f, W_f = self.W.shape


        # calculate the output dimensions
        H_out = int(np.ceil(H_in / self.stride))
        W_out = int(np.ceil(W_in / self.stride))



        # calculate the total padding needed
        if H_in % self.stride == 0:
            pad_h = max(H_f - self.stride, 0)
        else:
            pad_h = max(H_f - (H_in % self.stride), 0)
        if W_in % self.stride == 0:
            pad_w = max(W_f - self.stride, 0)
        else:
            pad_w = max(W_f - (W_in % self.stride), 0)



        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left



        X_padded = np.pad(X,((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),mode="constant", constant_values=0)      # pad 

        X_out = np.zeros((B_in, C_out, H_out, W_out))            # create output tensor

        
        for b in range(B_in):                   #loop over the batches
            for c_out in range(C_out):          #loop over the filters
                for h in range(H_out):          
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + H_f
                        w_end = w_start + W_f           #finding the start and end of the current window in the input tensor

                        X_out[b, c_out, h, w] = np.sum(X_padded[b, :, h_start:h_end, w_start:w_end] * W[c_out])

        return X_out


    def backwards(self, dX_out):
        """Inp:
        dX_out: gradient of the loss with respect to the output tensor of shape (B_in, C_out, H_out, W_out)
        out:
        dX: gradient of the loss with respect to the input tensor of shape (B_in, C_in, H_in, W_in)"""

        

    







class max_pool:
    def __init__(self, pool_size):
        """parameters:
        pool_size: size of the pooling window (pxp)"""
        self.pool_size = pool_size

        
    def max_pool_forward_prop(self, X):
        """Inp:
        X: input tensor of shape (B_in, C_in, H_in, W_in)
        p: pool size (pxp)
        out:
        X: output tensor of shape (B_in, C_in, H_out, W_out)"""


        B_in, C_in, H_in, W_in = X.shape
        
        H_out = int(np.ceil(H_in / self.pool_size))
        W_out = int(np.ceil(W_in / self.pool_size))                  # calculate output tensor dimensions

        # calculate the total padding needed
        if H_in % self.pool_size == 0:
            pad_h = 0
        else:
            pad_h = max(self.pool_size - (H_in % self.pool_size), 0)

        if W_in % self.pool_size == 0:
            pad_w = 0
        else:
            pad_w = max(self.pool_size - (W_in % self.pool_size), 0)


        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        X_padded = np.pad(X, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)  

        out_tensor = np.zeros((B_in, C_in, H_out, W_out))

        for b in range(B_in):
            for c in range(C_in):
                for row in range(H_out):
                    for col in range(W_out):
                        window = X_padded[b, c, row * self.pool_size:row * self.pool_size + self.pool_size, col * self.pool_size:col * self.pool_size + self.pool_size] # define the current window
                        out_tensor[b, c, row, col] = np.max(window)
        return out_tensor







class ReLU:
    def relu_forward_prop(self, X):
        """Inp:
        X: input tensor of shape (B_in, C_in, H_in, W_in)
        out:
        X: output tensor of shape (B_in, C_in, H_in, W_in)"""

        return np.maximum(0, X)





class up_conv:
    def __init__(self, C_in, C_out, H_f, W_f, stride):
        """parameters:
        C_in: number of input channels
        C_out: number of output channels
        H_f: height of the filter
        W_f: width of the filter
        stride: stride """

        self.C_in = C_in
        self.C_out = C_out
        self.H_f = H_f
        self.W_f = W_f
        self.stride = stride


        scale = np.sqrt(2.0 / (C_in * H_f * W_f))
        self.W = np.random.randn((C_out, C_in, H_f, W_f)) * scale                   #He normal initialization of the weights
        self.b = np.zeros((C_out, 1))                                               #initializing the bias to zero

    def up_conv_forward_prop(self, X):
        """Inp:
        X: input tensor of shape (B_in, C_in, H_in, W_in)
        out:
        X: output tensor of shape (B_in, C_out, H_out, W_out)"""


        B_in, C_in, H_in, W_in = X.shape
        C_in_w, C_out, H_f, W_f = self.W.shape

        # calculate the output dimensions
        H_out = (H_in - 1) * self.stride + H_f
        W_out = (W_in - 1) * self.stride + W_f

        X_out = np.zeros((B_in, C_out, H_out, W_out))

        for b in range(B_in):
            for c_out in range(C_out):
                for h in range(H_in):
                    for w in range(W_in):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + H_f
                        w_end = w_start + W_f

                        X_out[b, c_out, h_start:h_end, w_start:w_end] += X[b, :, h, w] @ W[:, c_out]

        return X_out






#show input and show spectrogram of input 

song, sr = load_audio("C:/Users/toma/OneDrive/Documents/visual studio code/NEA/The Prodigy - Breathe.mp3")
stft = STFT(song)
plot_spectrogram(stft)
