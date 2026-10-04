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
        self.X = X_padded  # save the input for back prop
        self.pads = (pad_top, pad_bottom, pad_left, pad_right)  # save the padding for back prop

        X_out = np.zeros((B_in, C_out, H_out, W_out))            #create output tensor

        
        for b in range(B_in):                   #loop over the batches
            for c_out in range(C_out):          #loop over the filters
                for h in range(H_out):          
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + H_f
                        w_end = w_start + W_f           #finding the start and end of the current window in the input tensor

                        X_out[b, c_out, h, w] = np.sum(X_padded[b, :, h_start:h_end, w_start:w_end] * self.W[c_out]) + self.b[c_out] 

        return X_out


    def backwards(self, dX_out):
        """inp:
            dX_out: gradient of the loss with respect to the output tensor of shape (B_in, C_out, H_out, W_out) from the next layer
        out:
            dX: gradient of the loss with respect to the input tensor of shape (B_in, C_in, H_in, W_in)
            dW: gradient of the loss with respect to the weight tensor of shape (C_out, C_in, H_f, W_f)
            db: gradient of the loss with respect to the bias tensor of shape (C_out, 1)
            """

        X_pad = self.X
        W = self.W
        b = self.b

        B_in, C_in, H_in_pad, W_in_pad = X_pad.shape
        C_out, C_in_w, H_f, W_f = W.shape
        B_in, C_out, H_out, W_out = dX_out.shape

        dX_pad = np.zeros_like(X_pad)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)           #creating the gradients matrices to be same size as the originals

        db = np.sum(dX_out, axis=(0, 2, 3)).reshape(C_out, 1)  # formula for the bias gradient

        for b in range(B_in):                   #loop over the batches
            for c_out in range(C_out):          #loop over the filters
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + H_f
                        w_end = w_start + W_f           #finding the start and end of the current window in the input tensor

                        dX_pad[b, :, h_start:h_end, w_start:w_end] += W[c_out] * dX_out[b, c_out, h, w]  # formula for the input gradient
                        dW[c_out] += X_pad[b, :, h_start:h_end, w_start:w_end] * dX_out[b, c_out, h, w]  # formula for the weight gradient

        pad_top, pad_bottom, pad_left, pad_right = self.pads
        H_in = H_in_pad - pad_top - pad_bottom
        W_in = W_in_pad - pad_left - pad_right
        dX = dX_pad[:, :, pad_top:pad_top + H_in, pad_left:pad_left + W_in]  # remove padding from the input gradient

        self.dW = dW
        self.db = db
        return dX












class max_pool:
    def __init__(self, pool_size):
        """parameters:
        pool_size: size of the pooling window (pxp)"""
        self.pool_size = pool_size

        
    def forward(self, X):
        """Inp:
        X: input tensor of shape (B_in, C_in, H_in, W_in)
        p: pool size (pxp)
        out:
        X: output tensor of shape (B_in, C_in, H_out, W_out)"""


        B_in, C_in, H_in, W_in = X.shape
        self.in_shape = (H_in, W_in)  # save the input shape for back prop
        
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
        self.X = X_padded  # save the input for back prop
        self.pads = (pad_top, pad_bottom, pad_left, pad_right)  # save the padding for back prop


        out_tensor = np.zeros((B_in, C_in, H_out, W_out))

        for b in range(B_in):
            for c in range(C_in):
                for row in range(H_out):
                    for col in range(W_out):
                        window = X_padded[b, c, row * self.pool_size:row * self.pool_size + self.pool_size, col * self.pool_size:col * self.pool_size + self.pool_size] # define the current window
                        out_tensor[b, c, row, col] = np.max(window)
        return out_tensor




    def backwards(self, dX_out):
        """inp:
            dX_out: gradient of the loss with respect to the output tensor of shape (B_in, C_in, H_out, W_out) from the next layer
        out:
            dX: gradient of the loss with respect to the input tensor of shape (B_in, C_in, H_in, W_in)
            """

        X_pad = self.X
        p =  self.pool_size
        B_in, C_in, H_out, W_out = dX_out.shape

        dX_pad = np.zeros_like(X_pad)

        for b in range(B_in):
            for c in range(C_in):
                for row in range(H_out):
                    for col in range(W_out):
                        h_start = row * p
                        w_start = col * p
                        h_end = h_start + p
                        w_end = w_start + p
                        window = X_pad[b, c, h_start:h_end, w_start:w_end]
                        i, j = np.unravel_index(np.argmax(window), window.shape)  # find the index of the max value in the window
                        dX_pad[b, c, h_start + i, w_start + j] += dX_out[b, c, row, col]  # assign the gradient to the max values position


        pad_top, pad_bottom, pad_left, pad_right = self.pads
        dX = dX_pad[:, :, pad_top:dX_pad.shape[2] - pad_bottom, pad_left:dX_pad.shape[3] - pad_right]  # remove padding from the input gradient
        return dX





class ReLU:
    def __init__(self):
        self.X = None
        
    def forward(self, X):
        """Inp:
        X: input tensor of shape (B_in, C_in, H_in, W_in)
        out:
        X: output tensor of shape (B_in, C_in, H_in, W_in)"""

        self.X = X
        return np.maximum(0, X)

    def backwards(self, dX_out, X=None):
        """inp:
            dX_out: gradient of the loss with respect to the output tensor of shape (B_in, C_in, H_in, W_in) from the next layer
            X: input tensor of shape (B_in, C_in, H_in, W_in)
        out:
            dX: gradient of the loss with respect to the input tensor of shape (B_in, C_in, H_in, W_in)
            """
        
        if X is None:
            X = self.X
        dX = dX_out * (X > 0) #checks if every input value X is > 0, for these values dx = dxout, else 0
        return dX





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
        self.W = np.random.randn(C_out, C_in, H_f, W_f) * scale                   #He normal initialization of the weights
        self.b = np.zeros((C_out, 1))                                               #initializing the bias to zero

    def forward(self, X):
        """Inp:
        X: input tensor of shape (B_in, C_in, H_in, W_in)
        out:
        X: output tensor of shape (B_in, C_out, H_out, W_out)"""

        self.X = X  # save the input for back prop
        B_in, C_in, H_in, W_in = X.shape
        C_out, C_in_w, H_f, W_f = self.W.shape
        s = self.stride

        # calculate the output dimensions
        H_out = (H_in - 1) * s + H_f
        W_out = (W_in - 1) * s + W_f

        X_out = np.zeros((B_in, C_out, H_out, W_out))

        for b in range(B_in):
            for c_out in range(C_out):
                for h in range(H_in):
                    for w in range(W_in):
                        h_start = h * s
                        w_start = w * s
                        h_end = h_start + H_f
                        w_end = w_start + W_f

                        X_out[b, c_out, h_start:h_end, w_start:w_end] += np.sum(X[b, :, h, w, None, None] * self.W[c_out],axis=0,)

        X_out += self.b.reshape(1, C_out, 1, 1)
        return X_out


    def backwards(self, dX_out):
        """inp:
            dX_out: gradient of the loss with respect to the output tensor of shape (B_in, C_out, H_out, W_out) from the next layer
        out:
            dX: gradient of the loss with respect to the input tensor of shape (B_in, C_in, H_in, W_in)
            dW: gradient of the loss with respect to the weight tensor of shape (C_out, C_in, H_f, W_f)
            db: gradient of the loss with respect to the bias tensor of shape (C_out, 1)
            """

        X = self.X
        W = self.W
        s = self.stride

        B_in, C_in, H_in, W_in = X.shape
        C_out, C_in_w, H_f, W_f = W.shape

        dX = np.zeros_like(X)
        dW = np.zeros_like(W)           #creating the gradients matrices to be same size as the originals


        for b in range(B_in):
            for c_out in range(C_out):
                for h in range(H_in):
                    for w in range(W_in):
                        h_start = h * s
                        w_start = w * s
                        h_end = h_start + H_f
                        w_end = w_start + W_f

                        dX[b, :, h, w] += np.sum(W[c_out] * dX_out[b, c_out, h_start:h_end, w_start:w_end], axis=(1, 2))  # formula for the input gradient
                        dW[c_out] += X[b, :, h, w, None, None] * dX_out[b, c_out, h_start:h_end, w_start:w_end]  # formula for the weight gradient

        self.dW = dW
        db = np.sum(dX_out, axis=(0, 2, 3)).reshape(C_out, 1)  # formula for the bias gradient
        self.db = db

        return dX
    





class Unet:
    def __init__(self):
        #encoder
        self.conv1_1 = conv(1, 16, 3, 3, 1) # conv layer with 1 input channel, 16 output channels, 3x3 filter and stride of 1
        self.relu1_1 = ReLU() # relu activation
        self.conv1_2 = conv(16, 16, 3, 3, 1) # conv layer with 16 input channels, 16 output channels, 3x3 filter and stride of 1
        self.relu1_2 = ReLU() # relu activation   

        self.pool1 = max_pool(2) # max pooling layer with pool size of 2x2

        self.conv2_1 = conv(16, 32, 3, 3, 1) # conv layer with 16 input channels, 32 output channels, 3x3 filter and stride of 1
        self.relu2_1 = ReLU() # relu activation
        self.conv2_2 = conv(32, 32, 3, 3, 1) # conv layer with 32 input channels, 32 output channels, 3x3 filter and stride of 1
        self.relu2_2 = ReLU() # relu activation

        self.pool2 = max_pool(2) # max pooling layer with pool size of 2x2


        #bottleneck
        self.conv3_1 = conv(32, 64, 3, 3, 1) # conv layer with 32 input channels, 64 output channels, 3x3 filter and stride of 1
        self.relu3_1 = ReLU() # relu activation
        self.conv3_2 = conv(64, 64, 3, 3, 1) # conv layer with 64 input channels, 64 output channels, 3x3 filter and stride of 1
        self.relu3_2 = ReLU() # relu activation


        #decoder
        self.upconv1 = up_conv(64, 32, 2, 2, 2) # up conv layer with 64 input channels, 32 output channels, 2x2 filter and stride of 2

        self.conv4_1 = conv(64, 64, 3, 3, 1) # conv layer with 64 input channels, 64 output channels, 3x3 filter and stride of 1
        self.relu4_1 = ReLU() # relu activation
        self.conv4_2 = conv(64, 64, 3, 3, 1) # conv layer with 64 input channels, 64 output channels, 3x3 filter and stride of 1
        self.relu4_2 = ReLU() # relu activation

        self.upconv2 = up_conv(64, 32, 2, 2, 2) # up conv layer with 64 input channels, 32 output channels, 2x2 filter and stride of 2

        self.conv5_1 = conv(48, 64, 3, 3, 1) # conv layer with 48 input channels, 64 output channels, 3x3 filter and stride of 1
        self.relu5_1 = ReLU() # relu activation
        self.conv5_2 = conv(64, 64, 3, 3, 1) # conv layer with 64 input channels, 64 output channels, 3x3 filter and stride of 1
        self.relu5_2 = ReLU() # relu activation

        self.conv_final = conv(64, 1, 1, 1, 1) # conv layer with 64 input channels, 1 output channel, 1x1 filter and stride of 1


    def forward(self, X):
        #encoder
        c1_1 = self.conv1_1.forward(X)
        r1_1 = self.relu1_1.forward(c1_1)
        c1_2 = self.conv1_2.forward(r1_1)
        r1_2 = self.relu1_2.forward(c1_2)
        self.skip1 = r1_2 # save the output of the first block for skip connection

        p1 = self.pool1.forward(r1_2)

        c2_1 = self.conv2_1.forward(p1)
        r2_1 = self.relu2_1.forward(c2_1)
        c2_2 = self.conv2_2.forward(r2_1)
        r2_2 = self.relu2_2.forward(c2_2)
        self.skip2 = r2_2 # save the output of the second block for skip connection

        p2 = self.pool2.forward(r2_2)

        #bottleneck
        c3_1 = self.conv3_1.forward(p2)
        r3_1 = self.relu3_1.forward(c3_1)
        c3_2 = self.conv3_2.forward(r3_1)
        r3_2 = self.relu3_2.forward(c3_2)

        #decoder
        u1 = self.upconv1.forward(r3_2)
        concat1 = np.concatenate((u1, self.skip2), axis=1) # concatenate the output of the upconv layer with the skip connection
        c4_1 = self.conv4_1.forward(concat1)
        r4_1 = self.relu4_1.forward(c4_1)
        c4_2 = self.conv4_2.forward(r4_1)
        r4_2 = self.relu4_2.forward(c4_2)

        u2 = self.upconv2.forward(r4_2)
        concat2 = np.concatenate((u2, self.skip1), axis=1) # concatenate the output of the upconv layer with the skip connection
        c5_1 = self.conv5_1.forward(concat2)
        r5_1 = self.relu5_1.forward(c5_1)
        c5_2 = self.conv5_2.forward(r5_1)
        r5_2 = self.relu5_2.forward(c5_2)

        output = self.conv_final.forward(r5_2)
        return output

    def backwards(self, dX_out):
        dr5_2 = self.conv_final.backwards(dX_out)

        #decoder
        dc5_2 = self.relu5_2.backwards(dr5_2)
        dr5_1 = self.conv5_2.backwards(dc5_2)
        dc5_1 = self.relu5_1.backwards(dr5_1)
        dconcat2 = self.conv5_1.backwards(dc5_1)
        du2, dskip1 = np.split(dconcat2, [32], axis=1) # split the gradient 
        self.dskip1 = dskip1 # save the gradient of the skip connection

        dr4_2 = self.upconv2.backwards(du2)

        dc4_2 = self.relu4_2.backwards(dr4_2)
        dr4_1 = self.conv4_2.backwards(dc4_2)
        dc4_1 = self.relu4_1.backwards(dr4_1)
        dconcat1 = self.conv4_1.backwards(dc4_1)
        du1, dskip2 = np.split(dconcat1, [32], axis=1)
        self.dskip2 = dskip2 

        dr3_2 = self.upconv1.backwards(du1)

        #bottleneck
        dc3_2 = self.relu3_2.backwards(dr3_2)
        dr3_1 = self.conv3_2.backwards(dc3_2)
        dc3_1 = self.relu3_1.backwards(dr3_1)

        dp2 = self.conv3_1.backwards(dc3_1)

        #encoder
        dr2_2 = self.pool2.backwards(dp2)
        dr2_2_sum = dr2_2 + self.dskip2 # add the gradient of the skip connection

        dc2_2 = self.relu2_2.backwards(dr2_2_sum)
        dr2_1 = self.conv2_2.backwards(dc2_2)
        dc2_1 = self.relu2_1.backwards(dr2_1)

        dp1 = self.conv2_1.backwards(dc2_1)

        dr1_2 = self.pool1.backwards(dp1)
        dr1_2_sum = dr1_2 + self.dskip1 

        dc1_2 = self.relu1_2.backwards(dr1_2_sum)
        dr1_1 = self.conv1_2.backwards(dc1_2)
        dc1_1 = self.relu1_1.backwards(dr1_1)
        dx = self.conv1_1.backwards(dc1_1)

        return dx


    def get_parameters(self):
        layers = [self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2, self.conv3_1, self.conv3_2, self.upconv1, self.conv4_1, self.conv4_2, self.upconv2, self.conv5_1, self.conv5_2, self.conv_final]

        params = []
        for layer in layers:
            params.append((layer.W, layer.dW))  # append the weights and their gradients
            params.append((layer.b, layer.db))  # append the biases and their gradients
        return params


    
def s_gradient_descent(model, X, y, learning_rate=0.01):
    params = model.get_parameters()
    for param, grad in params:
        param -= learning_rate * grad  # update the weights and biases using the gradients


def mse_loss(y_pred, y_true):
    """calculate the mean squared error loss"""
    loss = np.mean((y_pred - y_true) ** 2)
    grad = 2 * (y_pred - y_true) / y_true.size  #formula for gradient of loss 
    return loss, grad


def train(model, X_train, y_train, epochs=10, learning_rate=0.01):
    for epoch in range(epochs):
        y_pred = model.forward(X_train)
        loss, grad = mse_loss(y_pred, y_train)
        model.backwards(grad)
        s_gradient_descent(model, X_train, y_train, learning_rate)
        print(f"step {epoch + 1}, Loss: {loss:.4f}")


