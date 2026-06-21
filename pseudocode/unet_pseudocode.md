# U-Net pseudocode (MaxPool + ReLU convs + Up-conv)

This file contains a focused U-Net pseudocode using the components you specified:
- Downsampling with MaxPool2d
- Convolutional blocks using ReLU activations
- Upsampling with up-convolution (ConvTranspose2d / "UpConv")

## Contract (inputs / outputs)
- Input: image tensor of shape (B, C_in, H, W)
- Output: segmentation tensor of shape (B, C_out, H, W)
- Note: spatial dims will be restored by up-convolutions; keep input dims divisible by 2^depth or pad/resize.

## High-level flow
1. Encoder: repeated ConvReLUBlock -> MaxPool2d (store skip outputs before pooling)
2. Bottleneck: ConvReLUBlock at smallest resolution
3. Decoder: UpConv (ConvTranspose2d) -> Concat(skip) -> ConvReLUBlock
4. Final 1x1 conv -> activation (sigmoid/softmax)

---

## Layer subprograms (pseudocode)

function ConvReLUBlock(x, out_channels, kernel=3):
    # Two conv layers with ReLU nonlinearity (common U-Net block)
    x = Conv2D(x, out_channels, kernel_size=kernel, padding=1)
    x = BatchNorm(x)           # optional but recommended
    x = ReLU(x)
    x = Conv2D(x, out_channels, kernel_size=kernel, padding=1)
    x = BatchNorm(x)
    x = ReLU(x)
    return x

function MaxPool2x2(x):
    # Downsample by factor 2
    return MaxPool2D(x, kernel_size=2, stride=2)

function UpConv2x2(x, out_channels, kernel=2, stride=2):
    # Up-convolution (learned upsampling) using ConvTranspose2d
    # This doubles spatial H,W and reduces channel dimension to out_channels
    return ConvTranspose2D(x, out_channels, kernel_size=kernel, stride=stride)

function ConcatChannels(a, b):
    # Concatenate along channel dimension
    # Assumes spatial dims match (H and W)
    return Concat(a, b, axis=channel)

function FinalConv(x, n_classes):
    x = Conv2D(x, out_channels=n_classes, kernel_size=1)
    if n_classes == 1:
        return Sigmoid(x)
    else:
        return Softmax(x, axis=channel)

---

## UNet (uses the above subprograms)

function UNet(input, n_channels=3, n_classes=1, base_filters=64, depth=4):
    # input: (B, C_in, H, W)
    x = input
    skips = []
    filters = base_filters

    # Encoder (downsampling)
    for i in range(depth):
        x = ConvReLUBlock(x, filters)
        skips.append(x)            # store for skip connection
        x = MaxPool2x2(x)         # downsample by 2
        filters = filters * 2

    # Bottleneck
    x = ConvReLUBlock(x, filters)

    # Decoder (upsampling)
    for i in range(depth):
        filters = filters // 2
        x = UpConv2x2(x, filters)   # learned upsampling
        skip = skips.pop()          # corresponding encoder output
        x = ConcatChannels(x, skip)
        x = ConvReLUBlock(x, filters)

    # Output mapping
    output = FinalConv(x, n_classes)
    return output

---

## Example hyperparameters and notes
- base_filters = 64 (common), depth = 4 means 4 poolings → spatial /16 at bottleneck
- optimizer = Adam(lr=1e-4)
- loss = BCEWithLogitsLoss (binary) or CrossEntropyLoss (multi-class)
- batch_size chosen based on GPU memory

## Training loop sketch

for epoch in range(epochs):
    for images, targets in dataloader:
        preds = UNet(images)
        loss = LossFn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validate, log metrics, save checkpoint

## Implementation notes and edge cases
- Use padding=1 in Conv2D to preserve spatial dims inside ConvReLUBlock.
- If input H/W aren't divisible by 2^depth, pad or resize before feeding the network.
- ConvTranspose2d can produce off-by-one spatial sizes for odd sizes; test shapes and adjust padding/output_padding if needed.
- BatchNorm is optional but often stabilizes training.

---

Edit or expand any subprogram into a concrete PyTorch/TensorFlow function if you want runnable code.