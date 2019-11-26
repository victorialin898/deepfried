# deepfried v0.0.0

## Introduction
In this project we aim to recreate the audio processing technique discussed in Audio Super-Resolution Using Neural Nets by Volodymyr Kuleshov, S. Zayd Enam, and Stefano Ermon. The paper focused on bandwidth extension, the audio generation problem of reconstructing high quality audio samples from low quality samples. The paper approaches this problem with a lightweight modeling algorithm using a deep convolutional neural network with residual connections. The described model takes sequential data and applies techniques like batch normalization, successive downsampling and upsampling blocks, skipping connections, and a subpixel shuffling layer (in order to increase the time dimension). Despite being significantly simpler, this model was able to outperform previous approaches to the bandwidth extension problem, such as the cubic B-spline baseline and a neural network-based technique discussed in a Dnn-based speech bandwidth… (Li, Huang, Xu, Lee).


This problem is a supervised learning problem, due to the methodology. We will be taking audio samples from various data sets (VCTK, Beethoven Sonata Piano) and downsampling them, and training our model to generate upsampled versions. To evaluate our model, we will compare the discretized audio signal predictions to the actual original audio sample, using a mean squared error loss function.

