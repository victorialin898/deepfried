# deepfried v0.0.0

## Introduction
In this project we aim to recreate the audio processing technique discussed in [Audio Super-Resolution Using Neural Nets by Volodymyr Kuleshov, S. Zayd Enam, and Stefano Ermon](https://arxiv.org/pdf/1708.00853.pdf). The paper focused on bandwidth extension, the audio generation problem of reconstructing high quality audio samples from low quality samples. The paper approaches this problem with a lightweight modeling algorithm using a deep convolutional neural network with residual connections. The described model takes sequential data and applies techniques like batch normalization, successive downsampling and upsampling blocks, skipping connections, and a subpixel shuffling layer (in order to increase the time dimension). Despite being significantly simpler, this model was able to outperform previous approaches to the bandwidth extension problem, such as the cubic B-spline baseline and a neural network-based technique discussed in a Dnn-based speech bandwidth… (Li, Huang, Xu, Lee).


This problem is a supervised learning problem, due to the methodology. We will be taking audio samples from various data sets (VCTK, Beethoven Sonata Piano) and downsampling them, and training our model to generate upsampled versions. To evaluate our model, we will compare the discretized audio signal predictions to the actual original audio sample, using a mean squared error loss function.

## Getting Started

Make sure to start your pipenv environment through running
```
pipenv shell
```

The dependencies and their versions are all locked in `Pipfile.lock`


## How to run

Due to the size of the dataset, we recommend you train this program using the Google Cloud Platform.

* Clone the github repo: `git clone https://github.com/victorialin898/deepfried.git`
* We used the cs142 env: `source /home/cs147/cs147_venv/bin/activate`
* You may need to download some libraries: `pip3 install librosa sndfile`
* If you get a build wheel error while installing sndfile run `sudo apt-get install libsndfile1-dev`
* Downgrade scipy: `pip3 install scipy==0.18.0` and ignore the librosa ERROR
* To download the VLTK dataset: within the root directory `deepfried/`, from your command line `cd data` to move into the data directory, and run `python3 download.py`. The VLTK corpus will begin downloading. Once the download bar reaches 100% the dataset will be located in the dataset folder.
* Run the program: `cd ..` and `python3 main.py VLTK`.
