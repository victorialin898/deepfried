import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import get_dataset_iterator
from model import Model
import sys
import time
import soundfile as sf

"""
    @param batch : a tensor of (batch_size, 6000)
"""
def corrupt_batch(batch):
    pass

"""
    Trains model in batches on corrupted and original audio files.
"""
def train(model, train_data_iterator):
    for iteration, batch in enumerate(train_data_iterator):
        batch_corrupted = corrupt_batch(batch)

        with tf.GradientTape() as tape:
            batch_sharpened = model.call(batch_corrupted)
            loss = model.loss_function(batch_sharpened, batch)

        gradients = tape.gradient(loss, model.trainable_variables) # Maybe trainable_variables here?
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

"""
    Tests model in batches on corrupted and original audio files.

    @returns : average loss over input test data, sharpened audio files
"""
def test(model, test_data_iterator):
    losses = []

    for iteration, batch in enumerate(test_data_iterator):
        batch_corrupted = corrupt_batch(batch)
        batch_sharpened = model.call(batch_corrupted)
        loss = model.loss_function(batch_sharpened, batch_originals)
        losses.append(loss)

    return tf.reduce_mean(losses)

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"VCTK","PIANO"}:
        print("USAGE: python assignment.py <Data Set>")
        print("<Data Set>: [VCTK/PIANO]")
        exit()

    model = Model()

    print("Running preprocessing...")
    if sys.argv[1] == "VCTK":
        train_data_iterator, test_data_iterator = get_dataset_iterator(batch_size=model.batch_size, VCTK=True)
    elif sys.argv[1] == "PIANO":
        train_data_iterator, test_data_iterator = get_dataset_iterator(batch_size=model.batch_size, VCTK=False)
    print("Finished reprocessing.")

    # TODO: remove this loop. this is just to test that our data is the correct shape
    for iteration, data in enumerate(train_data_iterator):
        print(iteration, data.shape)

    print("Beginning training...")
    for _ in range(model.epochs):
	    train(model, train_data_iterator)
    print("Training complete.")

    print("Beginning testing...")
    loss = test(model, test_data_iterator)
    print("Average loss: " + str(loss.numpy()))

    # TODO: figure out a way to write some files to disk as demo
    # below is an attempt but probably doesn't work
    # Write 20 sharpened files as .wav for demo purposes
    # for i in range(len(demo_sr)):
    #     sf.write(file=os.path.join('output', str(i + 1)+'.wav'), data=sharpened, samplerate=demo_sr[i])


if __name__ == '__main__':
   main()
