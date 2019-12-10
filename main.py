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
    Trains model in batches on corrupted and original audio files.
"""

def train(model, train_data_iterator):
    # TODO: reimplement parts of this using the train_data_iterator

    bs = model.batch_size
    for i in range(len(corrupted) // bs):
        batch_corrupted = corrupted[i*bs : (i+1)*bs]
        batch_originals = originals[i*bs : (i+1)*bs]

        with tf.GradientTape() as tape:
            batch_sharpened = model.call(batch_corrupted)
            loss = model.loss_function(batch_sharpened, batch_originals)

        gradients = tape.gradient(loss, model.variables) # Maybe trainable_variables here?
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

"""
    Tests model in batches on corrupted and original audio files.

    @returns : average loss over input test data, sharpened audio files
"""
def test(model, test_data_iterator):
    # TODO: reimplement parts of this using the test_data_iterator

    losses = []
    sharpened = []

    bs = model.batch_size
    for i in range(len(corrupted) // bs):
        batch_corrupted = corrupted[i*bs : (i+1)*bs]
        batch_originals = originals[i*bs : (i+1)*bs]

        with tf.GradientTape() as tape:
            batch_sharpened = model.call(batch_corrupted)
            sharpened += batch_sharpened
            loss = model.loss_function(batch_sharpened, batch_originals)
            losses.append(loss)

    return tf.reduce_mean(losses), sharpened

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"VCTK","PIANO"}:
        print("USAGE: python assignment.py <Data Set>")
        print("<Data Set>: [VCTK/PIANO]")
        exit()

    print("Running preprocessing...")
    if sys.argv[1] == "VCTK":
        train_data_iterator, test_data_iterator = get_dataset_iterator(VCTK=True)
    elif sys.argv[1] == "PIANO":
        train_data_iterator, test_data_iterator = get_dataset_iterator(VCTK=False)
    print("Finished reprocessing.")
    
    # TODO: remove this loop. this is just to test that our data is the correct shape
    for iteration, data in enumerate(train_data_iterator):
        print(iteration, data.shape)


    model = Model()

    print("Beginning training...")
    for _ in range(model.epochs):
	    train(model, train_data_iterator)
    print("Training complete.")

    print("Beginning testing...")
    loss, sharpened = test(model, test_data_iterator)
    print("Average loss: " + str(loss.numpy()))

    # Write 20 sharpened files as .wav for demo purposes
    for i in range(len(demo_sr)):
        sf.write(file=os.path.join('output', str(i + 1)+'.wav'), data=sharpened, samplerate=demo_sr[i])


if __name__ == '__main__':
   main()
