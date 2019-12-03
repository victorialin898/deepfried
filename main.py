import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import get_data
from model import Model
import sys
import time

"""
    Trains model in batches on corrupted and original audio files.
"""

def train(model, corrupted, originals):
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

    @returns : average loss over input test data
"""
def test(model, corrupted, originals):

    losses = []

    bs = model.batch_size
    for i in range(len(corrupted) // bs):
        batch_corrupted = corrupted[i*bs : (i+1)*bs]
        batch_originals = originals[i*bs : (i+1)*bs]
        
        with tf.GradientTape() as tape:
            batch_sharpened = model.call(batch_corrupted)
            loss = model.loss_function(batch_sharpened, batch_originals)
            losses.append(loss)

    return tf.reduce_mean(losses)

def main():	
    if len(sys.argv) != 2 or sys.argv[1] not in {"VCTK","PIANO"}:
        print("USAGE: python assignment.py <Data Set>")
        print("<Data Set>: [VCTK/PIANO]")
        exit()
            
    print("Running preprocessing...")
    if sys.argv[1] == "VCTK":
        train_corrupted, train_originals, test_corrupted, test_originals = get_data(VCTK=True) 
    elif sys.argv[1] == "PIANO":
        train_corrupted, train_originals, test_corrupted, test_originals = get_data(VCTK=False)
    print("Preprocessing complete.")

    # model = Model()

    # print("Beginning training...")
    # for _ in range(model.epochs):
	#     train(model, train_corrupted, train_originals)
    # print("Training complete.")

    # print("Beginning testing...")
    # test(model, test_corrupted, test_originals)



if __name__ == '__main__':
   main()


