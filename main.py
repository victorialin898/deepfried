import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import get_data
from model import Model
import sys
import time


def train(model, corrupted, originals):

    for i in range(int(len(corrupted)/model.batch_size)):
        train_x = corrupted[i*model.batch_size:(i+1)*model.batch_size]
        train_y = originals[i*model.batch_size:(i+1)*model.batch_size]
        
        with tf.GradientTape() as tape:
            encodings = model.call(train_x)
            loss = model.loss_function(encodings, train_y)
        
        gradients = tape.gradient(loss, model.variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, corrupted, originals):
    for i in range(int(len(corrupted)/model.batch_size)):
        train_x = corrupted[i*model.batch_size:(i+1)*model.batch_size]
        train_y = originals[i*model.batch_size:(i+1)*model.batch_size]
        
        with tf.GradientTape() as tape:
            encodings = model.call(train_x)
            loss = model.loss_function(encodings, train_y)

        gradients = tape.gradient(loss, model.variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #TODO should be calculating loss here i think

    return 

def main():	
    if len(sys.argv) != 2 or sys.argv[1] not in {"VCTK","PIANO"}:
        print("USAGE: python assignment.py <Data Type>")
        print("<Data Type>: [VCTK/PIANO]")
        exit()
            
    print("Running preprocessing...")
    if sys.argv[1] == "VCTK":
        train_corrupted, train_originals, test_corrupted, test_originals = get_data("filepath", VCTK=True)  # TODO: finish preprocessing
    elif sys.argv[1] == "PIANO":
        train_corrupted, train_originals, test_corrupted, test_originals = get_data("filepath", VCTK=False)

    print("Preprocessing complete.")

    model = Model()

    for _ in range(model.epochs):
	    train(model, train_corrupted, train_originals)

    test(model, test_corrupted, test_originals)



if __name__ == '__main__':
   main()


