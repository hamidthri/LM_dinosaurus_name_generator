import numpy as np
from utils import *
import random
import pprint
import copy

data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(ix_to_char)

def clip(gradients, maxValue):
    gradients = copy.deepcopy(gradients)
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out = gradient)
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients

def sample(parameters, char_to_ix, seed):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size, 1))
    print(x.shape)
    a_prev = np.zeros((n_a, 1))
    indices = []
    idx = -1
    counter = 0
    newline_character = char_to_ix['\n']

    while (idx != newline_character and counter != 50):
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        np.random.seed(counter + seed)
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())
        indices.append(idx)
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a
        seed += 1
        counter +=1
    if (counter == 50):
        indices.append(char_to_ix['\n'])
    print(f'indices are {indices}')
    return indices

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, 5)
    parameters = update_parameters(parameters, gradients, learning_rate)
    return loss, gradients, a[len(X)-1]

def model(data_x, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27, verbose = False):
    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)
    loss = get_initial_loss(vocab_size, dino_names)
    examples = [x.strip() for x in data_x]
    print(examples)
    np.random.seed(0)
    np.random.shuffle(examples)
    a_prev = np.zeros((n_a, 1))
    last_dino_name = "abc"
    for j in range(num_iterations):
        idx = j % len(examples)
        single_example = examples[idx]
        single_example_chars = list(single_example)
        single_example_ix = [ char_to_ix[char] for char in single_example_chars ]
        X = [None] + single_example_ix
        ix_newline = char_to_ix['\n']
        Y = X[1:] + [char_to_ix["\n"]]
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)
        if verbose and j in [0, len(examples) -1, len(examples)]:
            print("j = " , j, "idx = ", idx,)
        if verbose and j in [0]:
            print("single_example =", single_example)
            print("single_example_chars", single_example_chars)
            print("single_example_ix", single_example_ix)
            print(" X = ", X, "\n", "Y =       ", Y, "\n")
        loss = smooth(loss, curr_loss)
        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            seed = 0
            for name in range(dino_names):
                sampled_indices = sample(parameters, char_to_ix, seed)
                last_dino_name = get_sample(sampled_indices, ix_to_char)
                print(last_dino_name.replace('\n', ''))

                seed += 1

            print('\n')

    return parameters, last_dino_name

parameters, last_name = model(data.split("\n"), ix_to_char, char_to_ix, 22001, verbose = True)
