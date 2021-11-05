# -*- coding: utf-8 -*-
"""
# Backpropagation step-by-step

## Shorthand:
- "pd_" as a variable prefix means "partial derivative"
- "d_" as a variable prefix means "derivative"
- "\_wrt_" is shorthand for "with respect to"
- "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively

## Comment references:

- [1] Wikipedia article on Backpropagation
  http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
- [2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton
  https://class.coursera.org/neuralnets-2012-001/lecture/39
- [3] The Back Propagation Algorithm
  https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf

## Live Example
You can see here a live example of how the weights changes during the training.

http://www.emergentmind.com/neural-network# 

(note that the site has an error in the JS, you may want to override that. 

See: https://developer.chrome.com/blog/new-in-devtools-65/#overrides )

First of all, define our Neuron.

Each neuron will have the weights and the bias.
"""

import random
import math
import numpy as np

# definizione del neurone utilizzano le CLASSI in py
class Neuron:
    # init method or constructor
    def __init__(self, bias): # define all variables of the class
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs): # calculate the output value of the neuron 
        self.inputs = inputs
        # COME FUNZIONE DI USCITA UTILIZZA SIGMOIDE *****
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self): # calculate the input val (input*weights + bias)
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # Apply the logistic function to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input): #  definizione di SIGMOIDE *****
        return 1 / (1 + math.exp(-total_net_input)) 

    def calculate_error(self, target_output):
    	# TODO: compute the loss. For this task, which loss is appropriated?
        # ((target_output - self.output)**2)*0.5
    	#raise NotImplementedError("Compute the loss here!")
        return ((target_output - self.output)**2)*0.5

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input()

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_total_net_input_wrt_input(self):
        # TODO: implement
    	#raise NotImplementedError()
        # !!! posso applicare questa semplificazione perchè è la derivata della funzione
        # SIGMOIDE utilizzata come funzione di attivazione 
        return self.output(1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

##############################################################################
# definizione di un livello di neuroni di dimensione: num_neurons
# contiene un array di oggetti Neuron inizializzati bias di valore random
# ha inspect: che visualizza tot neuroni e pesi correlati
# feed_forward: che calcola il valore di uscita della funzione di attivazione
# get_outputs: ti restitusice il valore della funzione di attivazione
#############################################################################
class NeuronLayer:
    def __init__(self, num_neurons, bias):# salva e crea tutti i neuroni in questo livello

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random() # inizialmente sono dei valori casuali di bias
        self.neurons = []

        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self): # visualizza pesi e bias del neurone 
        print('Neurons in this layer:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs): # calcola il valore d'uscita di ogni neurone
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self): # visualizza gli output di adesso
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, 
                hidden_layer_weights = None, hidden_layer_bias = None, 
                output_layer_weights = None, output_layer_bias = None):
        
        # solo 3 livelli questa piccola NN
        self.num_inputs = num_inputs    # input layer
                                        # 2 neuroni    1 bias per livello
        # qua vengono creati sia i neuroni che i livelli
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)  # hidden layer
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias) # output layer
        
        # inizializza i vari pesi
        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    # inizializzi pesi da input ad hidden layer
    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)): # per ogni neurone di questo livello
            for i in range(self.num_inputs): # grazie perchè conosco la struttura interna e tutto....
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random()) # non può esistere un peso a 0!!
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1
    

    # inizializzi pesi da hidden layer ad output
    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self): # sai che la struttura della tua rete interna è 1 livello input, 1 nascosto ed 1 output
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    # aggiornamento dell'output dei neuroni presenti nell'hidden layer e dei neuroni presenti nel livello d'uscita
    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        # a list of 1 element and repeat it a len() number of times! come vect[n] tipo...
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error




# Train --> NEURAL NETWORK ANALIZZATA A LEZIONE 19/10/21
# self, num_inputs, num_hidden, num_outputs, --> proprio il numero di neuroni attesi per ogni livello
# hidden_layer_weights = None, --> ricrei te la stessa NN che è stata presentata a lezione
# hidden_layer_bias = None, 
# output_layer_weights = None, 
# output_layer_bias = None):

nn = NeuralNetwork( 2, 2, 2, 
                    hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], 
                    hidden_layer_bias=0.35, 
                    output_layer_weights=[0.4, 0.45, 0.5, 0.55], 
                    output_layer_bias=0.6
                )

for i in range(10000):
    # def train(self, training_inputs, training_outputs) il valore degli inputs dei 2 neuroni input ed
    # il valore di output target dei due neuroni d'uscita 
    nn.train([0.05, 0.1], [0.01, 0.99]) # nel train hai solo la definizione dei pesi di input ed output
    if i % 100 == 0: # ogni 100 aggiornamenti è passata un'epoca
        print("Epoch:", i, " Loss: ", round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

# TODO: EXERCISE: Train a network that is capable of doing the XOR

