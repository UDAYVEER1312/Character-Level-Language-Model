# Character-Level-Language-Model
This mini model is simply built using Python `Numpy`

I shared the python `.ipynb` file in this repo along with the explanation of major steps in training as well as testing.

# Set Up
Just make sure you have a working python environment and install Numpy because things are aboout to get exited, which were for me as well when i first read about such model.

# Dataset
Create a txt file with a poem or a phrase that you want your model to memorize, for example in my case i used the poem  *"Twinkle Twinkle little star, how i wonder what you are, up above the worlds so high, like a diamond in teh sky."* Sounds good right?

# Extracting data vocalubary:
In the below section i extracted data from my file 'data_cllm1.txt'
```bash
import numpy as np
file_data = open('data_cllm1.txt','r').read()
data = list(file_data)
vocab = set(file_data)
data_size , vocab_size = len(data) , len(vocab)
print('length of data : ',data_size)
print('Vocabulary size : ',vocab_size)
print('Vocabulary : ',vocab)
```
---output---
```bash
length of data :  130
Vocabulary size :  24
Vocabulary :  {'e', 'p', 'u', 'd', 's', 'h', 'T', 'a', '.', 't', 'm', 'g', 'v', 'w', 'b', ',', 'l', 'k', 'i', 'y', 'n', 'r', 'o', ' '}
```
# Structure of RNN:
Here i used only 30 hidden neurons, of course you can add more but there's a little check in it. Since my data is so small and has a vocabulary size of 24 characters only, then one thing i need to take note in mind is that when there's small data then the sequence length of the input gets lower in order to increase the learning of the RNN, i.e. RNN can memorize the sequence fastly.
(Note that this model only showcase the power of RNN and is bad at memorizing relationships between charcters if we add more data)
```bash
hidden_neurons = 30
sequence_length = 5 # small sequence input
# hyperparameters : Normalization
w_xh = np.random.randn(hidden_neurons,vocab_size) * 0.01 # input to hidden neurons
w_hh = np.random.randn(hidden_neurons,hidden_neurons) * 0.01 # hidden to hidden
w_hy = np.random.randn(vocab_size,hidden_neurons) * 0.01 # hidden to output
b_h = np.zeros((hidden_neurons,1))
b_y = np.zeros((vocab_size,1))
```

# Feeding In and Calculatin loss:
First we need to create some structres to store the states of input layer, hidden layer and outout layer along with the probabilities at each time stamp (after each sequence input)
```bash
def feed_and_loss(inputs,targets,h_previous):
    x_states = {}
    h_states = {}
    y_states = {}
    prob_logits = {}
    h_states[-1] = h_previous  # initial hidden state before time stamp = 1
```
Actual Feeding in sequence:
```bash
    for t in range(len(inputs)):
        x_states[t] = np.zeros((vocab_size,1))
        x_states[t][inputs[t]] = 1
        h_states[t] = np.tanh(np.dot(w_xh,x_states[t]) + np.dot(w_hh,h_states[t-1]) + b_h)
        y_states[t] = np.dot(w_hy,h_states[t]) + b_y
        # probability distributions of the output :  softmax
        prob_logits[t] = np.exp(y_states[t])/np.sum(np.exp(y_states[t]))
        # summing up losses at each time stamp
        # cross entropy loss
        loss += -np.log(prob_logits[t][targets[t],0])

        # computing loss
    dw_xh = np.zeros_like(w_xh)
    dw_hh = np.zeros_like(w_hh)
    dw_hy = np.zeros_like(w_hy)
    db_h = np.zeros_like(b_h)
    db_y = np.zeros_like(b_y)
    # For the loss wrt hidden state also influence the future hidden state 
    dhnext = np.zeros_like(b_h) 
```
Backpropagate and calculate total loss with respect to each parameter.
I know the code will look a bit difficult but below i also commented down some explanation for an ease.
You can literally differentiate the error in probabilities with each paremeter to find the total loss and you'll find that the loss calculated by each parametre is just the differentiation we'll get.
For example, let L be the total loss then `dL/dw_hy` (Gradient of loss with respect to the hiden to output weight matrix) = `dL/dy` * `dy/dw_hy` (using the chian rule of differentiation).
Since y = `w_hy`*`hidden_state[t]` + `b_y` then dy/dw_hy would simply be hidden_state at time stamp t. 

But wait... since i am calculating the gradients, what if the gradients become so large or small? Well it can badly affect the model learning.
Surely you've herad of the Vanishing graidnet problem, which is most common but yet bad for a model. That is taken care as well, if you notice the end lines of this snipet.
```bash
    for t in reversed(range(len(inputs))):
        # calculating loss wrt output
        # dy = dL/dy
        dy = np.copy(prob_logits[t])  # same as the probabilities if not target class
        dy[targets[t]] -= 1 # Loss = probability - 1 if target class  
        dw_hy += np.dot(dy,h_states[t].T) # dL/dwhy = dL/dy * dy/dwhy
        db_y += dy                 # dL/dby = dL/dy * dy/dby
        # calculating loss wrt hidden state
        # dh(t) = dL/dy * dy/dh  + gradient wrt the future time stamps 
        dh = np.dot(w_hy.T,dy) + dhnext
        # backpropagating in the tanh non-linearity in hidden states
        # dh(z)/dp = (1 - h(z)^2) * dz/dp  where z is the input to the hidden state
        dhraw = (1 - h_states[t]**2) * dh
        db_h += dhraw
        dw_hh += np.dot(dhraw,h_states[t-1].T)
        dw_xh += np.dot(dhraw,x_states[t].T)
        # updating the gradients wrt the future hidden states
        # dL/dh(t+1)
        dhnext += np.dot(w_hh.T,dhraw)
    # clipping to prevent exploding gradients
    for dpara in [dw_xh,dw_hh,dw_hy,db_h,db_y]:
        np.clip(dpara,-5,5,out=dpara)
    return loss,dw_xh,dw_hh,dw_hy,db_h,db_y,h_states[len(inputs)-1]  # last hidden states the sequence input
```
# Training
For training, the model uses Adagrad Optimization technique. 
I trained over 65000 iterations (65000 times sequence is passed and evaluated)
For thatt you can look into this [file](https://github.com/UDAYVEER1312/Character-Level-Language-Model/blob/main/Vanilla_RNN_character_level_language_model.ipynb).
ALso if you look at each iteration you can see the model is really learning the sequence that we passed, for example this is the sequence generated after 1000 iteration :
```bash
--iter--1000		--loss--9.261540183866181
---
 Twinkle littt.ondee why............................ ---
```
And this is after 50000 iteration:
```bash
--iter--50000		--loss--0.015352855290855836
Twinkle twinkle little star, how i wonder what you  ---
```
