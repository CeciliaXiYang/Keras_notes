# Concerned Questions of Keras

## ** [Keras Github Docs](https://github.com/fchollet/keras)**

## Mixed length of sequences:

### 1) How to solve mixed lengths of sequences?

a) Pad & truncate sequences to be the same length: In sequence preprocessing step, conduct **pad_sequences**. 
  
   - Converted num_timesteps either indicated by **max_len** argument, or the length of longest sequence. 
  
   - Position of padding or truncting is indicted like: **padding='pre'**, **truncating='post'**
  
   - **value=0.** in argument indicates the value to be padded. (or pad with neutual data)


b) Group sequences into batches by length, i.e. same length of sequences grouped into same batch.


c) Set batch size as 1. [Recurrent Models with sequences of mixed length](https://github.com/fchollet/keras/issues/40)


d) Use a masking layer

   - Q: masking layer among different batch??

e) Set sample_weight parameter


### 2) Difference between <font style="color:green">*Masking layer*</font> and <font style="color: green">*Sample_weights*</font>?

**Sample_weight is a hand defined mask.** [Using Masking Layer for Sequence to Sequence Learning](https://github.com/fchollet/keras/issues/957)

a) Sample_weight:

   - List or numpy array with 1:1 mapping to the training samples, used for scaling the loss function during training only.
  
   - For time-distributed data, there is one weight per sample **per timestep**, i.e. if output data is shaped (nb_sample, timesteps, output_dim), the mask should be of shape (nb_sample, timesteps). 
  
   - **sampe_weight allows to mask out or reweight individual output timesteps in sequence to sequence learning.** 
   [sample_weight docs](https://github.com/fchollet/keras/pull/494/commits/73fdaf6d6f8cd4de98db79ae93638d300b8de2b5)
  
   - Need to specify **sample_weight_mode="temporal"** in compile().
  
   - For validation data, passed as a part of validation_data tuple:
  
  [Set sample_weight in validation](https://github.com/fchollet/keras/issues/496)
  
  [Optionally mask cost function for sequence to sequence learning](https://github.com/fchollet/keras/pull/451)
  
  [Is the sequence to sequence learning right?](https://github.com/fchollet/keras/issues/395)
  
b) Masking layer:


References:

[Does masking only work for homogeneous batches?](https://github.com/fchollet/keras/issues/1206)

[How does Masking work?](https://github.com/fchollet/keras/issues/3086)

## Many to many rnn

1) **Set return_sequence=True, and then make sure to warp Dense with a TimeDistributed wrapper layer.**

2) Return_sequence

   - return_sequence=True: Information transfered not only to the next layer, but also to the next timestep.

   - return_sequence=False: For input 0 to seq_len-2, the prediction only passed to the layer itself for the next timestep and not as input to the next layer. Only the seq_len-1 input is passed forward to the dense layer for the loss computation against the target.

References:

[**Andrej Karpathy blog**: The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[LSTM many to many mapping problem](https://github.com/fchollet/keras/issues/2403)

** A very detailed explaination**: [Keras recurrent tutorial](https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent)

## Motivation of using batch_size?

### 1) Concepts of main parameters:

1) **Epoch**: One forward pass and backward pass of **ALL** training samples.

2) **Batch_size**: The number of training samples in one forward and backward pass. The larger batch size, the more memory required.

3) **Iteration**: Number of passes for passing batch size of samples.

### 2) Advantage and disadvantage of using batch_size?

1) Advantage: 

   - Less memory required.
   - Trains faster with mini_batch ?? (Update paramter after each Iteration)

2) Disadvantage:
   
   The smaller the batch size, the less of the accuracy. (Model tend to be stochestic)

Reference:

[What is batch size in neural network?](https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network)

## Optimization methods in keras?

## What does stateful or not stateful mean in rnn? 

If argument **stateful=True**, the last state for each sample at index **i** in a batch will be utilized as initial state for the sample of index **i** in the following batch. 

## Prediction resuslts of rnn

1) evaluate: returns list of scalars/metrics, with the metrics indicted by **`model.metrics_names`**.

2) predict

3) predict_class

Reference:

[**Brownlee Bolg**: Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)
