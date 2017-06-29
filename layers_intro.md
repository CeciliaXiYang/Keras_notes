# Key Layers in Keras

## ** [Keras Manual Book](https://keras.io/layers/about-keras-layers/) **

## TimeDistributed Wrapper:

1) Apply a layer to every temporal slice of an input. The input should be at least 3D. 

-- If return_sequence = True and TimeDistributed(Dense(1)): utilized for many-to-many or one-to-many applications.

-- If return_sequence = False and without TimeDistributed wrapper, the output is only for the last time step. 

2) **TimeDistributed can be used together with arbitrary layers, not limited to Dense.**

Reference:

[**Brownlee Bolg**: How to Use the TimeDistributed Layer for Long Short-Term Memory Networks in Python](http://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/)

[When and How to use TimeDistributedDense](https://github.com/fchollet/keras/issues/1029)

[Difference between Dense and TimeDistributedDense](https://github.com/fchollet/keras/issues/2038)

## Dense Layer & Dropout Layer:

1) Dense Layer: fully connected, i.e. each unit/neuron is connected to each neuron in the next layer.

2) Dropout Layer: regularization to avoid overfitting. 

-- Randomly deactive certain units/neurons in a layer with a certain probability from a Bernoulli distribution. 
-- Would speed up the training procedure. 
-- Only applied in training, and need to rescale the remaining neuron activations. 
-- Use the complete network for testing, i.e. set the dropout probability as 0. 
-- Keras dropout layer could do above things automatically. 

Reference:

[What is a Dense and a Dropout layer](https://www.quora.com/In-Keras-what-is-a-dense-and-a-dropout-layer)

## Embedding Layer: 

1) Turns **positive integers** (indexes) into dense vectors of fixed size, e.g. Word2Vec. **Limited to integer sequences with finite range.** Essentially, it's a matrix multiplication: (nb_words, vocab_size) * (vocab_size, embedding_dim)

2) **This layer can only be used as the first layer in a model.** 

3) Input shape: (batch_size, sequence_len); Output shape: (batch_size, sequence_len, output_dim)

4) Argument **mask_zero**: Whether or not input value of 0 is a special "padding" value that should be masked out. **This is useful when using recurrent layers which may take variable length input.** If mask_zero=True, then all subsequent layers in the model need to support masking or an exception will be raised, and index 0 cannot be used in the vocabulary.  

Reference:

[Keras doc for embedding](https://github.com/fchollet/keras/blob/master/keras/layers/embeddings.py#L22)

## Masking Layer:

1) Masks a sequence by using a mask value to skip timesteps.

2) For each timestep in the input tensor, if all values in the input tensor at that timestep are equal to **mask_value**, then the timestep will be masked in all downstream layers.

3) If any downstream layer does not support masking yet receives such an input mask, an exception will be raised.
