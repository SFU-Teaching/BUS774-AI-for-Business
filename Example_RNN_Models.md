# RNN Model Structures for Time Series Prediction
BUS 888 ML for Finance

C Kaligotla

23 Jun 2024

## RNN Models

### 1. Basic RNN Model
```python
model = Sequential([
    SimpleRNN(50, input_shape=(sequence_length, features)),
    Dense(1)
])
```
This is a simple RNN model with a single SimpleRNN layer followed by a Dense layer for output. It's suitable for basic sequence prediction tasks where long-term dependencies are not crucial.

### 2. Stacked RNN Model
```python
model = Sequential([
    SimpleRNN(50, return_sequences=True, input_shape=(sequence_length, features)),
    SimpleRNN(50),
    Dense(1)
])
```
This model uses two SimpleRNN layers stacked on top of each other. The first RNN layer returns sequences for the second RNN layer to process, potentially capturing more complex temporal patterns.

### 3. Bidirectional RNN
```python
from keras.layers import Bidirectional

model = Sequential([
    Bidirectional(SimpleRNN(50, input_shape=(sequence_length, features))),
    Dense(1)
])
```
A bidirectional RNN processes the input sequence both forwards and backwards. This can be beneficial when both past and future context in the sequence are important for prediction.

### 4. RNN with Dropout
```python
from keras.layers import Dropout

model = Sequential([
    SimpleRNN(50, input_shape=(sequence_length, features), return_sequences=True),
    Dropout(0.2),
    SimpleRNN(50),
    Dropout(0.2),
    Dense(1)
])
```
This model includes dropout layers to help prevent overfitting. Dropout randomly sets a fraction of input units to 0 during training, which can improve generalization.


### 5. RNN with Time Distributed Dense Layer
```python
from keras.layers import TimeDistributed

model = Sequential([
    SimpleRNN(50, return_sequences=True, input_shape=(sequence_length, features)),
    TimeDistributed(Dense(1))
])
```
This model uses a TimeDistributed wrapper to apply a Dense layer to every temporal slice of the RNN output. It's useful when you want to generate predictions for each time step in the sequence.


### 6. CNN-RNN Model
```python
from keras.layers import Conv1D, MaxPooling1D

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', 
           input_shape=(sequence_length, features)),
    MaxPooling1D(pool_size=2),
    SimpleRNN(50),
    Dense(1)
])
```
This model combines a Convolutional Neural Network (CNN) layer with an RNN. The CNN layer can help in extracting local features from the input sequence before feeding it into the RNN.


### 7. Attention-based RNN
```python
from keras.layers import Attention, concatenate, Input
from keras.models import Model

inputs = Input(shape=(sequence_length, features))
rnn = SimpleRNN(50, return_sequences=True)(inputs)
attention = Attention()([rnn, rnn])
concat = concatenate([rnn, attention])
output = Dense(1)(concat)

model = Model(inputs=inputs, outputs=output)
```
This model incorporates an attention mechanism, allowing the model to focus on different parts of the input sequence when making predictions. However, attention is less commonly used with simple RNNs compared to LSTMs or GRUs.


## Considerations for Choosing a Model Structure

When selecting between these structures, consider:

1. The nature of the temporal dependencies in their data (short-term vs. long-term).

2. The amount of training data available and the risk of overfitting.

3. The computational resources at their disposal (RNNs are generally less computationally intensive than LSTMs).

4. The specific requirements of their prediction task (e.g., predicting for each time step vs. a single future value).

It's important to note that while RNNs can be effective for some sequence prediction tasks, they often struggle with learning long-term dependencies due to the vanishing gradient problem. For tasks requiring long-term memory, LSTM GRU models are often preferred.

**Remember to experiment with these different structures and to monitor both training and validation performance to find the best model for their specific task**