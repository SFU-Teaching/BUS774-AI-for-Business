# LSTM Model Structures for Time Series Prediction

BUS 888 ML for Finance

C Kaligotla

23 Jun 2024


## LSTM Models

### 1. Basic LSTM Model

```python
model = Sequential([
    LSTM(50, input_shape=(sequence_length, features)),
    Dense(1)
])

```
This simple LSTM model has a single LSTM layer followed by a Dense layer for output. 
It's suitable for basic sequence prediction tasks.

### 2. Stacked LSTM Model

```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, features)),
    LSTM(50),
    Dense(1)
])
```
This model uses two LSTM layers stacked on top of each other. The first LSTM layer returns sequences for the second LSTM layer to process, helping to learn more complex patterns in the data.

### 3. Bidirectional LSTM

```python
from keras.layers import Bidirectional

model = Sequential([
    Bidirectional(LSTM(50, input_shape=(sequence_length, features))),
    Dense(1)
])

```
A bidirectional LSTM processes the input sequence both forwards and backwards. This can be helpful when the context of the entire sequence is important for prediction.


### 4. LSTM with Dropout

```python
from keras.layers import Dropout

model = Sequential([
    LSTM(50, input_shape=(sequence_length, features), return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
```
This model includes dropout layers, which help prevent overfitting by randomly setting a fraction of input units to 0 during training.

### 5. LSTM with Time Distributed Dense Layer

```python
from keras.layers import TimeDistributed

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, features)),
    TimeDistributed(Dense(1))
])
```
This model uses a TimeDistributed wrapper to apply a Dense layer to every temporal slice of an input. It's useful when you want to output a prediction for each time step.

### 6. CNN-LSTM Model

```python
from keras.layers import Conv1D, MaxPooling1D

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, features)),
    MaxPooling1D(pool_size=2),
    LSTM(50),
    Dense(1)
])
```
This model combines a Convolutional Neural Network (CNN) layer with an LSTM. The CNN layer can help in extracting features from the input sequence before feeding it into the LSTM.


### 7. Attention-based LSTM

```python
from keras.layers import Attention, concatenate, Input
from keras.models import Model

inputs = Input(shape=(sequence_length, features))
lstm = LSTM(50, return_sequences=True)(inputs)
attention = Attention()([lstm, lstm])
concat = concatenate([lstm, attention])
output = Dense(1)(concat)

model = Model(inputs=inputs, outputs=output)
```
This model incorporates an attention mechanism, which allows the model to focus on different parts of the input sequence when making predictions.



## Considerations for Choosing a Model Structure

When selecting between these structures, consider:

1. The complexity of the problem and the amount of data available.

2. The computational resources at their disposal.

3. The specific characteristics of their time series data.

4. The trade-off between model complexity and potential overfitting.


**Remember to experiment with these different structures and to monitor both training and validation performance to find the best model for their specific task**