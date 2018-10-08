
# Introduction to Deep Learning
## Part 1 - Linear Perceptron


```python
import matplotlib.pyplot as plt
import numpy as np
import utils_plot as uplot
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, log_loss
%matplotlib inline  
```

## Inspiration - The Biological Neuron Model
**The Neuron cell takes a binary input (of electrical signals), process it, and produce binary output (of electrical signal).**<br>

![title](https://raw.githubusercontent.com/naomifridman/Introduction_to_deep_learning/master/imgs/neuron3.PNG)

**Neuron Input**<br>
A neuron accepts inputs, usually from other neurons, through its dendrites, that are connect to other neurons via a gap called a synapse which assigns a "weight" to any specific input. <br>
All of these inputs are processed together in the cell body (soma)<br>

**Neuron Output**<br>
The Neurons process the combination of the inputs, and if it exceeds a certain threshold, an output signal is produced (the neuron "fires")<br>
When the neuron fires, the signal (output) travels along the axon to the synapse terminals,there it assigned a "weight", and continue to the dendrites of other neurons through the synapse.<br>
With this biological model in mind, we can define a mathematical artificial neuron.

## Perceptron - An Artificial Neural Network
As in the Biological model, we want the neuron to intake an inputs, process it (weighted sum) and if it exceeds a threshold, the neuron will produce an output. 

![title](https://raw.githubusercontent.com/naomifridman/Introduction_to_deep_learning/master/imgs/perceptron_paradigma3.png)


## Decision Units - Activation functions
There are many options for decision units, and we will see later. Lets start with the basic one, defined by McCulloch-Pitts (1943)
### Binary threshold unit
* Compute weighted sum and compare it to threshold:
$$Is \sum_{i} x_{i}w_{i} >  Threshold ?$$ 
* Equivalently, we can define a bias to add to the weighted input sum: 
$$z=b+\sum_{i} x_{i}w_{i} > 0 ?$$
* Then output is 1 if weighted sum larger then zero.
$$y = \begin{cases} 1 & z >  0 \\
                       0 & otherwise \\
                     \end{cases}$$



```python
uplot.drow_binary_threshold()
```


![png](output_6_0.png)


## Learning the Bias
Now we need to learn the weights and the bias. But we can use a trick, toavoid separate scheme, for learning the bias. 
* We can add 1 to all input vectors: <br>
For X = $(X_{0}, X_{1}, ...Xn)$ Lets define:  $\bar{X}$ = $(1, X_{0}, X_{1}, ...Xn)$ and $\bar{W}$ = $(1,W)$<br>
=> $$z=b+\sum_{i} x_{i}w_{i} = \sum_{i} \bar{x}_{i}\bar{w}_{i}$$
> This way, we can learn the bias as it was part of the weights. 

## Learning weights
The idea is to go over the samples, and correct/update the weights according to the results on the samples.
![title](https://raw.githubusercontent.com/naomifridman/Introduction_to_deep_learning/master/imgs/dl3.jpg)
#### Update the weights
There are many strategies to update the weights, and we will see them later.
### Perpetron Example - Logic OR
Lets build a Perceptron, that performs a logic OR:

X1 | X2 | OR
--- | --- | ---
*0* | 0 | **0**
*1* | 0 | **1**
*0* | 1 | **1**
*1* | 1 | **1**
* The scheme will be to Go over all training cases in any order, and:
    * If the output is correct, continue
    * If output < lable  ("increas W"): $\bar{W} = \bar{W} + \bar{X}_{i}$
    * If output > lable ("decreas W"): $\bar{W} = \bar{W} - \bar{X}_{i}$


```python
def binary_threshold_activation(Xw):
    yhat = 0
    if(Xw > 0):
        yhat = 1
    return yhat
```


```python
def predict(X, w):
    yhat = 0
    z = np.dot(X, w)
    if(z > 0):
        yhat = 1
    return yhat
```


```python
def perceptron_train(X, Y):

    w = np.zeros(len(X[0]))
    n = 5
    errors = []
    
    for t in range(n):
        
        print('--------------------------\nepoch: ' , t)
        total_error = 0
        
        for i in range(X.shape[0]):
             
            z = np.dot(X[i], w)
            yhat = binary_threshold_activation(z)
            
            if (yhat == Y[i]):
                print(i,'x:',X[i],'w:', w,'Wx: ', np.dot(X[i], w),
                      'y:',Y[i], end=' ')
                print('correct no update')
                continue
                
            elif (yhat < Y[i]):
                total_error += 1
                print(i, 'yhat<y', 'X:',X[i],'w:', w,'Wx:',
                      np.dot(X[i], w), 'y:',Y[i], end=' ')
        
                
                w = w + X[i]
                print('=> W updated:', w)
                
            elif (yhat > Y[i]):
                total_error += 1
                print(i, 'yhat>y' ,'X:',X[i],'w:', w,'Wx:', 
                      np.dot(X[i], w), 'y:',Y[i] ,end = ' ')
                
                w = w - X[i]
                print('=> W updated:', w)
                
        errors.append(total_error)
        
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    
    return w
```


```python
X = [[1,0,0],
    [1,1,0],
    [1,0,1],
    [1,1,1]]

y = [0, 1, 1, 1]

w=perceptron_train(np.array(X),y)
print('Leared Weights:', w)

```

    --------------------------
    epoch:  0
    0 x: [1 0 0] w: [0. 0. 0.] Wx:  0.0 y: 0 correct no update
    1 yhat<y X: [1 1 0] w: [0. 0. 0.] Wx: 0.0 y: 1 => W updated: [1. 1. 0.]
    2 x: [1 0 1] w: [1. 1. 0.] Wx:  1.0 y: 1 correct no update
    3 x: [1 1 1] w: [1. 1. 0.] Wx:  2.0 y: 1 correct no update
    --------------------------
    epoch:  1
    0 yhat>y X: [1 0 0] w: [1. 1. 0.] Wx: 1.0 y: 0 => W updated: [0. 1. 0.]
    1 x: [1 1 0] w: [0. 1. 0.] Wx:  1.0 y: 1 correct no update
    2 yhat<y X: [1 0 1] w: [0. 1. 0.] Wx: 0.0 y: 1 => W updated: [1. 1. 1.]
    3 x: [1 1 1] w: [1. 1. 1.] Wx:  3.0 y: 1 correct no update
    --------------------------
    epoch:  2
    0 yhat>y X: [1 0 0] w: [1. 1. 1.] Wx: 1.0 y: 0 => W updated: [0. 1. 1.]
    1 x: [1 1 0] w: [0. 1. 1.] Wx:  1.0 y: 1 correct no update
    2 x: [1 0 1] w: [0. 1. 1.] Wx:  1.0 y: 1 correct no update
    3 x: [1 1 1] w: [0. 1. 1.] Wx:  2.0 y: 1 correct no update
    --------------------------
    epoch:  3
    0 x: [1 0 0] w: [0. 1. 1.] Wx:  0.0 y: 0 correct no update
    1 x: [1 1 0] w: [0. 1. 1.] Wx:  1.0 y: 1 correct no update
    2 x: [1 0 1] w: [0. 1. 1.] Wx:  1.0 y: 1 correct no update
    3 x: [1 1 1] w: [0. 1. 1.] Wx:  2.0 y: 1 correct no update
    --------------------------
    epoch:  4
    0 x: [1 0 0] w: [0. 1. 1.] Wx:  0.0 y: 0 correct no update
    1 x: [1 1 0] w: [0. 1. 1.] Wx:  1.0 y: 1 correct no update
    2 x: [1 0 1] w: [0. 1. 1.] Wx:  1.0 y: 1 correct no update
    3 x: [1 1 1] w: [0. 1. 1.] Wx:  2.0 y: 1 correct no update
    Leared Weights: [0. 1. 1.]
    


![png](output_12_1.png)


### Predict OR with The learned weights of the Perceptron


```python
for i in range(4):
    print(X[i], X[i][1],' OR ', X[i][2],' = ', predict(X[i],w))

```

    [1, 0, 0] 0  OR  0  =  0
    [1, 1, 0] 1  OR  0  =  1
    [1, 0, 1] 0  OR  1  =  1
    [1, 1, 1] 1  OR  1  =  1
    

### Re-train Percepron to Preform logic AND
Same perceptron,can learn to perform logic AND, When its trained on the correct data:

X1 | X2 | AND
--- | --- | ---
*0* | 0 | **0**
*1* | 0 | **0**
*0* | 1 | **0**
*1* | 1 | **1**


```python
X = [[1,0,0],
    [1,1,0],
    [1,0,1],
    [1,1,1]]

y = [0, 0, 0, 1]

w = perceptron_train(np.array(X),y)
print('Leared Weights:', w)

```

    --------------------------
    epoch:  0
    0 x: [1 0 0] w: [0. 0. 0.] Wx:  0.0 y: 0 correct no update
    1 x: [1 1 0] w: [0. 0. 0.] Wx:  0.0 y: 0 correct no update
    2 x: [1 0 1] w: [0. 0. 0.] Wx:  0.0 y: 0 correct no update
    3 yhat<y X: [1 1 1] w: [0. 0. 0.] Wx: 0.0 y: 1 => W updated: [1. 1. 1.]
    --------------------------
    epoch:  1
    0 yhat>y X: [1 0 0] w: [1. 1. 1.] Wx: 1.0 y: 0 => W updated: [0. 1. 1.]
    1 yhat>y X: [1 1 0] w: [0. 1. 1.] Wx: 1.0 y: 0 => W updated: [-1.  0.  1.]
    2 x: [1 0 1] w: [-1.  0.  1.] Wx:  0.0 y: 0 correct no update
    3 yhat<y X: [1 1 1] w: [-1.  0.  1.] Wx: 0.0 y: 1 => W updated: [0. 1. 2.]
    --------------------------
    epoch:  2
    0 x: [1 0 0] w: [0. 1. 2.] Wx:  0.0 y: 0 correct no update
    1 yhat>y X: [1 1 0] w: [0. 1. 2.] Wx: 1.0 y: 0 => W updated: [-1.  0.  2.]
    2 yhat>y X: [1 0 1] w: [-1.  0.  2.] Wx: 1.0 y: 0 => W updated: [-2.  0.  1.]
    3 yhat<y X: [1 1 1] w: [-2.  0.  1.] Wx: -1.0 y: 1 => W updated: [-1.  1.  2.]
    --------------------------
    epoch:  3
    0 x: [1 0 0] w: [-1.  1.  2.] Wx:  -1.0 y: 0 correct no update
    1 x: [1 1 0] w: [-1.  1.  2.] Wx:  0.0 y: 0 correct no update
    2 yhat>y X: [1 0 1] w: [-1.  1.  2.] Wx: 1.0 y: 0 => W updated: [-2.  1.  1.]
    3 yhat<y X: [1 1 1] w: [-2.  1.  1.] Wx: 0.0 y: 1 => W updated: [-1.  2.  2.]
    --------------------------
    epoch:  4
    0 x: [1 0 0] w: [-1.  2.  2.] Wx:  -1.0 y: 0 correct no update
    1 yhat>y X: [1 1 0] w: [-1.  2.  2.] Wx: 1.0 y: 0 => W updated: [-2.  1.  2.]
    2 x: [1 0 1] w: [-2.  1.  2.] Wx:  0.0 y: 0 correct no update
    3 x: [1 1 1] w: [-2.  1.  2.] Wx:  1.0 y: 1 correct no update
    Leared Weights: [-2.  1.  2.]
    


![png](output_16_1.png)


### Predict AND with The learned weights of the Perceptron


```python
for i in range(4):
    print(X[i], X[i][1],' OR ', X[i][2],' = ', predict(X[i],w))

```

    [1, 0, 0] 0  OR  0  =  0
    [1, 1, 0] 1  OR  0  =  0
    [1, 0, 1] 0  OR  1  =  0
    [1, 1, 1] 1  OR  1  =  1
    
