# Keras : Chapter 2
---


**텐서 : Tensor**
텐서플로우에서 상수 텐서에는 값을 할당할 수 없다. 따라서 변수 텐서를 사용한다.


```python
#!pip install tensorflow
import tensorflow as tf

# 변수 텐서
v = tf.Variable(initial_value=tf.random.normal(shape=(3,1)))
print(v)
v.assign(tf.ones((3,1)))
print(v)
```

    <tf.Variable 'Variable:0' shape=(3, 1) dtype=float32, numpy=
    array([[-1.020172 ],
           [ 0.6342499],
           [-1.4028376]], dtype=float32)>
    <tf.Variable 'Variable:0' shape=(3, 1) dtype=float32, numpy=
    array([[1.],
           [1.],
           [1.]], dtype=float32)>
    


**역전파 알고리즘 : Back Propagation**
역전파 알고리즘은 신경망의 그래디언트 값을 계산하는 데 미적분의 연쇄 법칙을 적용하는 것.
연쇄 법칙을 계산 그래프에 적용한 것이다.

- Tensorflow 의 Gradient Tape(Computational Graph) API



```python
import tensorflow as tf

x = tf.Variable(0,) #초깃값 0으로 스칼라 변수생성
with tf.GradientTape() as tape: #Gradient tape 블록 생성
    y = 2*x + 3 #텐서 연산
    grad_of_y_yrt_x = tape.gradient(y, x) #변수 x에 대한 출력 y의 그래디언트 계산
    
    
#다차원 텐서의 그래디언트
x = tf.Variable(tf.zeros(2,2)) #(2,2) 크기의 0으로 채워진 변수생성
with tf.GradientTape() as tape: #Gradient tape 블록 생성
    y = 2*x + 3 #텐서 연산
    grad_of_y_yrt_x = tape.gradient(y, x)
```

grad_of_y_yrt_x 는 x와 크기가 같은 (2,2) 크기의 텐서이다. x = [[0,0],[0,0]] 일 때, y=2*x+3의 기울기(곡률)을 나타낸다.


```python
#변수 리스트의 그래디언트
W = tf.Variable(tf.random.uniform((2,2)))
b = tf.Variable(tf.zeros((2,)))
x = tf.random.uniform((2,2))
with tf.GradientTape() as tape:
    y = tf.matmul(x,W) + b #점곱 함수.
grad_of_y_yrt_W_and_b = tape.gradient(y, [W, b])
```

grad_of_y_yrt_x 는 2개의 텐서를 담은 리스트. 각 텐서는 W, b와 크기가 같다.

## MNIST 예제 : Keras


```python
#tensorflow.keras.layers에서 Dense 불러오기
#데이터 입력
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터셋 로드 및 전처리
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# 모델 정의
model = Sequential([
    Dense(512, activation="relu"),
    Dense(10, activation="softmax") #layers,Dense 가 아닌 Dense 로 사용
])

# 모델 컴파일
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 모델 학습
model.fit(train_images, train_labels, epochs=5, batch_size=128)
```

    Epoch 1/5
    [1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.8737 - loss: 0.4401
    Epoch 2/5
    [1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 3ms/step - accuracy: 0.9662 - loss: 0.1177
    Epoch 3/5
    [1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9797 - loss: 0.0700
    Epoch 4/5
    [1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.9852 - loss: 0.0503
    Epoch 5/5
    [1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.9894 - loss: 0.0363
    




    <keras.src.callbacks.history.History at 0x12237b0ad30>




```python
#tensorflow.keras 에서 layers 불러오기
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

#입력 데이터 : 데이터 타입 float32, 훈련 데이터 크기는 (60000, 784) 크기 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

#모델 : 2개의 Dense 층이 연결되어 있고, 각 층은 가중치 텐서를 포함해 입력 데이터에 대한 텐서 연산을 적용한다.
#가중치 텐서는 모델이 정보를 저장하는 곳
model = keras.Sequential([
    layers.Dense(512, activation="relu"), 
    layers.Dense(10, activation="softmax")
])

# 모델 컴파일 : 모델학습에 필요한 필수적인 요소 정의 및 설정
model.compile(optimizer="rmsprop", #손실함수를 줄이기위해 가중치를 어떻게 조절할지
              loss="sparse_categorical_crossentropy",#모델 학습 예측값과 실제 값 차이를 측정
              metrics=["accuracy"])#분류 문제이므로 정확도를 사용해 성능확인

#모델 학습 : 훈련 반복
model.fit(train_images, train_labels, epochs=5, batch_size=128)
```

    Epoch 1/5
    [1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 3ms/step - accuracy: 0.8739 - loss: 0.4450
    Epoch 2/5
    [1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 3ms/step - accuracy: 0.9680 - loss: 0.1141
    Epoch 3/5
    [1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 3ms/step - accuracy: 0.9788 - loss: 0.0693
    Epoch 4/5
    [1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9853 - loss: 0.0532
    Epoch 5/5
    [1m469/469[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9890 - loss: 0.0358
    




    <keras.src.callbacks.history.History at 0x1223ba7e160>



# MNIST 예제 : Tensorflow로 구현하기
---

- 1. 단순한 Dense 클래스
Dense 층은 입력 변환을 구현합니다.


모델 파라미터 : W(Weight), b(bias)

Activation : 각 원소에 적용되는 함수.

output = activation(dot(W, input) + b)



```python
import tensorflow as tf

class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        w_shape = (input_size, output_size) #랜덤값으로 초기화된 (input_size, output_size)크기의 행렬
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)

        b_shape = (output_size,) #벡터의 크기지정
        b_initial_value = tf.zeros(b_shape) #(output_size,) 크기의 0으로 이루어진 벡터
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]
```

- 2. 단순한 Sequential 클래스


```python
class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
           x = layer(x)
        return x

    @property
    def weights(self):
       weights = []
       for layer in self.layers:
           weights += layer.weights
       return weights
```

- NaiveDense 클래스와 NaiveSequential 클래스를 이용하여 케라스와 유사한 모델을 만들 수 있다.


```python
model = NaiveSequential([ #2
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu), #1
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax) #1
])
assert len(model.weights) == 4
```

- 3. 배치 제너레이터 : MNIST 데이터를 미니 배치로 순회하는 방법


```python
import math

class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels
```

- 4. 훈련 스텝 실행코드


```python
#훈련 스텝 실행 : 가장 어려운 파트
#한 배치 데이터에서 모델을 실행하고 가중치를 업데이트하는 일

def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape: 
        # GradientTape 블록 안에서 모델의 예측을 계산
        #정방향 패스를 실행 
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
    gradients = tape.gradient(average_loss, model.weights)
    #가중치에 대한 손실의 그래디언트를 계산. Gradients 리스트의 각 항목은 model.weights리스트에 있는 가중치에 매칭됨
    update_weights(gradients, model.weights) #5
    #해당 그래디언트를 사용하여 가중치를 업데이트하는함수.
    return average_loss
```

- 5. update_weights 정의


```python
learning_rate = 1e-3

def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign_sub(g * learning_rate) #tensorflow 변수의 assign_sub 매서드는 -= dhk ehddlf
```

가중치 업데이트 단계를 수동으로 구현하는 경우는 거의 없고, Optimizer인스턴스를 사용한다.


```python
from tensorflow.keras import optimizers

optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))
```

- 전체 훈련 루프


```python
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"에포크 {epoch_counter}")
        batch_generator = BatchGenerator(images, labels) #3
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch) #4
            if batch_counter % 100 == 0:
                print(f"{batch_counter}번째 배치 손실: {loss:.2f}")
```

- 모델 학습 테스트


```python
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, epochs=10, batch_size=128)
```

    에포크 0
    0번째 배치 손실: 4.74
    100번째 배치 손실: 2.26
    200번째 배치 손실: 2.21
    300번째 배치 손실: 2.09
    400번째 배치 손실: 2.19
    에포크 1
    0번째 배치 손실: 1.90
    100번째 배치 손실: 1.90
    200번째 배치 손실: 1.83
    300번째 배치 손실: 1.71
    400번째 배치 손실: 1.80
    에포크 2
    0번째 배치 손실: 1.58
    100번째 배치 손실: 1.60
    200번째 배치 손실: 1.51
    300번째 배치 손실: 1.43
    400번째 배치 손실: 1.49
    에포크 3
    0번째 배치 손실: 1.32
    100번째 배치 손실: 1.36
    200번째 배치 손실: 1.24
    300번째 배치 손실: 1.21
    400번째 배치 손실: 1.26
    에포크 4
    0번째 배치 손실: 1.13
    100번째 배치 손실: 1.18
    200번째 배치 손실: 1.04
    300번째 배치 손실: 1.05
    400번째 배치 손실: 1.10
    에포크 5
    0번째 배치 손실: 0.98
    100번째 배치 손실: 1.04
    200번째 배치 손실: 0.90
    300번째 배치 손실: 0.93
    400번째 배치 손실: 0.98
    에포크 6
    0번째 배치 손실: 0.87
    100번째 배치 손실: 0.93
    200번째 배치 손실: 0.79
    300번째 배치 손실: 0.83
    400번째 배치 손실: 0.89
    에포크 7
    0번째 배치 손실: 0.79
    100번째 배치 손실: 0.84
    200번째 배치 손실: 0.72
    300번째 배치 손실: 0.76
    400번째 배치 손실: 0.83
    에포크 8
    0번째 배치 손실: 0.73
    100번째 배치 손실: 0.77
    200번째 배치 손실: 0.65
    300번째 배치 손실: 0.71
    400번째 배치 손실: 0.77
    에포크 9
    0번째 배치 손실: 0.68
    100번째 배치 손실: 0.72
    200번째 배치 손실: 0.60
    300번째 배치 손실: 0.66
    400번째 배치 손실: 0.73
    

- 모델 평가


```python
predictions = model(test_images)
predictions = predictions.numpy() # 넘파이 배열로 바꿈
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"정확도: {matches.mean():.2f}")
```

    정확도: 0.82
    


```python

```
