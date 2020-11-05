import tensorflow as tf

# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers:
# 加载数据， 将int转化为float
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 将模型的各层堆叠起来，以搭建 tf.keras.Sequential 模型。为训练选择优化器和损失函数：
model = tf.keras.Sequential([
  # flatter 是拍平，将二维拍平成一维
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # 稠密的神经网络层
  tf.keras.layers.Dense(128, activation='relu'),
  # 进行dropout
  tf.keras.layers.Dropout(0.2),
  # 稠密的神经网络层
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练并验证模型：
model.fit(x_train, y_train, epochs=10)

model.evaluate(x_test,  y_test, verbose=1)
