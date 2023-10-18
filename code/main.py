import os
import json
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import TensorBoard
from tensorboard import program

# Define the cluster specification
cluster_spec = {
    "cluster": {
        "worker": ["192.168.137.229:2223", "192.168.137.1:2223"]
    },
    "task": {"type": "worker", "index": 1}
}

# Set the TF_CONFIG environment variable
os.environ['TF_CONFIG'] = json.dumps(cluster_spec)

# Load and preprocess data
(train_data, train_labels), _ = datasets.mnist.load_data()
train_data = train_data[10000:10010]  # Machine 1 gets the first 20,000 examples (adjust as needed)
train_labels = train_labels[10000:10010]  # Make sure to slice labels accordingly
train_data = train_data / 255.0
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)  # Reshape the data
"""
# Create a directory for TensorBoard logs
log_dir = "logs"

# Configure TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir)

# Start TensorBoard
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()

print("TensorBoard is available at:", url)
"""


# Distributed training
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    # Define the model within the strategy scope
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Now start the training with the TensorBoard callback
model.fit(train_data, train_labels, epochs=3)
# Save the entire model (architecture and weights) to a file
model.save('node1.h5')
