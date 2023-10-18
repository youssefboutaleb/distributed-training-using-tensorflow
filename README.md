

# TensorFlow-Distributed-Deep-Learning-Training
<center><p align="center" width="100%"><img 
src="https://www.jeuneafrique.com/medias/2018/02/26/enis.png" /></p></center>
This is an academic project titled "TensorFlow-Distributed-Deep-Learning-Training" that aims to showcase the fundamentals of distributed deep learning using TensorFlow. In this project, we provide a simple yet informative example of distributed training setup using TensorFlow. The primary objectives of this project are as follows:

1.  **Multi-Worker Configuration:** We demonstrate how to set up a multi-worker TensorFlow cluster, allowing multiple machines to collaborate in the training process. The provided code illustrates how to define a cluster specification and configure the TF_CONFIG environment variable.
    
2.  **Data Distribution:** The project emphasizes data distribution, showcasing how to split and preprocess data for training across distributed workers. It offers insights into how data should be divided and labeled for effective training.
    
3.  **Model Training:** Within the multi-worker setup, we train a basic convolutional neural network (CNN) model on a portion of the MNIST dataset. This model architecture, training loop, and strategy configuration are designed to serve as educational material for those new to distributed deep learning.
    
4.  **Model Saving:** The trained model is saved to a file ('model.h5') for later use or evaluation. This demonstrates how to persist the model for future inference or fine-tuning.









## Implementation details:

- **Cluster Configuration:** The code sets up a multi-worker TensorFlow cluster for distributed training. It specifies the workers' IP addresses and their roles within the cluster using the TF_CONFIG environment variable.

- **Data Preprocessing:** A subset of the MNIST dataset is used for training. Data preprocessing involves loading and reshaping the data, as well as normalizing pixel values for effective training.

- **Model Architecture:** The project defines a simple convolutional neural network (CNN) model for training. The model consists of convolutional layers, max-pooling, and dense layers for classification.

- **Distributed Training:** TensorFlow's `tf.distribute.MultiWorkerMirroredStrategy` is utilized for distributed training, allowing multiple workers to collaborate in training the model. Training is executed within the defined strategy scope.

- **TensorBoard Integration:** TensorBoard is integrated into the project for visualizing training logs. 
- **Model Saving:** After training, the entire model is saved to a file ('model.h5') for future use or evaluation.

