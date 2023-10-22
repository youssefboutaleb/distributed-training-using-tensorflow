# Repository Description

## Project: Distributed Training Using TensorFlow

This repository contains code and resources related to the "Distributed Training Using TensorFlow" project.

### Directory Structure

- **code/**
  - `main.py`: The main code file for executing the project.

- **sources/**
  - `links.txt`: A curated list of useful links, including videos, blogs, and more.
  - `presentation.pptx`: A presentation detailing the methods and techniques used in the "Distributed Training Using TensorFlow" project.

- **src_readme/**
  - `enis.jfif`: Image of our school.
  - `tensorflow.png`: TensorFlow logo.

- `README.md`: The file you are currently reading.

- `environment.yml`: Environment configuration file.

### About the Project

This project focuses on distributed training using TensorFlow. It includes code for executing the project, a collection of useful links, and a presentation that explains the methods and techniques used. Additionally, you can find images related to our school and the TensorFlow logo in the `src_readme/` directory.

Feel free to explore the contents of this repository and refer to the provided resources for more information.



# TensorFlow-Distributed-Deep-Learning-Training
<center><p align="center" width="100%"><img 
src="https://www.jeuneafrique.com/medias/2018/02/26/enis.png" /></p></center>
This is an academic project titled "TensorFlow-Distributed-Deep-Learning-Training" that aims to showcase the fundamentals of distributed deep learning using TensorFlow. In this project, we provide a simple yet informative example of distributed training setup using TensorFlow. The primary objectives of this project are as follows:

1.  **Multi-Worker Configuration:** We demonstrate how to set up a multi-worker TensorFlow cluster, allowing multiple machines to collaborate in the training process. The provided code illustrates how to define a cluster specification and configure the TF_CONFIG environment variable.
    
2.  **Data Distribution:** The project emphasizes data distribution, showcasing how to split and preprocess data for training across distributed workers. It offers insights into how data should be divided and labeled for effective training.
    
3.  **Model Training:** Within the multi-worker setup, we train a basic convolutional neural network (CNN) model on a portion of the MNIST dataset. This model architecture, training loop, and strategy configuration are designed to serve as educational material for those new to distributed deep learning.
    
4.  **Model Saving:** The trained model is saved to a file ('model.h5') for later use or evaluation. This demonstrates how to persist the model for future inference or fine-tuning.

### presentation : 
you can found a presentation of the project under the sources directory or [here](https://docs.google.com/presentation/d/1u6-NVoZSwOWOAhT7Dotpoyo8v8j63153FxTbM7i8yaU/edit?usp=sharing) (online using google slides )





## Implementation details:

- **Cluster Configuration:** The code sets up a multi-worker TensorFlow cluster for distributed training. It specifies the workers' IP addresses and their roles within the cluster using the TF_CONFIG environment variable.

- **Data Preprocessing:** A subset of the MNIST dataset is used for training. Data preprocessing involves loading and reshaping the data, as well as normalizing pixel values for effective training.

- **Model Architecture:** The project defines a simple convolutional neural network (CNN) model for training. The model consists of convolutional layers, max-pooling, and dense layers for classification.

- **Distributed Training:** TensorFlow's `tf.distribute.MultiWorkerMirroredStrategy` is utilized for distributed training, allowing multiple workers to collaborate in training the model. Training is executed within the defined strategy scope.

- **TensorBoard Integration:** TensorBoard is integrated into the project for visualizing training logs. 
- **Model Saving:** After training, the entire model is saved to a file ('model.h5') for future use or evaluation.


## Run it locally
To run the code  locally, follow these steps:

#### Clone the repository 

```bash
  git clone https://github.com/youssefboutaleb/distributed-training-using-tensorflow.git
```

#### Go to the project directory

```bash
  cd distributed-training-using-tensorflow
```

#### Set up a virtual environment and install the required Python packages 
 Create a Virtual Environment

```bash
  python -m venv <venv>
```
 Activate the Virtual Environment:
- On Windows:

```bash
.\<venv>\Scripts\activate
```
- On macOS and Linux:

```bash
source venv/bin/activate
```

##### Note :
1.  Change the address IP of the  workers in " cluster_spec ".
2.  If you use anaconda you can easially create an envirement from "environment.yml".

By following these steps, you can ensure that the code can work sucessfully . But note if a worker start the training , he  will wait the others until start the training also. 

#### Start the training

```bash
  python main.py

```
#### Deactivate the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment by running the following command:

```bash
deactivate
```


