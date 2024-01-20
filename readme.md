# GAN for MNIST Digit Generation

This project implements a Generative Adversarial Network (GAN) for generating realistic handwritten digits using the MNIST dataset. The GAN architecture consists of a Generator and a Discriminator neural network, trained simultaneously to produce high-quality digit images.

## Project Structure

- **config.py**: Configuration file containing hyperparameters and model configurations.
  
- **model.py**: Definition of the Generator, Discriminator, and GAN models using TensorFlow and Keras.

- **train.py**: Training script that loads the MNIST dataset, initializes the models, and trains the GAN.

- **utils.py**: Utility functions, including data normalization and loss functions.

- **images**: Directory to store generated images during training.

- **models**: Directory to store saved models at specified intervals during training.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/gan-mnist.git
   cd gan-mnist# GAN for MNIST Digit Generation

2. Install dependencies:

   ```bash
   pip install -r requirements.txt

3. Run the training script::

   ```bash
   python train.py

## Configuration

- **config.py**: Adjust hyperparameters and model configurations in this file based on your requirements.

## Results

Generated images will be saved in the `images` directory during training. Additionally, the model weights for the Discriminator and Generator will be saved in the `models` directory at specified intervals.

## Customization

Feel free to experiment with different model architectures, hyperparameters, and datasets. You can also modify the training callback to save images and models at different intervals.

## References

- [Generative Adversarial Networks (GANs) - Ian Goodfellow et al.](https://arxiv.org/abs/1406.2661)
  
- [MNIST Dataset](https://www.tensorflow.org/datasets/catalog/mnist)



