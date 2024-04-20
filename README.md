<h1 align="center">Facial Encoding Generator</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="Made with PyTorch">
  <img src="https://img.shields.io/badge/One%20Shot%20Learning-✔-blue" alt="One Shot Learning">
  <img src="https://img.shields.io/github/license/eklavya-eg/facial-encoding-generator" alt="License">
</p>

<p align="center">
  Facial Encoding Generator is a project that implements one-shot learning using an Inception-Siamese network trained with PyTorch. It generates facial encodings and provides face verification functionality.
</p>

<h2 align="center">Features</h2>

- **One-Shot Learning:** Utilizes an Inception-Siamese network for one-shot learning, allowing effective face recognition with minimal data.
  
- **Facial Encoding Generation:** Generates facial encodings from input images using the trained network.
  
- **Face Verification:** Provides a face verification script to verify the identity of a person against stored facial encodings.
  
- **HDF5 Support:** Stores facial encodings in HDF5 (.h5) file format for efficient data storage and retrieval.

<h2 align="center">Usage</h2>

1. **Training:** Train the Inception-Siamese network using your dataset.
   
2. **Generating Encodings:** Use the trained model to generate facial encodings from input images.
   
3. **Face Verification:** Verify the identity of a person by comparing their facial encoding against stored encodings.

<h2 align="center">Getting Started</h2>

To get started with Facial Encoding Generator, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/eklavya-eg/facial-encoding-generator.git
