# VerifyNet
# Siamese Neural Network Face Verification Model  

## Overview

This project implements a **scalable Siamese Neural Network for face verification**, leveraging the concept of learning a similarity metric between pairs of images. 
The approach is inspired by the research paper "Siamese Neural Networks for One-shot Image Recognition" by Koch et al., which demonstrates how such networks excel in 
tasks where limited labeled data is available, such as one-shot learning and face verification[1].

## Problem Statement

Traditional face recognition systems require large datasets and retraining for new identities. This project aims to **verify whether two face images belong to the same person** 
(face verification) using a Siamese Neural Network, which is particularly effective in scenarios with limited data per identity and can generalize to unseen classes[1].

## Core Concepts

- **Siamese Network Structure:**  
  Two identical neural networks (sharing weights) process two input images (anchor and candidate).
  Their outputs are compared using a distance metric (e.g., L1 distance) to predict similarity[1][2].
- **Triplet Data Formation:**  
  Training uses three types of images:
  - **Anchor:** Reference image of a person.
  - **Positive:** Another image of the same person.
  - **Negative:** Image of a different person[2].
- **One-Shot Learning:**  
  The network learns to generalize from very few examples, making it suitable for face verification with limited labeled data[1].

## Project Structure

- **Data Collection:**  
  Uses webcam to capture anchor, positive, and negative images, storing them in separate directories
  (`data/anchor`, `data/positive`, `data/negative`)[2].
- **Preprocessing:**  
  - Images are resized to 100x100 pixels and normalized to [2] range.
  - TensorFlow's `tf.data.Dataset` API is used for efficient data loading and batching[2].
- **Model Architecture:**  
  - **Embedding Network:**  
    - Several convolutional and max-pooling layers extract features.
    - Output is a dense vector representing the face.
  - **Siamese Network:**  
    - Takes two images as input.
    - Computes feature embeddings for both.
    - Calculates the L1 distance between embeddings.
    - Final dense layer with sigmoid activation predicts similarity[2][1].
- **Training:**  
  - Binary cross-entropy loss.
  - Adam optimizer.
  - Custom training loop with TensorFlow's `GradientTape` for flexibility.
  - Precision and recall metrics are tracked[2].
- **Verification:**  
  - For a given input image, compares against a set of verification images.
  - Computes similarity scores and applies thresholds to decide verification[2].

## Key Implementation Details

### Data Pipeline

```python
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100,100))
    img = img / 255.0
    return img
```
- **Positive and Negative Pairing:**  
  Dataset is constructed by zipping anchor-positive (label 1) and anchor-negative (label 0) pairs,
   then concatenating and shuffling[2].

### Model Definition

```python
def make_embedding():
    inp = Input(shape=(100,100,3), name='input_image')
    # ... convolutional and pooling layers ...
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    return Model(inputs=[inp], outputs=[d1], name='embedding')
```

- **L1 Distance Layer:**  
  Custom Keras layer computes the absolute difference between two embeddings[2].

### Siamese Network

```python
def make_siamese_model():
    input_image = Input(name='input_img', shape=(100,100,3))
    validation_image = Input(name='validation_img', shape=(100,100,3))
    distances = L1Dist()(embedding(input_image), embedding(validation_image))
    classifier = Dense(1, activation='sigmoid')(distances)
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
```

## Training and Evaluation

- **Loss Function:**  
  Binary cross-entropy, as the task is to classify pairs as same/different[1][2].
- **Optimization:**  
  Adam optimizer with a learning rate of 1e-4.
- **Epochs:**  
  Typically trained for 20 epochs, with checkpoints saved periodically[2].
- **Metrics:**  
  Precision and recall are calculated per epoch for monitoring performance[2].
- **Data Augmentation:**  
  The referenced paper suggests affine distortions to improve generalization,
  though the provided code does not implement this[1].

## Verification Procedure

- **Input:**  
  A new face image is compared against a gallery of verification images.
- **Process:**  
  - Preprocess both input and gallery images.
  - Use the trained Siamese model to predict similarity scores.
  - Apply detection and verification thresholds to determine if the input matches any gallery image[2].
- **Output:**  
  Boolean indicating verification result, and similarity scores for analysis.

## Scalability Considerations

- **Efficient Data Handling:**  
  Uses TensorFlow's data pipeline for batching and prefetching.
- **Model Checkpointing:**  
  Allows training to resume and prevents loss of progress.
- **Modular Design:**  
  Embedding network can be replaced or fine-tuned for other domains or larger datasets.

## Research Alignment

- **Metric Learning:**  
  The project closely follows the approach in Koch et al., using a learned metric
  (L1 distance in embedding space) for verification tasks[1].
- **Generalization:**  
  The architecture is designed to generalize to unseen classes, as shown in the one-shot
  learning experiments in the referenced paper[1].
- **Performance:**  
  Convolutional Siamese networks have been shown to approach human-level accuracy on
  verification tasks and outperform traditional baselines[1].

## Usage Instructions

1. **Data Collection:**  
   - Run the script to collect anchor (`a`), positive (`p`), and negative images via webcam.
2. **Training:**  
   - Train the Siamese network using the provided training loop.
   - Monitor precision and recall.
3. **Verification:**  
   - Use the verification function to check if a new face matches any in the verification set.
   - Adjust thresholds for your application's security requirements.

## References

- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). *Siamese Neural Networks for One-shot Image Recognition*.[1]
