DESCRIPTION = """## Brief

This exploration tool allows visualisation of the impact of transformation 
techniques & deep learning models on cell images classification accuracy. Following is the general structure of the app -

- CNN Models
    - Description (you are here)
    - Overall
    - By Cluster
- Best Model
    - Overall
    - By Cluster
- Image Prediction

The CNN Models tab that we are currently in is responsible for showcasing the findings of our investigation.
The overall sub-tab takes us to a view of some general metrics, a learning curve, and a metrics table that dynamically 
updates based on the user's choice. Here, the panel on the left becomes useful; it lets us choose one of the **models**, **transformations**, and **masking** techniques 
we implemented, and updates the visualisations dynamically. Descriptions for all of these are provided below. The last sub-tab helps visualise the model's performance 
per cluster, which can give novel insights into the dataset and the problem as a whole.

The second tab is dedicated to a comparison between the best model, and the second best which comes from all the models we saw under CNN Models. 
This is a relatively static tab with a metrics comparison table that updates dynamically. The *overall* and *by cluster* sub-tabs are still present from before.

Lastly, the image prediction tab lets us upload a PNG image of a cell, and choose any one of the models seen before to make a prediction on which cluster it belongs to. 
The most computationally model, ViT, has been excluded from this list due to how much time it takes to run. We wish this to be a future implementation.

## Models

#### Basic CNN (Lab Model)

The `lab model` was the basic model provided to
the Image2 project team. This class is a basic CNN model designed for image
classification, built using **PyTorch** and have 9 layers deep. The input to the
model is a single-channel image, and the output is a tensor of 28 values
representing the predicted class probabilities for each of the 28 classes. This
model served as the baseline for the project exploring different transformation
and augmentation techniques.

#### ResNet18

The `ResNet18` class represents a more advanced
CNN model that is 18 layers deep, with target adjustments applied by our team.
This model incorporates several enhancements, including modifying the first
convolutional layer to accept single-channel input images, making it suitable
for grayscale image classification. It also includes dropout layers to prevent
overfitting and stabilise the training process. In general, as this model is
more complex than th basic lab model, we expect it to have better performance
and better fitting with the training data.

#### ResNet50

The `ModifiedResNet50` class represents an even
more advanced CNN model, based on the ResNet-50 architecture, which is 50 layers
deep. This model also includes several enhancements and incorporates dropout and
batch normalisation layers to prevent overfitting and stabilise the training
process. With its deeper architecture and more complex structure compared to the
ResNet-18, the ModifiedResNet50 is expected to deliver better performance and
more accurate fitting wit the training data, thus enhancing the overall
effectiveness of the image classification task

## Transformations

#### Normalization

The **Normalization** process involves scaling each pixel value in the images to a range between 0 and 1. This
is done by subtracting the mean and dividing by the standard deviation of the
original data, ensuring that the input images have a consistent scale.
(normalised_value = (original_value-mean)/std).

#### Random Flip and Random Rotation

Beyond the normalisation, we then
implement the augmentation technique including **Random Flipping** and **Random
Rotation** to enhance the robustness and generalization of models. Random flipping
involves randomly flipping images horizontally or vertically, helping the model
become invariant to the orientation of objects. Random rotation entails rotating
images by a random angle within a specified range, allowing the model to
recognize objects regardless of their rotational orientation. These
transformations expose the model to a wider variety of image variations.

## Masking

#### Cell Boundary

The **Cell Boundary** masking technique used is
based on a predefined boundary, which identifies the pixels inside and outside
of the boundary. All pixels outside the boundary are set to 0, focusing the
model's attention on the relevant areas within the boundary.

#### Gaussian Blur

The **Gaussian Blur** masking technique assigns weights to each pixel based on its distance from the centre, with closer
pixels having higher weights. This results in a blurred image where each pixel's
value is influenced by its neighbours, creating a smoothing effect.

#### Sobel Edge

The **Sobel Edge** technique works by convolving the image with Sobel kernels, which are matrices
specifically designed to respond maximally to edges running vertically and
horizontally relative to the pixel grid. These kernels calculate the gradient of
the image intensity at each pixel, identifying regions with high spatial
frequency. The result is an image that emphasizes the edges, making it easier to
detect boundaries and transitions between different regions."""