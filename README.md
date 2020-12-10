![IronHack Logo](https://s3-eu-west-1.amazonaws.com/ih-materials/uploads/upload_d5c5793015fec3be28a63c4fa3dd4d55.png)

# Leukemia Classification
<p <br/>
</p>
Veronica Agnolutto | Data PT Barcelona Jun 2020
<p <br/><br/>
</p>

 *It’s not some kind of miracle, cancer doesn’t grow from yesterday to today. It’s a pretty long process. There are signs in the tissue, but the human eye has limited ability to detect what may be very small patterns.*

<p style="text-align: right";>
  Regina Barzilay,<br />
  first winner of the Squirrel AI Award for AI for the Benefit of Humanity<br/><br/>
</p>

<p align="center", >
  <img width="1250" height="450" src="src/leukemia_cells.jpg">
</p>

## Contents

- [Overview](#Overview)
  - [Data](#Data)
  - [Description of the Dataset](#Description_of_the_Dataset)
- [Goal](#Goal)

- [Link](#Link)

- [ANNa](#ANNa)
  - [Meet ANNa](#Meet_ANNa)
      - [U-Net](#U-Net)
      - [ANNa's engineering](#ANNa's_engineering)
      - [Performance Charts](#Performance_Charts)
      - [Conclusions](#Conclusions)
      - [Future Improvements](#Future_Improvements)

- [Project structure](#Project_structure)

- [Tools](#Tools)
  - [Cloud](#Cloud)
  - [Libraries](#Libraries)

- [References](#References)


## Overview

[Acute lymphoblastic leukemia (ALL)](https://en.wikipedia.org/wiki/Acute_lymphoblastic_leukemia)  is the most common type of childhood cancer and accounts for approximately 25% of the pediatric cancers.

These cells have been segmented from microscopic images and are representative of images in the real-world because they contain some staining noise and illumination errors, although these errors have largely been fixed in the course of acquisition.

The task of identifying immature leukemic blasts from normal cells under the microscope is challenging due to morphological similarity and thus the ground truth labels were annotated by an expert oncologist.

### Data

Data are obtained from a  [Kaggle dataset](https://www.kaggle.com/andrewmvd/leukemia-classification) with more than 15,000 images, some of them of young patients affected by leukemia and others with no leukemia.

In total there are **15,135 images** from **118 patients** with two **labelled classes**:

- **Normal cell**
- **Leukemia blast**

### Description of the Dataset

**C-NMC Leukemia** (Classification  of Normal vs Malignant Cells) is a folder that contains the data arranged in three folds:

<p align="center", >
<img width="278" height="235" src="src/tree_C-NMC_Leukemia.jpg">
</p>

where:
  - all: Leukemia cells
  - hem: Normal (healthy) cells

## Goal

Tackle one of the major types of childhood cancer by creating a **model** that **classifies** between **diseased** and **healthy cells**.

To do this, we use **Deep learning** or **neural networks** are a flexible type of machine learning. They are models composed of **nodes** and **layers** inspired by the structure and function of the **brain**.

<p <br/><br/>
</p>

## Link
To access the entire project, please open the following link:
https://drive.google.com/drive/folders/1Tf5HlAoWrl_actixBloP_wcrP0V2pU92?usp=sharing

<p <br/><br/>
</p>

## ANNa

To achieve our goal, we create **ANNa**, a **Convolutional Network for Biomedical Image Segmentation** (**U-Net**)

As a general **convolutional neural network** focuses its task on **image classification**, where **input** is an **image** and **output** is one **label**, but in **biomedical cases**, it requires us not only to distinguish whether there is a disease, but also to **localise** the area of abnormality.

The reason it is able to localise and distinguish borders is by doing **classification on every pixel**, so the input and output share the same size.

### Meet ANNa

Let's discover **ANNa's Brain**!

In **Medical testing**, **Binary classification** is used to determine if a patient has certain disease or not (in our case, **Leukemia** or **Healty Cells**).

We create a **brain** made up of two parts:

- 1. **U-Net**: **Convolutional Network** for **Biomedical Image Segmentation**

      **Medical Image Segmentation** is the process of **automatic or semi-automatic detection of boundaries within a 2D or 3D image**.

      - Input: Image shape
      - Output: U-Net output

- 2. **Binary Classification**: **Neural Network**

      **Binary Classification**is used to determine if a patient has certain disease or not (in our case, **Leukemia** or **Healty Cells**).

        - Input: U-Net output
        - Output: Classification output

### U-Net

Now let’s get to the detail **implementation of U-Net** by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.

U-Net basic foundation looks like:

<div>
<p align="center", >
<img width="518" height="345" src="src/u-net-architecture.png">
    <p style="text-align: center;font-size:12px;">
        U-Net architecture
    </p>  
  </a>
</div>

 First sight, it has a `U shape`. The architecture is symmetric and consists of two major parts:

  - 1. `contracting path` (left): constituted by the general convolutional process;
  - 2. `expansive path` (right): constituted by transposed 2d convolutional layers.

For detailed information about U-Net structure, please check out this article: [U-Net line by line explanation](https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5).

### Classification

**Neural networks** will learn by figuring out that they got wrong and then working backward into the algorithm to discover what values and connections made the output incorrect.

Our **Neural Network** for **Binary Classification** is able to determine if a cell is healthy or not.

<div>
<p align="center", >
<img width="392" height="285" src="src/NN.JPG">
    <p style="text-align: center;font-size:12px;">
        Neural Network
    </p>  
  </a>
</div>


### Brain engineering

During the training process of a Neural Network, our aim is to try and minimize the loss function, by updating the values of the parameters (Weights) and make our predictions as accurate as possible.

Techniques used to **optimize** the behaviour of our Neuronal Networks.

- 1. **Early Stopping**

      **Early Stopping** is a regularization technique to combat the **overfitting issue**. With Early Stopping, we just stop training as soon as the validation error reaches the minimum.

      <p <br/><br/>
      </p>
- 2. **Learning Rate**

      **Deep learning neural** networks are trained using the **stochastic gradient descent optimization algorithm**.

      The **learning rate** is a hyperparameter that controls **how much to change the model** in response to the **estimated error** each time the model weights are updated.

### Performance Charts

Once you fit a deep learning neural network model, you must evaluate its performance on a test dataset.

- 1. **Classification Report** & **Confusion Matrix**

     Model Evaluation Metrics:
    - **Accuracy** tells you what percentage of the predictions are correct.
    - **Precision** tells us the following: from the positive predictions I have made, what percentage are positive.
    - **Recall** (opposed to precision), is useful when the cost of a false negative is high. It tells us: of all the positive predictions, what percentage confirms the prediction.
    - **F1-score** seeks a balance between precision and recall.

- 2. **Sensitivity** and **specificity** are statistical measures of the performance of a binary classification test that are widely used in **medicine**.

      - **Sensitivity** measures the proportion of true positives that are correctly identified.
      - **Specificity** measures the proportion of true negatives.

- 3. **ROC curve**

    **AUC-ROC** curve is a performance measurement for classification problem at various thresholds settings. It tells how much model is capable of distinguishing between classes.

<p <br/><br/>
</p>

### Conclusions

...

<p <br/><br/>
</p>

### Future Improvements

- Apply **ANNa** to **biomedical images** that require to **localise** the **area of abnormality**.

- **ANNa's Implementation** using **Pytorch** developed by Facebook’s AI research group. The advantage of Pytorch is that we can have more flexibility and control than Keras.

<p <br/><br/>
</p>



## Project structure  

- **README.md** Description of the project.
- **src** Images and resources.
- **input** Folder containing the original data.
- **notebooks**
  - **modules** Folder containing all the custom function created with Python.
  - **1.EDA**
  - **2.Image Preprocessing**
  - **3.Testing ANNa**
  - **4.ANNa**
  - **5.Test**

- **output**
  - **arrays**
  - **csv**
  - **models**
  - **viz**
- **.gitignore**

<p <br/><br/>
</p>

## Tools

**Cloud**

This project was done entirely in the **cloud**, using:

- **Google Drive**: cloud to store files
- **Google Colab**: platform
- **Python Compiler Editor**: run Python code in your browser directly


**Google Colab** is an online cloud based platform based on the Jupyter Notebook framework, designed mainly for use in ML and deep learning operations.

Other advantage of **Colab** is data versatility, allowing to ‘mount’ Google Drive onto our notebook.

**Libraries**

- File management: sys | os | glob
- Google: drive | colab
- Data Analysis: Numpy | Pandas
- Image manipulation: OpenCV | PIL
- Visualization: Matplotlib | Seaborn
- Neural Networks: TensorFlow | Keras
- Metrics: Scikit-learn

<p <br/>
</p>

## References

**Google Colab**
- [Google Colab](https://medium.com/swlh/machine-learning-google-colab-why-when-and-how-to-use-it-9624e34abd6d)

**Deep Learning**
- [Deep Learning and Medical Image Analysis with Keras](https://www.pyimagesearch.com/2018/12/03/deep-learning-and-medical-image-analysis-with-keras/)
- [Types of Convolutions in DL](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
- [Convolutional Neural Network](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
- [ML Model Regularization](https://towardsdatascience.com/machine-learning-model-regularization-in-practice-an-example-with-keras-and-tensorflow-2-0-52a96746123e)
- [How to Calculate Precision, Recall, F1 and More for Deep Learning Models](https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/)

**UNE-t**
- [U-Net line by line explanation](https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5)
