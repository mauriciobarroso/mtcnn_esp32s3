# TensorFlow Lite Micro MTCNN implementation for ESP32-S3
This is an implementation of MTCNN (Multitask Cascading Convolutional Networks) for the ESP32-S3 SoC (System on Chip) using TensorFlow Lite for Microcontrollers. The main goal is to detect faces in low-cost hardware and reduced technical features.

## MTCNN
MTCNN is a framework developed as a solution for both face detection and face alignment. It consist in three stages of convolutional networks that are able to recognize faces and landmark location such as eyes, nose and mouth. 

### P-Net (Proposal Network)
This is a FCN (Fully Convolutional Network) that is used to obtain candidate windows and their bounding box regression vectors. Bounding box regression is a popular technique to predict the localization of boxes when the goal is detecting an object of some pre-defined class. The candidate windows obtained are calibrated with bounding box regression vectors and processed with NMS (Non Max Supression) operator to combine overlapping regions.

<p align="center">
  <img src="images/p-net.webp" alt="P-Net architecture" width="75%"/>
</p>

### R-Net (Refine Network)
The R-Net further reduces the number of candidates, performs calibration with bounding box regression and employs NMS to merge overlapping candidates. This network  is a CNN, not a FCN like P-Net sice there is a dense layer at the last stage of its architecture.

<p align="center">
  <img src="images/r-net.webp" alt="R-Net architecture" width="75%"/>
</p>

### O-Net (Output Network)
This stage is similar to the R-Net, but this Output Network aims to describe the face in more detail and output the five facial landmarks’ positions for eyes, nose and mouth.

<p align="center">
  <img src="images/o-net.webp" alt="O-Net architecture" width="75%"/>
</p>

## TensorFlow implementation
To implement the MTCNN models the tools used were TensorFlow and Google Colab. TensorFlow is an open source library for ML (Machine Learning) developed for Google and it is capable to bulilding and training neural networks to detect patterns and correlations. Google Colab is a product from Google Research and allows write and execute arbitrary python code through the browser, and is specially well suited to ML, data analysis and education.

For a correct implementation of MTCNN, the input and output data of the models must be processed to guarantee the best results. The next diagram shows the diagram block of the pipeline implemented.

<p align="center">
  <img src="images/mtcnn_pipeline.png" alt="MTCNN pipeline" width="50%"/>
</p>

The first step is to perform an image pyramid to create different scales of the input image and detect faces of different sizes. These new scaled images are the inputs to P-Net which generates the offsets and scores for each candidate window. Then these outputs are post-processed to obtain the coordinates where the faces would meet. The R-Net input must be pre-processed with the previous outputs, in this way new candidate windows are obtained. The R-Net outputs are the offsets and scores of the candidate windows that are post-processed to obtain the new coordinates where the faces would meet. Finally, for O-Net, the R-Net process is repeated and the coordinates of the faces in the input image are obtained.

The pre-process consist of two steps, crop the input image according to the bounding boxes coordinates obtained before and resize the cropped images to match the input shape of the model.

<p align="center">
  <img src="images/mtcnn_preprocess.png" alt="MTCNN pre-process" width="25%"/>
</p>

In the other hand, the post-process consist of three steps, apply NMS to combine overlapped regions, calibrate the bounding boxes with the offset obtained before, square and correct the final bounding boxes coordinates.

<p align="center">
  <img src="images/mtcnn_postprocess.png" alt="MTCNN post-process" width="40%"/>
</p>

All the prrocesses detailed before were implemented in python in the next Google Colab Notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mauriciobarroso/mtcnn_esp32s3/blob/main/mtcnn.ipynb)

## Deploying to ESP32-S3


### Hardware

### Firmware

## Credits

## License

MIT License

Copyright (c) 2023 Mauricio Barroso Benavides

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.