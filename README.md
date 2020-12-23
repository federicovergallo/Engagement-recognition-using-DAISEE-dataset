# DAiSEE Engagement Recognition
Implementation in Tensorflow 2.x of Engagement Recognition using DAiSEE dataset

Engagement Recognition is made through a multi-label classification of the following status:
- engaged
- bored
- frustrated 
- confused

This implementation comes with a camera script so to test it using your webcam.


<img src="figures/output.png" style="zoom:80%;width:80%;height:80%" />

## Dataset
Download the dataset running the .sh script

## Installation
You can install dependencies as usual using pip install -r requirements.txt
Then you need to preprocess the dataset by running daisee_data_preprocessing.py script.
Afterwards run train.py to train the model and test it using main.py script



