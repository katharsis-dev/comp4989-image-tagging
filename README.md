# COMP4989 Project (Group 4)
### Project Structure
```
.
├── code (Code used to train and test our model)/
│   ├── main.py (Model Training File)
│   ├── generator.py (Image Loader Class)
│   ├── test_sample.py (File to test pretrained model on dataset)
│   ├── variables.py (List of variables for directories to load data from)
│   └── model/
│       └── resnet50_mirflickr.py (Trained model using Resnet50 on the MirFlickr dataset)
├── demo-material (Materials used in demo)/
│   ├── image (Code and images used in image demo)
│   └── kernel (Code and images used in kernel demo)
└── requirements.txt (Python env requirements)
```

### Datasets
MirFlickr: https://press.liacs.nl/mirflickr/mirdownload.html
Images: http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip
Labels: http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip

Note: Once you extract the images from the mirflickr25k.zip file you also need to copy the code/output.csv into the extracted directory. Should look something like this:
```
.
├── mirflickr/
│   ├── im1.jpg
│   ├── im2.jpg
│   └── ...
└── output.csv
```

Movie Dataset: https://github.com/laxmimerit/Movies-Poster_Dataset
## Running our Code
In order to run our code please create a python enviroment based on the requirements.txt file.

We use the Movie Dataset for initial testing but actually trained our model on the MirFlickr dataset with contains around 25k images.

Training the model (MirFlickr)
1. Navigate to the "code" directory
2. Modify the code/variables.py file to reflect the locations of where the mirflickr dataset is located on your device.
3. You can change the epoch on line 145 to something smaller if you just want to test the code quickly.
4. Run the code with the command "python code/main.py".

Testing the model (MirFlickr)
We have a pretrained model inside the code/model directory that you can load in order to test our mode.

1. Open the code/variables.py file and make sure the "MODEL_LOAD" variable has the correct path set to the pretrained model directory which should be in code/model/resnet50_mirflickr.keras.
2. You can then open the code/test_sample.py file and modify the variable "entry" to select the image index you want the model to predict.
3. Now run the code/test_sample.py with the command "python code/test_sample.py" and see the printed result.
```
# Example:
   animals  baby  bird  car   clouds  dog  female  flower  food  indoor  lake  male  night  people  plant_life  portrait  river  sea      sky  structures  sunset  transport  tree  water
0      0.0   0.0   0.0  0.0  1.00000  0.0     0.0     0.0   0.0     0.0   0.0   0.0    0.0     0.0         0.0       0.0    0.0  0.0  1.00000     1.00000     0.0        0.0   0.0    0.0
1      0.0   0.0   0.0  0.0  0.07362  0.0     0.0     0.0   0.0     0.0   0.0   0.0    0.0     0.0         0.0       0.0    0.0  0.0  0.35585     0.57053     0.0        0.0   0.0    0.0
Row 0: Actual labels provided from the dataset
Row 1: Predicted values from the loaded model
```
