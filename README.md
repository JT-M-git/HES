HandWrittenEqn Solver
This project focuses on reading hand written or digital equations in the form of a picture and solves the respective equation thereby giving the mathematical output.
It mainly uses the concept of CNN to train the model with a prebuilt Machine Learning dataset. The dataset consists of features extracted from the mathematical symbols of +,-,/ and *
stored in a csv file. Feature Extraction for the dataset has been done using the OpenCv library to read each and every number and symbols followed by saving their features.
The strings present in the dataset for +,-,*,/ have been converted to categorical data and has been stored in an numpy array thereby being assigned a separate variable for it. 
The matrices of features has been reshaped into another numpy array so as to be recognised as a 4D tensor by the neural network. These features have been ben trained using the
CNN model with a certain epoch and batchsize giving us an accuracy of 98% (approx).The model has been saved using the keras library as a .h5 file for future predictions.
The image to be read and predicted is uploaded followed by feature extraction, reshaping height and width to match the previous inputs and storing it into a numpy array.
All the features collected are then concatenated to a list and is passed on to the model for prediction. The final equation predicted is converted into a string and is solved by
the eval() function of python.
The whole project can be summarized into the following steps:-
1) Extracting features from mathematical images and storing them in a CSV file.
2) Training the CNN model using the dataset.
3) Processing the input image so as to pass it on to the CNN model for prediction and solving the string equation.

Tools and Softwares used:-

1. Keras
2. Tensorflow(CPU)
3. Numpy
4. Scikit-Learn
5. PIL
6. itertools
7. pandas
8. PycharmIDE
9. Anaconda environment
