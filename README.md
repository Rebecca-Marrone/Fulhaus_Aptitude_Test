# BedChairSofaNN.py
The python script named "BedChairSofaNN.py" is designed to perform classification on an image dataset consisting of bed, chair, and sofa images. The script uses a neural network model, specifically a pre-trained ResNet-18 model that is fine-tuned to classify the given dataset. The script also includes functions for training and evaluating the model.

## Dependencies:

- Python 3.x
- PyTorch
- Pandas
- scikit-learn
- Pillow (PIL)
- torchvision
- Data inputs

The images for the dataset are found in the folder 'Data for test' which should be saved within the same folder as the python scripts. The folder should contain three subfolders named 'bed', 'chair', and 'sofa', each containing the respective images.

## Functionality:

- The script defines the BedChairSofa dataset class and its functions.
- The script splits the dataset into training and testing sets.
- The script defines the transformations to be applied to the training and testing sets.
- The script defines the neural network model using a pre-trained ResNet-18 model that is fine-tuned.
- The script defines the train_nn and evaluate_test functions used to train and evaluate the model.
- The script saves the best model and its state in the file 'model_best_state.pth.tar'.

## Instructions:

To run the script, open the script in python editor and run.

## Output:

- The script prints the training and testing accuracy at each epoch of training. It also prints the final training and testing accuracy.
- The script saves the best model and its state in the file 'model_best_state.pth.tar'.

Note: It is assumed that the necessary libraries are installed and that the data is correctly formatted and located in the 'Data for test' folder.

# API
The Python script named "API.py" is designed to perform image classification using a pre-trained PyTorch model. Specifically, it is designed to classify images of beds, chairs, and sofas. The script uses a ResNet-18 model that has been fine-tuned on the dataset.

## Dependencies:

- Python 3.x
- Flask
- PyTorch
- Pillow (PIL)
- torchvision

## Functionality:

- The script defines a Flask web application that exposes an API endpoint for image classification.
- The script loads the pre-trained PyTorch model.
- The script defines the classes for image classification (bed, chair, and sofa).
- The script defines the image transformations to be applied before classification.
- The script defines the classify function to process incoming images and return a prediction.
- The script provides an HTML form for submitting images to the API.

## Instructions:

To run the script, open the script in python editor and run.
While the code is running, input the url into any browser search bar and press enter. 
Input the file path to any new image of a bed, chair or sofa in the appropriate area and submit.


## Output

- The script runs a Flask web application on the localhost, and listens on port 5000.
- The API can be accessed by submitting an image file through the HTML form, or by sending a POST request to the API endpoint with the image file path.
- The script returns the predicted class for the input image as a JSON response.
