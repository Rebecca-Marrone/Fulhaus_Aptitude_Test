#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:09:08 2023

@author: becca
"""


from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import io

app = Flask(__name__)

model = torch.load('best_model.pth')
classes = ['Bed', 'Chair', 'Sofa']

image_transform = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor()
])

# Define the route for the image classification API
@app.route('/', methods=['POST', 'GET'])
def classify():
    if request.method == 'POST':
        # Load the image from the request
        file_path = request.form['file_path']
        image = Image.open(file_path)

        # Transform the image
        image = image_transform(image).float()
        image = image.unsqueeze(0)

        # Pass the image through the model and get the prediction
        output = model(image)
        _, pred = torch.max(output.data, 1)
        result = classes[(int(pred.item())-1)]

        # Return the prediction as a JSON response
        return jsonify({'result': result})
    
    # If the request method is GET, return the upload form
    return '''
        <form method="POST">
            <label for="file_path">File path:</label>
            <input type="text" id="file_path" name="file_path">
            <input type="submit" value="Submit">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
    
