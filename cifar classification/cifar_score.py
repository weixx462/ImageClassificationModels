import os, json
import numpy as np
from azureml.core.model import Model
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from PIL import Image
import io
import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.applications.resnet50 import ResNet50


def init():
    global my_ResNet50_model
    model_path = Model.get_model_path(model_name='cifar_model')
    my_ResNet50_model = load_model(model_path + '/data/cifar_resnet50.h5')
    global base_ResNet50_model
    base_ResNet50_model = load_model(model_path + '/data/notop_resnet50.h5')

@rawhttp
def run(request):
    if request.method != 'POST':
        return AMLResponse("bad request, use post method", 500)
    image = Image.open(io.BytesIO(request.get_data(False)))
    # preprocess the image and prepare it for classification
    image = prepare_image(image, target=(32, 32))

    bottleneck_feature = base_ResNet50_model.predict(image)

    # make prediction
    predicted_vector = my_ResNet50_model.predict(bottleneck_feature)
    predicted_index = np.argmax(predicted_vector)

    labels = ['AIRPLANE',
     'AUTOMOBILE',
     'BIRD',
     'CAT',
     'DEER',
     'DOG',
     'FROG',
     'HORSE',
     'SHIP',
     'TRUCK']

    return json.dumps({'predicted_label': labels[predicted_index]})


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    # return the processed image
    return image