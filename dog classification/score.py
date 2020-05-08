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
    model_path = Model.get_model_path(model_name='dog_model')
    my_ResNet50_model = load_model(model_path + '/data/dog_resnet50.h5')
    global base_ResNet50_model
    base_ResNet50_model = load_model(model_path + '/data/notop_resnet50.h5')

@rawhttp
def run(request):
    if request.method != 'POST':
        return AMLResponse("bad request, use post method", 500)
    image = Image.open(io.BytesIO(request.get_data(False)))
    # preprocess the image and prepare it for classification
    image = prepare_image(image, target=(224, 224))

    bottleneck_feature = base_ResNet50_model.predict(image)

    # make prediction
    predicted_vector = my_ResNet50_model.predict(bottleneck_feature)
    predicted_index = np.argmax(predicted_vector)

    dog_names = ['AFFENPINSCHER',
                 'AFGHAN HOUND',
                 'AIREDALE TERRIER',
                 'AKITA',
                 'ALASKAN MALAMUTE',
                 'AMERICAN ESKIMO DOG',
                 'AMERICAN FOXHOUND',
                 'AMERICAN STAFFORDSHIRE TERRIER',
                 'AMERICAN WATER SPANIEL',
                 'ANATOLIAN SHEPHERD DOG',
                 'AUSTRALIAN CATTLE DOG',
                 'AUSTRALIAN SHEPHERD',
                 'AUSTRALIAN TERRIER',
                 'BASENJI',
                 'BASSET HOUND',
                 'BEAGLE',
                 'BEARDED COLLIE',
                 'BEAUCERON',
                 'BEDLINGTON TERRIER',
                 'BELGIAN MALINOIS',
                 'BELGIAN SHEEPDOG',
                 'BELGIAN TERVUREN',
                 'BERNESE MOUNTAIN DOG',
                 'BICHON FRISE',
                 'BLACK AND TAN COONHOUND',
                 'BLACK RUSSIAN TERRIER',
                 'BLOODHOUND',
                 'BLUETICK COONHOUND',
                 'BORDER COLLIE',
                 'BORDER TERRIER',
                 'BORZOI',
                 'BOSTON TERRIER',
                 'BOUVIER DES FLANDRES',
                 'BOXER',
                 'BOYKIN SPANIEL',
                 'BRIARD',
                 'BRITTANY',
                 'BRUSSELS GRIFFON',
                 'BULL TERRIER',
                 'BULLDOG',
                 'BULLMASTIFF',
                 'CAIRN TERRIER',
                 'CANAAN DOG',
                 'CANE CORSO',
                 'CARDIGAN WELSH CORGI',
                 'CAVALIER KING CHARLES SPANIEL',
                 'CHESAPEAKE BAY RETRIEVER',
                 'CHIHUAHUA',
                 'CHINESE CRESTED',
                 'CHINESE SHAR-PEI',
                 'CHOW CHOW',
                 'CLUMBER SPANIEL',
                 'COCKER SPANIEL',
                 'COLLIE',
                 'CURLY-COATED RETRIEVER',
                 'DACHSHUND',
                 'DALMATIAN',
                 'DANDIE DINMONT TERRIER',
                 'DOBERMAN PINSCHER',
                 'DOGUE DE BORDEAUX',
                 'ENGLISH COCKER SPANIEL',
                 'ENGLISH SETTER',
                 'ENGLISH SPRINGER SPANIEL',
                 'ENGLISH TOY SPANIEL',
                 'ENTLEBUCHER MOUNTAIN DOG',
                 'FIELD SPANIEL',
                 'FINNISH SPITZ',
                 'FLAT-COATED RETRIEVER',
                 'FRENCH BULLDOG',
                 'GERMAN PINSCHER',
                 'GERMAN SHEPHERD DOG',
                 'GERMAN SHORTHAIRED POINTER',
                 'GERMAN WIREHAIRED POINTER',
                 'GIANT SCHNAUZER',
                 'GLEN OF IMAAL TERRIER',
                 'GOLDEN RETRIEVER',
                 'GORDON SETTER',
                 'GREAT DANE',
                 'GREAT PYRENEES',
                 'GREATER SWISS MOUNTAIN DOG',
                 'GREYHOUND',
                 'HAVANESE',
                 'IBIZAN HOUND',
                 'ICELANDIC SHEEPDOG',
                 'IRISH RED AND WHITE SETTER',
                 'IRISH SETTER',
                 'IRISH TERRIER',
                 'IRISH WATER SPANIEL',
                 'IRISH WOLFHOUND',
                 'ITALIAN GREYHOUND',
                 'JAPANESE CHIN',
                 'KEESHOND',
                 'KERRY BLUE TERRIER',
                 'KOMONDOR',
                 'KUVASZ',
                 'LABRADOR RETRIEVER',
                 'LAKELAND TERRIER',
                 'LEONBERGER',
                 'LHASA APSO',
                 'LOWCHEN',
                 'MALTESE',
                 'MANCHESTER TERRIER',
                 'MASTIFF',
                 'MINIATURE SCHNAUZER',
                 'NEAPOLITAN MASTIFF',
                 'NEWFOUNDLAND',
                 'NORFOLK TERRIER',
                 'NORWEGIAN BUHUND',
                 'NORWEGIAN ELKHOUND',
                 'NORWEGIAN LUNDEHUND',
                 'NORWICH TERRIER',
                 'NOVA SCOTIA DUCK TOLLING RETRIEVER',
                 'OLD ENGLISH SHEEPDOG',
                 'OTTERHOUND',
                 'PAPILLON',
                 'PARSON RUSSELL TERRIER',
                 'PEKINGESE',
                 'PEMBROKE WELSH CORGI',
                 'PETIT BASSET GRIFFON VENDEEN',
                 'PHARAOH HOUND',
                 'PLOTT',
                 'POINTER',
                 'POMERANIAN',
                 'POODLE',
                 'PORTUGUESE WATER DOG',
                 'SAINT BERNARD',
                 'SILKY TERRIER',
                 'SMOOTH FOX TERRIER',
                 'TIBETAN MASTIFF',
                 'WELSH SPRINGER SPANIEL',
                 'WIREHAIRED POINTING GRIFFON',
                 'XOLOITZCUINTLI',
                 'YORKSHIRE TERRIER',
                 'SHIBA INU']
    return json.dumps({'dog_breed': dog_names[predicted_index]})


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