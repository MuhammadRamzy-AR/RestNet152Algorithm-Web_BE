from crypt import methods
import json
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from PIL import Image, ImageOps
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__) # new
CORS(app) # new
app.config["imgdir"] = "Images/"


"""
function untuk menjalankan klasifikasi menggunakan model ResNet-152 V2
"""
@app.route('/resnetv1', methods=['POST'])
def resnet_v1():
    # Load Model Resnet152_v1
    """
    Memanggil model ResNet 152 v1 yang telah dilatih dengan data training
    serta diujikan kepada data testing dan untuk prediksi gambarnya
    """
    model_v1_path = 'Model/resnet152_v1.h5'
    model_v1 = load_model(model_v1_path, compile=False)

    """
    Load gambar yang dikirim dari sisi front-end
    """
    img_uploaded = request.files
    img_file = img_uploaded.get('file')
    filename_orig = secure_filename(img_file.filename) # used for make it easier to see what is the original file
    filename = 'identification_img.png' # save file 
    filepath = os.path.join(app.config['imgdir'], filename);
    img_pred1 = img_file.save(filepath)
    img_loaded = Image.open(os.path.join(app.config['imgdir'], filename))
    img_saved = img_loaded.save('Images/identification_img.png')

    # img_dim = (1, 100, 100, 3)
    img_dim = (100, 100)
    img_pred = cv2.imread('Images/identification_img.png') # Membaca file gambar yang di-inputkan
    image_shape = cv2.resize(img_pred, img_dim) # Mengatur dimensi gambar yang berupa lebar (width) & tinggi (height) gambar
    image_pred = np.expand_dims(image_shape, axis=0) # Memperluas dimensi dari gambar yang akan diprediksi menjadi 4 dimensi

    # # melakukan prediksi terhadap gambar yang sudah di-pre-processing
    model_v1_pred = model_v1.predict(image_pred)

    # Melihat hasil prediksi gambar yang telah diinisialisasikan
    pred_labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    model_v1_pred_output = pred_labels[np.argmax(model_v1_pred)]
    output_str = 'Hasil Prediksi Kelas/Kategori Gambar ini adalah ' + model_v1_pred_output
    print(output_str)
   
    if not output_str:
        output_str = "Classification cannot be complete"
        response_string = "Data cannot be processed, please check your file"
        file_status = "Cannot be processed"
        success_status = False
        details = {
            'processed_file': filename_orig,
            'response_string': response_string, 
            'file_Status': file_status, 
            'success_status': success_status,
            'result': output_str
            }
    else:
        response_string = "Process is complete"
        file_status = "Received"
        success_status = True
        details = {
            'processed_file': filename_orig,
            'response_string': response_string, 
            'file_Status': file_status, 
            'success_status': success_status,
            'result': output_str
            }
        
    return jsonify(details)

"""
function untuk menjalankan klasifikasi menggunakan model ResNet-152 V2
"""
@app.route('/resnetv2', methods=['POST'])
def resnet_V2():
    # Load Model Resnet152_v1
    """
    Memanggil model ResNet 152 v1 yang telah dilatih dengan data training
    serta diujikan kepada data testing dan untuk prediksi gambarnya
    """
    model_v1_path = 'Model/resnet152_v2.h5'
    model_v1 = load_model(model_v1_path, compile=False)

    """
    Load gambar yang dikirim dari sisi front-end
    """
    img_uploaded = request.files
    img_file = img_uploaded.get('file')
    filename_orig = secure_filename(img_file.filename) # used for make it easier to see what is the original file
    filename = 'identification_img.png' # save file 
    filepath = os.path.join(app.config['imgdir'], filename);
    img_pred1 = img_file.save(filepath)
    img_loaded = Image.open(os.path.join(app.config['imgdir'], filename))
    img_saved = img_loaded.save('Images/identification_img.png')

    # img_dim = (1, 100, 100, 3)
    img_dim = (100, 100)
    img_pred = cv2.imread('Images/identification_img.png') # Membaca file gambar yang di-inputkan
    image_shape = cv2.resize(img_pred, img_dim) # Mengatur dimensi gambar yang berupa lebar (width) & tinggi (height) gambar
    image_pred = np.expand_dims(image_shape, axis=0) # Memperluas dimensi dari gambar yang akan diprediksi menjadi 4 dimensi

    # # melakukan prediksi terhadap gambar yang sudah di-pre-processing
    model_v1_pred = model_v1.predict(image_pred)

    # Melihat hasil prediksi gambar yang telah diinisialisasikan
    pred_labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    model_v1_pred_output = pred_labels[np.argmax(model_v1_pred)]
    output_str = 'Hasil Prediksi Kelas/Kategori Gambar ini adalah ' + model_v1_pred_output
    print(output_str)
   
    if not output_str:
        output_str = "Classification cannot be complete"
        response_string = "Data cannot be processed, please check your file"
        file_status = "Cannot be processed"
        success_status = False
        details = {
            'processed_file': filename_orig,
            'response_string': response_string, 
            'file_Status': file_status, 
            'success_status': success_status,
            'result': output_str
            }
    else:
        response_string = "Process is complete"
        file_status = "Received"
        success_status = True
        details = {
            'processed_file': filename_orig,
            'response_string': response_string, 
            'file_Status': file_status, 
            'success_status': success_status,
            'result': output_str
            }
        
    return jsonify(details)
