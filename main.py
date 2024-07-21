import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np 
import tensorflow as tf 
from http import HTTPStatus
from PIL import Image
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google.cloud import storage
import random

load_dotenv()
app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['png','jpg','jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_CLASSIFICATION'] = 'models/my_model.h5'
app.config['GCS_CREDENTIALS'] = './credentials/gcs.json'

model = tf.keras.models.load_model(app.config['MODEL_CLASSIFICATION'],compile=False)

bucket_name = os.environ.get('BUCKET_NAME','dauryuk-ml-bucket')
client = storage.Client.from_service_account_json(json_credentials_path=app.config['GCS_CREDENTIALS'])
bucket = storage.Bucket(client,bucket_name)
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1] in app.config['ALLOWED_EXTENSIONS']

# index [i][0] nama class index [i][1] deskripsi 
classess = [
    ['Cardboard',"To dispose of cardboard waste, flatten the boxes and place them in your recycling bin or take them to a local recycling center. Ensure the cardboard is clean and free from contaminants like food residue or tape before recycling."],
    ['Food Organics',"Dispose of food organics waste by placing it in a designated compost bin or green waste bin. Ensure that the waste is free from plastic, glass, and other non-organic materials to facilitate effective composting."],
    ['Glass',"To dispose of glass waste, first, ensure it is clean and free of any contaminants such as food residue or liquids. Then, place the glass in a designated recycling bin specifically for glass materials, or take it to a local recycling facility that accepts glass."],
    ['Metal',"To dispose of metal waste responsibly, first, ensure it's clean and free of any contaminants. Then, take it to a recycling center where it can be properly sorted and processed for reuse, reducing environmental impact and conserving valuable resources."],
    ['Miscellaneous Trash',"To dispose of miscellaneous trash waste, ensure it is properly separated from recyclables and hazardous materials. Then, either place it in designated trash bins for collection or take it to a nearby waste disposal facility for proper disposal."],
    ['Paper',"To dispose of paper waste responsibly, begin by sorting it into recyclable and non-recyclable categories. Recyclable paper can be placed in designated recycling bins, while non-recyclable paper should be securely bagged and disposed of in appropriate waste receptacles to minimize environmental impact."],
    ['Plastic',"To dispose of plastic waste responsibly, begin by segregating it from other types of waste to facilitate recycling. Then, either deposit it in designated recycling bins or take it to a nearby recycling facility, where it can be processed and repurposed into new products, minimizing its environmental impact."],
    ['Textile Trash',"To properly dispose of textile trash waste, begin by separating it from other recyclables. Next, consider donating usable items to thrift stores or textile recycling centers. For items that cannot be reused, explore local textile recycling programs or facilities that can process them responsibly, minimizing environmental impact."]
]

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'Message': 'Connected',
        'Data': {
            'Project': 'DaurYuk: Sorting Waste for a Sustainable Future!',
            'Team': 'C241-PS480',
            'Anggota': [
                { 'BangkitID': 'M004D4KX1508', 'Nama': 'Annisa Mufidatun Sholihah', 'Universitas': 'Institut Teknologi Sepuluh Nopember', 'Role' : 'Machine Learning Developer' },
                { 'BangkitID': 'M131D4KX1688', 'Nama': 'Farah Diva Nabila', 'Universitas': 'Politeknik Negeri Malang' ,'Role' : 'Machine Learning Developer'},
                { 'BangkitID': 'M006D4KX3150', 'Nama': 'Ni Putu Eka Dwi Yantii', 'Universitas': 'Universitas Brawijaya','Role' : 'Machine Learning Developer' },
                { 'BangkitID': 'C006D4KY0595', 'Nama': 'Ananda Fitra Diraja', 'Universitas': 'Universitas Brawijaya','Role' : 'Cloud Computing Developer' },
                { 'BangkitID': 'C007D4KY0712', 'Nama': 'Calvin Revianto', 'Universitas': 'Universitas Dian Nuswantoro','Role' : 'Cloud Computing Developer' },
                { 'BangkitID': 'A524D4KY4343', 'Nama': 'Muhammad Rizal Wahyudi', 'Universitas': 'Politeknik Negeri Banjarmasin','Role' : 'Mobile Developer' },
                { 'BangkitID': 'A007D4KY4437', 'Nama': 'Naufal Maldini', 'Universitas': 'Universitas Dian Nuswantoro','Role' : 'Mobile Developer' }
            ],         
    }}),HTTPStatus.OK
    

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        reqImage = request.files['image']
        if reqImage and allowed_file(reqImage.filename):
            filename =  secure_filename(reqImage.filename)
            reqImage.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            imagePath =  os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = Image.open(imagePath).convert('RGB')
            img = img.resize((300,300))
            normalizationImage = tf.keras.preprocessing.image.img_to_array(img)
            normalizationImage = np.expand_dims(normalizationImage,axis=0)
            normalizationImage = normalizationImage / 255
            predictResult = model.predict(normalizationImage)
            imageName = imagePath.split('/')[-1]
            blobPath = 'images/'+ str(random.randint(10000,99999))+imageName
            blob = bucket.blob(blobPath)
            blob.upload_from_filename(imagePath)
            os.remove(imagePath)
            result = {
                'predict' : classess[np.argmax(predictResult)][0],
                'description' : classess[np.argmax(predictResult)][1],
                'image' : 'https://storage.googleapis.com/'+'BUCKET_NAME'+ blobPath
            }
            return jsonify({
                            'status': {
                                'code': HTTPStatus.OK,
                                'message': 'Success predicting',
                            },
                            'data': result,
                        }),HTTPStatus.OK,
                
        else:
            return jsonify({
                            'status': {
                                'code': HTTPStatus.BAD_REQUEST,
                                'message': 'Invalid file format. Please upload a JPG, PNG, or JPEG image',
                            }
                            }),HTTPStatus.BAD_REQUEST,
                
    else:
        return jsonify({
                        'status': {
                            'code': HTTPStatus.METHOD_NOT_ALLOWED,
                            'message': 'Methode not allowed',
                        }
                        }),HTTPStatus.METHOD_NOT_ALLOWED,


if __name__ == "__main__":
    app.run()