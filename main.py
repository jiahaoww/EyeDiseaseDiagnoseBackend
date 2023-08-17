from flask import Flask, request, send_file
import methods
import os.path
from PIL import Image
from werkzeug.utils import secure_filename
import requests

image_dir = './image/'
jpg_dir ='./jpeg_temp/'
res_path = './result/res.txt'
production_mode = false

app = Flask(__name__)
allow_headers = "Origin, Expires, Content-Type, X-E4M-With, Authorization"
http_response_header = {"Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": allow_headers}

@app.route("/upload", methods=['POST', 'OPTIONS'])
def handle_upload_image():
   if request.method == 'POST':
      f = request.files['image']
      name = secure_filename(f.filename)
      path = f'./uploaded/{name}'
      f.save(f'./uploaded/{name}')
      list = methods.predict(path)
      return {"list": list, "classes": ['正常', '糖网', '老黄', '静阻', '高度近视']}, 200, http_response_header
   return {"code": 200}, 200, http_response_header