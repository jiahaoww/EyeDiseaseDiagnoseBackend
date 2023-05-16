from flask import Flask, request, send_file
import methods
import os.path
from PIL import Image
from werkzeug.utils import secure_filename

image_dir = './image/'
jpg_dir ='./jpeg_temp/'
res_path = './result/res.txt'

app = Flask(__name__)
allow_headers = "Origin, Expires, Content-Type, X-E4M-With, Authorization"
http_response_header = {"Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": allow_headers}

@app.route("/images", methods = ['GET', 'OPTIONS'])
def get_image_list():
    if request.method == 'GET':
      list = methods.imageList()
      return {"code": 200, "data": list}, 200, http_response_header
    if request.method == 'OPTIONS':
      return {"str": "ok"}, 202, http_response_header
    
@app.route("/fetchImage/<imageName>", methods=['GET', 'OPTIONS'])
def fetch_image_content(imageName):
   name = os.path.splitext(imageName)[0]
   ext = os.path.splitext(imageName)[1]
   file_path = ''
   if ext == '.tiff' or ext == '.tif':
      jpg_file_path = os.path.join(jpg_dir + name) + '.jpg'
      if (not os.path.isfile(jpg_file_path)) :
        im = Image.open(os.path.join(image_dir + imageName))
        im.thumbnail(im.size)
        im.save(jpg_file_path, "JPEG", quality=100)
      file_path = jpg_file_path
   else:
      file_path = image_dir + imageName
   return send_file(file_path), 200, http_response_header

@app.route("/heatmapImage/<imageName>", methods=['GET', 'OPTIONS'])
def fetch_heatmap_image_content(imageName): 
   name = os.path.splitext(imageName)[0]
   heatmap_file_path = methods.heatmap_path + name + '.jpg'
   if (not os.path.isfile(heatmap_file_path)):
      methods.generate_heatmap_image(imageName)
   return send_file(heatmap_file_path), 200, http_response_header

@app.route("/predictImage/<imageName>", methods=['GET', 'OPTIONS'])
def predict_image(imageName):
   image_path = image_dir + imageName
   list = methods.predict(image_path)
   return {"list": list, "classes": ['正常', '糖网', '老黄', '静阻', '高度近视']}, 200, http_response_header

@app.route("/savePredictResult", methods=['GET', 'OPTIONS'])
def save_predict_result():
   global res_path
   res_file = open(res_path, 'a')
   print(request.args['imageName'], ' : ', request.args['predict'], file = res_file)
   res_file.close()
   return {"code": 200}, 200, http_response_header

@app.route("/upload", methods=['POST', 'OPTIONS'])
def handle_upload_image():
   if request.method == 'POST':
      print('POST')
      f = request.files['image']
      name = secure_filename(f.filename)
      path = f'./uploaded/{name}'
      f.save(f'./uploaded/{name}')
      list = methods.predict(path)
      return {"list": list, "classes": ['正常', '糖网', '老黄', '静阻', '高度近视']}, 200, http_response_header
   return {"code": 200}, 200, http_response_header