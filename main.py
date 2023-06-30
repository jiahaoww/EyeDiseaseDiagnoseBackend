from flask import Flask, request, send_file
import methods
import os.path
from PIL import Image
from werkzeug.utils import secure_filename
import requests

image_dir = './image/'
jpg_dir ='./jpeg_temp/'
res_path = './result/res.txt'
production_mode = True
HMWebService_URL = 'http://192.168.186.61:9001/HMWebService.asmx/Execute1' if production_mode == True else 'http://127.0.0.1:5000/Execute1'
# 'http://192.168.186.61:9001/HMWebService.asmx/Execute1'
print(HMWebService_URL)

app = Flask(__name__)
allow_headers = "Origin, Expires, Content-Type, X-E4M-With, Authorization"
http_response_header = {"Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": allow_headers}

@app.route("/patientList", methods=['GET', 'OPTIONS'])
def get_patient_list():
   paramXml = request.args['xml']
   payload = {'cmd': 'GetExamPatList', 'strXmlorJson': paramXml}
   r = requests.get(HMWebService_URL, params = payload)
   return r.text, 200, http_response_header


@app.route("/patientReport", methods=['GET', 'OPTIONS'])
def get_patient_report_by_id():
   paramXml = request.args['xml']
   payload = {'cmd': 'GetPatExamRpt', 'strXmlorJson': paramXml}
   r = requests.get(HMWebService_URL, params = payload)
   return r.text, 200, http_response_header

    
@app.route("/fetchImage", methods=['GET', 'OPTIONS'])
def fetch_image_content():
   imageUrl = request.args['url']
   filename = secure_filename(imageUrl)
   path = image_dir + filename
   if (not os.path.isfile(path)):
    print('download')
    # if it is not downloaded yet, download the image at first
    response = requests.get(imageUrl)
    if response.status_code == 200:
       with open(path, 'wb') as f:
          f.write(response.content)

   name = os.path.splitext(filename)[0]
   ext = os.path.splitext(filename)[1]

   # if it is tiff format, it should be transformed to jpg format
   send_file_path = ''
   if ext == '.tiff' or ext == '.tif':
      jpg_file_path = os.path.join(jpg_dir + name) + '.jpg'
      if (not os.path.isfile(jpg_file_path)):
        print('convert')
        im = Image.open(path)
        im.thumbnail(im.size)
        im.save(jpg_file_path, "JPEG", quality=100)
      send_file_path = jpg_file_path
   else:
      send_file_path = path
   return send_file(send_file_path), 200, http_response_header

@app.route("/heatmapImage", methods=['GET', 'OPTIONS'])
def fetch_heatmap_image_content(): 
   imageUrl = request.args['url']
   filename = secure_filename(imageUrl)
   name = os.path.splitext(filename)[0]
   heatmap_file_path = methods.heatmap_path + name + '.jpg'
   if (not os.path.isfile(heatmap_file_path)):
      methods.generate_heatmap_image(filename)
   return send_file(heatmap_file_path), 200, http_response_header

@app.route("/predictImage", methods=['GET', 'OPTIONS'])
def predict_image():
   imageUrl = request.args['url']
   filename = secure_filename(imageUrl)
   image_path = image_dir + filename
   list = methods.predict(image_path)
   return {"list": list, "classes": ['正常', '糖网', '老黄', '静阻', '高度近视']}, 200, http_response_header

@app.route("/savePredictResult", methods=['GET', 'OPTIONS'])
def save_predict_result():
   global res_path
   res_file = open(res_path, 'a')
   print(request.args['patientId'], ' : ', request.args['predict'], file = res_file)
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