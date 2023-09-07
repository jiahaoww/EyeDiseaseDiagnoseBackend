from flask import Flask, request, send_file, make_response
import methods
import os.path
from PIL import Image
from werkzeug.utils import secure_filename
import requests
from datetime import datetime
import DRModel.DRModelMethods as DRModelMethods
import multiModel.multiModelMethods as multiModelMethods

image_dir = './image/'
jpg_dir ='./jpeg_temp/'
res_path = './result/res.txt'
HMWebService_URL = 'http://192.168.186.61:9001/HMWebService.asmx/Execute1'

app = Flask(__name__)
allow_headers = "Origin, Expires, Content-Type, X-E4M-With, Authorization"
http_response_header = {
   "Access-Control-Allow-Origin": "*",
   "Access-Control-Allow-Headers": allow_headers
}

disease_abbr = ['正常', '糖网', '老黄', '静阻', '高度近视']

disease_fullname = ['正常', '糖尿病视网膜病变', '老年性黄斑变性', '视网膜静脉阻塞', '高度近视']

dr_types = ['NPDR', 'PDR', '正常', '玻璃体出血']

multi_types = ['糖网', '老黄', '高度近视', '静阻', '青光眼', '正常']

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
   return {"list": list, "classes": disease_abbr}, 200, http_response_header

@app.route("/savePredictResult", methods=['GET', 'OPTIONS'])
def save_predict_result():
   global res_path
   userName = request.cookies.get('CAD_USER')
   file = open(res_path, 'a')
   formatted_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
   file.write(f"{request.args['patientId']}:{request.args['patientName']}:{request.args['predict']}:{userName}:{formatted_time}\n")
   file.close()
   login_response_header = {
      "Access-Control-Allow-Headers": allow_headers,
      "Access-Control-Allow-Credentials": "true"
   }
   origin = request.headers.get('Origin')
   login_response_header['Access-Control-Allow-Origin'] = origin
   return {"code": 200}, 200, login_response_header

@app.route("/getPredictResult", methods=['GET', 'OPTIONS'])
def get_predict_result():
   global res_path
   file = open(res_path, 'r')
   lines = file.readlines()
   list = []
   for line in lines:
      strArr = line.split(':')
      patientId = strArr[0]
      patientName = strArr[1]
      predict = strArr[2]
      userName = strArr[3]
      time = strArr[4].rstrip('\n')
      record = {
         "patientId": patientId,
         "patientName": patientName,
         "predict": predict,
         "userName": userName,
         "time": time,
      }
      list.append(record)
   file.close()
   return {"code": 200, "list": list}, 200, http_response_header


@app.route("/upload", methods=['POST', 'OPTIONS'])
def handle_upload_image():
   if request.method == 'POST':
      f = request.files['image']
      name = secure_filename(f.filename)
      path = f'./uploaded/{name}'
      f.save(f'./uploaded/{name}')
      list = methods.predict(path)
      return {"list": list, "classes": disease_fullname}, 200, http_response_header
   return {"code": 200}, 200, http_response_header

@app.route("/DRType", methods=['POST', 'OPTIONS'])
def handle_upload_dr_image():
   if request.method == 'POST':
      f = request.files['image']
      name = secure_filename(f.filename)
      path = f'./uploaded/{name}'
      f.save(f'./uploaded/{name}')
      res = methods.getDRType(path)
      return {"DRType": dr_types[res]}, 200, http_response_header
   return {"code": 200}, 200, http_response_header

@app.route("/multiModel", methods=['POST', 'OPTIONS'])
def handle_upload_multi_image():
   if request.method == 'POST':
      f = request.files['image1']
      name1 = secure_filename(f.filename)
      path1 = f'./uploaded/{name1}'
      f.save(path1)
      f = request.files['image2']
      name2 = secure_filename(f.filename)
      path2 = f'./uploaded/{name2}'
      f.save(path2)
      res = multiModelMethods.get_OCT_FUNDUS_Type([path1, path2])
      return {"type": multi_types[res]}, 200, http_response_header
   return {"code": 200}, 200, http_response_header

@app.route("/login", methods=['POST', 'OPTIONS'])
def handle_user_login():
   userName = request.form['userName']
   password = request.form['password']
   file = open("./users/record.txt", "r")
   lines = file.readlines()
   account = {}
   for line in lines:
      strArr = line.split(':')
      _userName = strArr[0]
      _password = strArr[1].rstrip('\n')
      account[_userName] = _password
   file.close()

   login_response_header = {
      "Access-Control-Allow-Headers": allow_headers,
      "Access-Control-Allow-Credentials": "true"
   }
   origin = request.headers.get('Origin')
   login_response_header['Access-Control-Allow-Origin'] = origin

   if userName in account:
      if password != account[userName]:
         return {"code": 208, "msg": "密码错误"}, 208, login_response_header
   else: 
      return {"code": 208, "msg": "该用户未注册"}, 208, login_response_header

   resp = make_response({"code": 200, "msg": "登录成功" })
   resp.set_cookie('CAD_USER', userName, max_age= 24 * 3600 * 365, httponly=True)
   return resp, 200, login_response_header

@app.route("/logout", methods=['GET', 'OPTIONS'])
def handle_user_logout():
   login_response_header = {
      "Access-Control-Allow-Headers": allow_headers,
      "Access-Control-Allow-Credentials": "true"
   }
   origin = request.headers.get('Origin')
   login_response_header['Access-Control-Allow-Origin'] = origin

   resp = make_response({"code": 200, "msg": "登出成功" })
   resp.set_cookie('CAD_USER', '', max_age= 24 * 3600 * 365, httponly=True)
   return resp, 200, login_response_header

@app.route("/register", methods=['POST', 'OPTIONS'])
def handle_user_register():
   userName = request.form['userName']
   password = request.form['password']
   file = open("./users/record.txt", "r")
   lines = file.readlines()
   account = {}
   for line in lines:
      strArr = line.split(':')
      _userName = strArr[0]
      _password = strArr[1].rstrip('\n')
      account[_userName] = _password
   file.close()
   if userName in account:
      return {"code": 208, "msg": "该用户名已被注册过"}, 208, http_response_header
   file = open("./users/record.txt", "a")
   file.write(f"{userName}:{password}\n")
   return {"code": 200, "msg": "注册成功" }, 200, http_response_header

@app.route("/reset", methods=['POST', 'OPTIONS'])
def handle_user_reset():
   userName = request.form['userName']
   password = request.form['password']
   file = open("./users/record.txt", "r")
   lines = file.readlines()
   account = {}
   for line in lines:
      strArr = line.split(':')
      _userName = strArr[0]
      _password = strArr[1].rstrip('\n')
      account[_userName] = _password
   file.close()
   if not userName in account:
      return {"code": 208, "msg": "请检查用户名填写是否正确"}, 208, http_response_header
   account[userName] = password
   file = open("./users/record.txt", "w")
   for _userName, _password in account.items():
      file.write(f"{_userName}:{_password}\n")
   return {"code": 200, "msg": "重置密码成功" }, 200, http_response_header

@app.route("/modify", methods=['POST', 'OPTIONS'])
def handle_user_modify():
   userName = request.form['userName']
   password = request.form['password']
   file = open("./users/record.txt", "r")
   lines = file.readlines()
   account = {}
   for line in lines:
      strArr = line.split(':')
      _userName = strArr[0]
      _password = strArr[1].rstrip('\n')
      account[_userName] = _password
   file.close()
   if not userName in account:
      return {"code": 208, "msg": "请检查用户名填写是否正确"}, 208, http_response_header
   account[userName] = password
   file = open("./users/record.txt", "w")
   for _userName, _password in account.items():
      file.write(f"{_userName}:{_password}\n")
   return {"code": 200, "msg": "修改密码成功, 请重新登录" }, 200, http_response_header

@app.route("/checkLogin", methods=['GET', 'OPTIONS'])
def check_login():
   login_response_header = {
      "Access-Control-Allow-Headers": allow_headers,
      "Access-Control-Allow-Credentials": "true"
   }
   origin = request.headers.get('Origin')
   login_response_header['Access-Control-Allow-Origin'] = origin
   userName = request.cookies.get('CAD_USER')
   if userName and len(userName) > 0:
      return {"code": 200, "msg": "已登录", "userName": userName }, 200, login_response_header
   else:
      return {"code": 208, "msg": "未登录" }, 208, login_response_header