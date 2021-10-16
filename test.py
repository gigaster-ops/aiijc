import requests
import json

json_ = {
    'model': 'detection'
}

files = {'file': open('bd01c6bd-20a6-461c-880a-0dae75b9d4a1_png.rf.8b9e40268d84e3ae2bb1222ed61e6037.jpg','rb')}
q = json.loads(requests.post('http://01a4-37-146-116-128.ngrok.io/predict', data=json_, files=files).text)
#q = requests.post('http://01a4-37-146-116-128.ngrok.io/predict',files=files).text
print(q)