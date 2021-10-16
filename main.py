from flask import Flask, jsonify, request, render_template, url_for
import os, torch
from pathlib import Path
from models_ import get_densenet_121
from utils_ import get_class
from datasets import get_test_transform
import random

from detect import get_model, predict
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

device = torch.device('cpu')
save_dir = Path('outputs/')
save_img = True
save_txt = True
imgsz = 128
conf_thres = 0.25  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000
model, stride, names = get_model(os.path.join('weights', 'best.pt'), torch.device('cpu'))

val_transform = get_test_transform()
m = get_densenet_121('cpu', 'checkpoints/DENSE2(128,128).ckpt')


def get_random_name():
    l = 64
    name = ''.join([chr(random.randint(97, 123)) for _ in range(l)])
    return name


@app.route('/predict', methods=['POST'])
def predict_():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        source = os.path.join('images', get_random_name() + '.jpg')
        file.save(source)
        if request.form['model'] == 'classification':
            out = get_class(source, m, transform=val_transform)
            return jsonify({'label': out})
        elif request.form['model'] == 'detection':
            out = predict(model, stride, names, source, device, save_dir, save_img, save_txt, imgsz, conf_thres,
                          iou_thres, max_det)
            return jsonify({'bbox': out})


@app.route('/info')
def info():
    return render_template('main.html')

@app.route('/main')
def main_html():
    return render_template('index.html')

@app.route('/app')
def app_html():
    return render_template('app.html')

if __name__ == '__main__':
    app.run()
