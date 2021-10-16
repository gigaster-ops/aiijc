import numpy as np
import pandas as pd
import torch
from PIL import Image

from constants import *
from datasets import get_label_replacers

parsed1 = pd.read_csv('data/signs.csv', delimiter=';', index_col='sign')
parsed2 = pd.read_csv('data/signs2.csv', delimiter=';', index_col='sign')

trans = {'2_3_L': '2_3_3', '2_3_R': '2_3_2', '5_19': '5_19_1', '4_8_2': '4_8_1', "4_8_3": '4_8_1', "4_8_4": '4_8_1',
         "4_8_5": '4_8_1', "4_8_6": '4_8_1', '4_1_2_1': '4_1_2', '4_1_3_1': '4_1_3'}

d = {'1': 'Предупреждающие знаки', '2': 'Знаки приоритета', '3': 'Запрещающие знаки', '4': 'Предписывающие знаки',
     '5': 'Знаки особых предписаний', '6': 'Информационные знаки', '7': 'Знаки сервиса'}


def get_class(path: str, model, model2=None, ckpt_path: str = None, transform=None) -> str:
    """Returns class of image predicted by the model"""
    label2int, int2label = get_label_replacers(TRAIN_DATAFRAME_PATH)
    x = Image.open(path)
    x = x.convert('RGB')
    if transform:
        x = transform(x)
    x = x[np.newaxis, :]
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        pred = model(x)
        if model2:
            pred2 = model2(x)
            pred += pred2
    return int2label[pred.argmax(1).numpy()[0]]


def get_all(f):
    """Returns information about predicted class of traffic sign"""
    if f == '3_17':
        return 'Знак 3.17.1-3.17.3: \nТаможня-Опасность-Контроль', 'https://media.am.ru/pdd/sign/m/3.17.3.png', 'https://www.drom.ru/pdd/pdd/signs/#65400', '3. Запрещающие знаки'
    if f == 'smoke':
        return "Знак: \nКурение запрещено-Пользование открытым огнем и курение запрещено", 'http://www.kuzalians.ru/images/ooo_price/e919330e-32cd-11e1-bc36-0025b3ad0991_1.jpg', 'http://87.rospotrebnadzor.ru/index.php/deyatelnost/zashchita-prav-potrebitelej/494-trebovaniya-k-znaku-o-zaprete-kureniya', 'Просто знак'
    if f == 'unknown':
        return "Тут ничего нет", 'https://sundownaudio.ru/upload/iblock/150/150f20206310ac30be38e1204dc24ab8.jpg', 'https://www.google.com/search?q=%D0%B8%D0%B7%D0%B2%D0%B8%D0%BD%D0%B8%D1%82%D0%B5,+%D0%BD%D0%BE+%D0%BF%D0%BE%D0%B4%D1%80%D0%BE%D0%B1%D0%BD%D0%B5%D0%B5+%D0%BD%D0%B5%D0%BA%D1%83%D0%B4%D0%B0', 'Да нету тут знака'
    if f in trans:
        f = trans[f]
    if '+' in f:
        f = f.split('+')[0]
    f = f.replace('_', '.') + '.'
    if f in parsed2.index:
        return parsed2.loc[f].title, parsed2.loc[f].img, parsed1.loc[f].link, f[0] + '. ' + d[f[0]]
