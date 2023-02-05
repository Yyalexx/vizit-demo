from PIL import Image, ImageDraw
import numpy as np
import torch
import pandas as pd
import streamlit as st


st.set_page_config(layout="wide", page_title="Vizit Item Counting")
st.write("## ВИЗИТ. Загрузите фотографию витрины и узнаете, сколько и каких позиций на ней.")

st.sidebar.write("## Загрузка фото :gear:")
col1, col2 = st.columns(2)
col1.write("Resized Image :camera:")
col2.write("Подсчет количества позиций по классам на фото")
my_upload = st.sidebar.file_uploader("Upload an image", type=["jpg"])
if my_upload is not None:
    input_image_path = my_upload
else:
    input_image_path = "./1.jpg"

class_list = {0:[0],
            1:[1,2],
            2:[3,4,5,6,7],
            3:[8,9]
                    }
class_dict = {
        0:'V_cl_0.45',
        1:'V_cl_1.5',
        2:'V_vech_0.45',
        3:'Nad_G_1.5',
        4:'Nad_G_2.0',
        5:'Nad_NG_1.5',
        6:'Nad_NG_2.0',
        7:'Nad_5.0',
        8:'V_cl_RF_1.5',
        9:'V_ZH_RF_1.5'
}    
    
class_list = {0:[0],
            1:[1,2],
            2:[3,4,5,6,7],
            3:[8,9]
                    }
class_dict = {
        0:'V_cl_0.45',
        1:'V_cl_1.5',
        2:'V_vech_0.45',
        3:'Nad_G_1.5',
        4:'Nad_G_2.0',
        5:'Nad_NG_1.5',
        6:'Nad_NG_2.0',
        7:'Nad_5.0',
        8:'V_cl_RF_1.5',
        9:'V_ZH_RF_1.5'
}    

color_dict = {0:(255,255,255),
              1:(255,255,0),
              2:(0,100,200),
              3:(0,255,255),
              4:(0,128,128),
              5:(128,128,255),
              6:(255,128,255),
              7:(0,128,255),
              8:(255,0,128),
              9:(0,0,0)
              
              }

col_list = list(class_dict.keys())
predict_df = pd.DataFrame(columns=['Класс','Количество'])

original_image = Image.open(input_image_path)
resized_image = original_image.resize((640, 640), Image.LANCZOS)

if hasattr(original_image, '_getexif') or original_image._getexif() is not None:
    orientation = original_image._getexif().get(0x112)
    rotate_values = {3: 180, 6: 270, 8: 90}
    if orientation in rotate_values:
        img = resized_image.rotate(rotate_values[orientation])
    else:
        img = resized_image 
else:
    img = resized_image


m_1 = torch.hub.load('.', 'custom', path='./model_cl_0.pt', 
            source='local', device='cpu', force_reload=True, _verbose=False)
m_2 = torch.hub.load('.', 'custom', path='./m_cl_1_2.pt', 
            source='local', device='cpu', force_reload=True, _verbose=False)
m_3 = torch.hub.load('.', 'custom', path='./water.pt', 
            source='local', device='cpu', force_reload=True, _verbose=False)
m_4 = torch.hub.load('.', 'custom', path='./rf.pt', 
            source='local', device='cpu', force_reload=True, _verbose=False)
models = {0:m_1, 1:m_2, 2:m_3, 3:m_4}


draw = ImageDraw.Draw(img)
curr_cl_list = []
d={}

for m in class_list.keys():   # для каждой модели
    model = models[m]
    results = model(img, size=640)
    result_df = model(img, size=640).pandas().xyxy[0]
    class_df = result_df.groupby('class').agg('count')['name']

    if len(result_df) != 0:
        # Наносим рамки на изображение
        for i in range(len(result_df)):
            curr_cl = result_df.iloc[i][5]
            glob_cl = curr_cl + class_list[m][0] # + смещение
            curr_lbl = tuple(result_df.iloc[i].values[:4])
            draw.rectangle(curr_lbl, outline=color_dict[glob_cl], width=2 )
        for cls in list(class_df.index):
            d[cls+class_list[m][0]] = class_df.loc[cls]

for k in col_list:
    if k in d.keys():
        curr_cl_list = [k, d[k]]
    else:
        curr_cl_list = [k, 0]
    predict_df.loc[len(predict_df.index)] = curr_cl_list

col1.image(img)

predict_df['Наименование'] =  predict_df['Класс'].apply(lambda x: class_dict[x])
predict_df.drop(['Класс'], axis=1, inplace=True)
col2.dataframe(predict_df)
