from PIL import Image
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

model_1_class_list = [0]
model_2_class_list = [1,2]

col_list = model_1_class_list + model_2_class_list

predict_df = pd.DataFrame(columns=col_list)
predict_df = pd.DataFrame(columns=['Класс','Количество'])

original_image = Image.open(input_image_path)
resized_image = original_image.resize((640, 640), Image.LANCZOS)
try:    
    if hasattr(original_image, '_getexif') or original_image._getexif() is not None:
        orientation = original_image._getexif().get(0x112)
        rotate_values = {3: 180, 6: 270, 8: 90}
        if orientation in rotate_values:
            img = resized_image.rotate(rotate_values[orientation])
        else:
            img = resized_image 
except:
    img = resized_image
col1.image(img)

model_1 = torch.hub.load('.', 'custom', path='./model_cl_0.pt', 
            source='local', device='cpu', force_reload=True, _verbose=False)
model_2 = torch.hub.load('.', 'custom', path='./m_cl_1_2.pt', 
            source='local', device='cpu', force_reload=True, _verbose=False)

results = model_1(img, size=640) # ЗАГАДКА, без этой строчки неустойчиво

d={}
results = model_1(img, size=640)
col_class=results.xyxy[0][:,5]

for cls in model_1_class_list:
    d[cls] = len(np.where(col_class==cls)[0])
    curr_cl_list = [cls, d[cls]]
    predict_df.loc[len(predict_df.index)] = curr_cl_list


results = model_2(img, size=640)
col_class=results.xyxy[0][:,5]

for cls in model_2_class_list:
    d[cls] = len(np.where(col_class==cls-len(model_1_class_list))[0])
    curr_cl_list = [cls, d[cls]]
    predict_df.loc[len(predict_df.index)] = curr_cl_list

mapper = {0:'Визит классический 0.45', 
          1:'Визит классический ПЭТ', 
          2:'Визит вечерний 0.45'}
predict_df['Наименование'] =  predict_df['Класс'].apply(lambda x: mapper[x])
predict_df.drop(['Класс'], axis=1, inplace=True)
col2.dataframe(predict_df)
