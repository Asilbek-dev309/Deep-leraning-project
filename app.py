import streamlit as st
from fastai.vision.all import *
import pathlib
import pickle
import plotly.express as px
import platform
plt=platform.system()
if plt =='Linux': pathlib.WindowsPath=pathlib.PosixPath

st.title('Transport klassifikatsiya qilivchi model')
file=st.file_uploader("Rasm yuklash",type=['png','jpeg','gif','svg'])
if file:
  st.image(file)
  img=PILImage.create(file)
  model=load_learner('transport_model.pkl')
  prediction,pred_id,probs=model.predict(img)
  st.success(f'Bashorat:{prediction}')
  st.info(f'Ehtimollik:{probs[pred_id]*100:.1f}%')

  #plotting
  fig=px.bar(x=probs*100,y=model.dls.vocab)
  st.plotly_chart(fig)
