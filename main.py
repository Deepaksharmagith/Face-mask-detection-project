import streamlit as st
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array,load_img
import cv2
st.set_page_config(page_title="Mask Detection",page_icon="https://cdn-icons-png.flaticon.com/512/5985/5985970.png")
facemodel=cv2.CascadeClassifier("face.xml")
maskmodel=load_model("mask.h5")
st.title("FASK MASK DETECTION SYSTEM")
st.sidebar.image("https://cdn.hackernoon.com/images/oO6rUouOWRYlzw88QM9pb0KyMIJ3-bxfy3m27.png")
choice=st.sidebar.selectbox("MENU",("HOME","IMAGE","URL/IP CAM","WEB CAM"))
if(choice=="HOME"):
    st.image("http://via-vis.com/wp-content/uploads/2020/08/Mask-Detection-gif-1.gif")
    st.write("Face Mask Detection System is a Computer Vision Machine Learning Application which can be accessed through IP Cameras and can detect whether the person is wearing a mask or not.")
elif(choice=="IMAGE"):
    st.markdown('<center><h2>IMAGE DETECTION</h2></center>',unsafe_allow_html=True)
    file=st.file_uploader("Upload an Image")
    if file:
        b=file.getvalue()
        a=np.frombuffer(b,np.uint8)
        img=cv2.imdecode(a,cv2.IMREAD_COLOR)
        face=facemodel.detectMultiScale(img)
        for (x,y,l,w) in face:
            cv2.imwrite("temp.jpg",img[y:y+w,x:x+l])
            face_img=load_img('temp.jpg',target_size=(150,150,3))
            face_img=img_to_array(face_img)
            face_img=np.expand_dims(face_img,axis=0)
            pred=maskmodel.predict(face_img)[0][0]
            if(pred==1):
                cv2.rectangle(img,(x,y),(x+l,y+w),(0,0,255),8)
            else:
                cv2.rectangle(img,(x,y),(x+l,y+w),(0,255,0),8)
        st.image(img,channels='BGR')
elif(choice=='WEB CAM'):
    k=st.text_input("Enter 0 for Primary Camera or 1 for Secondary Camera") 
    btn=st.button('Start Camera')
    if btn:
        window=st.empty()               
        k=int(k)
        vid=cv2.VideoCapture(k)
        btn2=st.button("Stop Camera")
        if(btn2):
            vid.release()
            st.experimental_rerun()
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                face=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in face:
                    cv2.imwrite("temp.jpg",frame[y:y+w,x:x+l])
                    face_img=load_img('temp.jpg',target_size=(150,150,3))
                    face_img=img_to_array(face_img)
                    face_img=np.expand_dims(face_img,axis=0)
                    pred=maskmodel.predict(face_img)[0][0]
                    if(pred==1):
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),8)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),8)            
                window.image(frame,channels='BGR')
elif(choice=='URL/IP CAM'):
    k=st.text_input("Enter URL for the video") 
    btn=st.button('Start Camera')
    if btn:
        window=st.empty()      
        vid=cv2.VideoCapture(k)
        btn2=st.button("Stop Camera")
        if(btn2):
            vid.release()
            st.experimental_rerun()
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                face=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in face:
                    cv2.imwrite("temp.jpg",frame[y:y+w,x:x+l])
                    face_img=load_img('temp.jpg',target_size=(150,150,3))
                    face_img=img_to_array(face_img)
                    face_img=np.expand_dims(face_img,axis=0)
                    pred=maskmodel.predict(face_img)[0][0]
                    if(pred==1):
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),8)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),8)            
                window.image(frame,channels='BGR')







