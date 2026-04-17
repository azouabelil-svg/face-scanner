import streamlit as st
import cv2
from deepface import DeepFace
import os
from PIL import Image
import numpy as np

# 1. قاعدة بيانات بانيو - ولاية المسيلة
persons_info = {
    "Ali": {"full_name": "Belil Ali", "birth": "1967", "age": "59", "address": "Baniou, Maarif, M'sila"},
    "Kamel": {"full_name": "Belil Kamel", "birth": "1972", "age": "54", "address": "Baniou, Maarif, M'sila"},
    "Bouhalfa": {"full_name": "Belil Mabrouk", "birth": "1984", "age": "42", "address": "Baniou, Maarif, M'sila"},
    "Salim": {"full_name": "Selmani Salim", "birth": "1982", "age": "44", "address": "Baniou, Maarif, M'sila"},
    "Youcef": {"full_name": "Belil Youcef", "birth": "1992", "age": "34", "address": "Baniou, Maarif, M'sila"}
}

st.title("نظام بليل الذكي لكشف الوجوه 🇩🇿")
st.write("بلدية المعاريف - ولاية المسيلة")

uploaded_file = st.file_uploader("ارفع صورة الشخص أو التقطها بالكاميرا", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # معالجة الصورة
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite("temp.jpg", img_bgr)
    
    with st.spinner('جاري فحص الوجه ومطابقته...'):
        try:
            # البحث عن الشخص في مجلد الصور
            results = DeepFace.find(img_path="temp.jpg", db_path="./my_faces/", model_name='VGG-Face', enforce_detection=False)
            
            person_name = "Unknown"
            if len(results) > 0 and not results[0].empty:
                matched_path = results[0].iloc[0]['identity']
                person_name = os.path.basename(matched_path).split('.')[0]
            
            data = persons_info.get(person_name, {"full_name": "غير مسجل", "age": "??", "address": "غير معروف"})
            
            # رسم المربع والبيانات
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 10)
                st.success(f"تم التعرف على الشخص: {data['full_name']}")
                st.info(f"العمر: {data['age']} سنة | السكن: {data['address']}")
            
            st.image(img_array, caption="نتيجة الماسح الذكي")
            
        except Exception as e:
            st.error("حدث خطأ أثناء المعالجة، يرجى التأكد من وضوح الوجه.")
