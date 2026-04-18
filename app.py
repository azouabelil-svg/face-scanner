import streamlit as st
import face_recognition
import cv2
import numpy as np
import os

st.title("نظام التعرف على الوجوه الذكي")

# قائمة الصور والأسماء بناءً على الملفات في مستودعك
images_files = [
    {"name": "Ali", "file": "Ali.jpg"},
    {"name": "Bouhalfa", "file": "Bouhalfa.jpg"},
    {"name": "Kamel", "file": "Kamel.jpg"},
    {"name": "Mabrouk", "file": "Mabrouk.jpg"},
    {"name": "Salim", "file": "Salim.jpg"},
    {"name": "Youcef", "file": "Youcef.jpg"}
]

known_face_encodings = []
known_face_names = []

# تحميل الصور وتشفيرها
for person in images_files:
    if os.path.exists(person["file"]):
        image = face_recognition.load_image_file(person["file"])
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(person["name"])

st.success(f"تم تحميل {len(known_face_names)} وجوه معروفة بنجاح!")

# خيار استخدام الكاميرا
img_file_buffer = st.camera_input("التقط صورة للتعرف على صاحبها")

if img_file_buffer is not None:
    # تحويل الصورة إلى تنسيق يفهمه OpenCV
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    # البحث عن الوجوه في الصورة الملتقطة
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "شخص غير معروف"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        st.write(f"النتيجة: **{name}**")
