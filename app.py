import streamlit as st
import face_recognition
import cv2
import numpy as np

st.title("نظام ماسح الوجوه 🕵️‍♂️")

# قائمة الصور التي رفعتها على جيت هاب
known_images = ["Ali.jpg", "Bouhalfa.jpg", "Kamel.jpg", "Mabrouk.jpg", "Salim.jpg", "Youcef.jpg"]
known_face_encodings = []
known_face_names = []

# تحميل وتشفير الوجوه
for img_path in known_images:
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(img_path.split(".")[0])

img_file_buffer = st.camera_input("التقط صورة للشخص")

if img_file_buffer is not None:
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "شخص مجهول"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            st.success(f"تم التعرف على: {name}")
        else:
            st.warning("هذا الشخص غير مسجل لدينا")
