import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# استدعاء أدوات ميديا بايب
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.title("نظام التعرف السريع على الوجوه 🚀")

# تشغيل الكاميرا
img_file_buffer = st.camera_input("التقط صورة")

if img_file_buffer is not None:
    # تحويل الصورة إلى مصفوفة نيمباي
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    # تحويل اللون من BGR إلى RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # بدء عملية الاكتشاف
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
            
            st.success(f"تم اكتشاف {len(results.detections)} وجه!")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="النتيجة")
        else:
            st.warning("لم يتم العثور على أي وجه في الصورة")
