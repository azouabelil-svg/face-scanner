import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# إعدادات ميديا بايب
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.title("ماسح الوجه الذكي (نسخة سريعة)")
st.write("هذا النظام يستخدم تقنية Google MediaPipe للتعرف السريع")

# تشغيل الكاميرا
img_file_buffer = st.camera_input("وجه الكاميرا نحو وجهك")

if img_file_buffer is not None:
    # تحويل الصورة
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # معالجة الصورة للبحث عن الوجوه
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                # رسم مربع حول الوجه
                mp_drawing.draw_detection(cv2_img, detection)
                
                # حساب درجة الثقة
                score = detection.score[0]
                st.success(f"تم اكتشاف وجه بنسبة ثقة: {score:.2f}")
            
            # عرض الصورة مع تحديد الوجه
            st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption="تمت المعالجة بنجاح")
        else:
            st.warning("لم يتم العثور على وجه، حاول التقريب من الكاميرا")
