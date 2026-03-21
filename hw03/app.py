import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="人脸检测Web应用", page_icon="😊")
st.title("😊 人脸检测Web应用")

# 加载 OpenCV 人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])

def detect_faces(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_bgr, "Face", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return result, len(faces)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="原图", use_container_width=True)
    
    with st.spinner("正在检测中..."):
        result_image, face_count = detect_faces(original_image)
    
    st.image(result_image, caption=f"检测结果（找到 {face_count} 张人脸）", use_container_width=True)
    
    if face_count > 0:
        st.success(f"✅ 成功检测到 {face_count} 张人脸")
    else:
        st.warning("⚠️ 未检测到人脸")

else:
    st.info("👈 请从左侧上传图片")
