import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image
import os

# 页面标题
st.set_page_config(page_title="人脸识别Web应用", page_icon="😊")
st.title("😊 人脸识别Web应用")
st.write("上传一张图片，系统会自动检测并识别其中的人脸")

# 初始化人脸库
@st.cache_resource
def load_known_faces():
    """加载已知人脸库（示例：用名人图片）"""
    known_face_encodings = []
    known_face_names = []
    
    # 这里可以添加你自己的已知人脸
    # 示例：如果文件夹中有图片，可以自动加载
    faces_dir = "known_faces"
    if os.path.exists(faces_dir):
        for filename in os.listdir(faces_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image = face_recognition.load_image_file(f"{faces_dir}/{filename}")
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name)
    
    return known_face_encodings, known_face_names

# 侧边栏选项
st.sidebar.header("设置")

# 人脸识别模式
mode = st.sidebar.radio(
    "选择模式",
    ["仅检测人脸", "识别身份（需人脸库）"]
)

# 加载已知人脸
known_face_encodings, known_face_names = load_known_faces()

# 如果选择识别模式但没有人脸库，提示用户
if mode == "识别身份（需人脸库）" and len(known_face_names) == 0:
    st.sidebar.warning("⚠️ 未找到人脸库，请在 'known_faces' 文件夹中添加人脸图片")
    st.sidebar.info("每张图片命名为人名，如 '张三.jpg'")

# 上传图片
uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])

# 示例图片按钮
st.sidebar.markdown("---")
st.sidebar.subheader("或者使用示例图片")

# 创建示例图片（如果不存在）
sample_dir = "sample_images"
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

sample_files = os.listdir(sample_dir) if os.path.exists(sample_dir) else []
if sample_files:
    selected_sample = st.sidebar.selectbox("选择示例图片", sample_files)
    use_sample = st.sidebar.button("使用此示例")
else:
    st.sidebar.info("暂无示例图片，请上传图片")

def process_image(image):
    """处理图片：人脸检测和识别"""
    # 转换颜色空间（PIL -> RGB -> BGR for face_recognition）
    img_array = np.array(image)
    
    # 人脸检测
    face_locations = face_recognition.face_locations(img_array)
    face_encodings = face_recognition.face_encodings(img_array, face_locations)
    
    # 转换为BGR用于OpenCV绘图
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 处理每个人脸
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # 画框
        cv2.rectangle(img_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # 识别身份
        name = "未知"
        if mode == "识别身份（需人脸库）" and known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[i])
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[i])
            
            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = known_face_names[best_match_index]
        
        # 添加标签
        label = f"{name} ({i+1})" if name != "未知" else f"人脸 {i+1}"
        cv2.rectangle(img_bgr, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(img_bgr, label, (left + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 转换回RGB显示
    result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return result, len(face_locations)

# 处理上传的图片
if uploaded_file is not None:
    # 显示原图
    st.subheader("原图")
    original_image = Image.open(uploaded_file)
    st.image(original_image, use_container_width=True)
    
    # 处理图片
    with st.spinner("正在处理中..."):
        result_image, face_count = process_image(original_image)
    
    # 显示结果
    st.subheader(f"处理结果（检测到 {face_count} 张人脸）")
    st.image(result_image, use_container_width=True)
    
    # 显示人脸位置信息
    if face_count > 0:
        st.success(f"✅ 成功检测到 {face_count} 张人脸")
    else:
        st.warning("⚠️ 未检测到人脸，请尝试其他图片")

# 使用示例图片
elif 'use_sample' in locals() and use_sample:
    sample_path = os.path.join(sample_dir, selected_sample)
    sample_image = Image.open(sample_path)
    
    st.subheader("示例图片")
    st.image(sample_image, use_container_width=True)
    
    with st.spinner("正在处理中..."):
        result_image, face_count = process_image(sample_image)
    
    st.subheader(f"处理结果（检测到 {face_count} 张人脸）")
    st.image(result_image, use_container_width=True)

else:
    # 显示说明
    st.info("👈 请从左侧上传图片或选择示例图片")
    
    # 显示使用说明
    with st.expander("📖 使用说明"):
        st.markdown("""
        ### 如何使用
        1. 点击左侧的"选择一张图片"
        2. 上传包含人脸的图片
        3. 系统会自动检测并框出所有人脸
        4. 如果选择"识别身份"模式，需要先建立人脸库
        
        ### 如何建立人脸库
        1. 在项目目录下创建 `known_faces` 文件夹
        2. 放入你想识别的人脸图片
        3. 图片命名为人名，如 `张三.jpg`、`李四.png`
        4. 程序会自动加载这些图片作为人脸库
        
        ### 技术支持
        - 人脸检测：`face_recognition` 库（基于dlib）
        - 界面框架：Streamlit
        - 图像处理：OpenCV + PIL
        """)
