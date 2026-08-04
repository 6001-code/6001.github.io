import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Ciprofloxacin Resistance Classifier",
    page_icon="🔬",
    layout="centered"
)

# 2. โหลดโมเดล (ใช้ cache เพื่อไม่ให้โหลดใหม่ทุกครั้งที่กดปุ่ม)
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model('cipro_model.keras')

try:
    model = load_trained_model()
except Exception as e:
    st.error("ไม่สามารถโหลดไฟล์ cipro_model.keras ได้ กรุณาตรวจสอบว่ามีไฟล์อยู่ในโฟลเดอร์เดียวกันหรือไม่")

# 3. ส่วนหัวของเว็บไซต์
st.title("🔬 Ciprofloxacin Resistance Classifier")
st.subheader("ระบบทำนายการดื้อยา Ciprofloxacin จากภาพรายงาน MALDI-TOF")
st.write("อัปโหลดภาพรายงาน MALDI-TOF (ขนาดใดก็ได้ ระบบจะปรับเป็น 224x224 ให้อัตโนมัติ)")

# แสดงคำเตือนเรื่องขนาด Dataset ตามที่ระบุใน Notebook
st.info("⚠️ **ข้อจำกัดของระบบ:** โมเดลนี้ถูกเทรนด้วยชุดข้อมูลขนาดเล็ก (N=87) ผลลัพธ์นี้ใช้เพื่อการวิจัยและการสาธิตเท่านั้น ไม่ควรใช้เป็นเครื่องมือยืนยันทางการแพทย์เดี่ยวๆ")

# 4. ส่วนอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("เลือกไฟล์ภาพ PNG หรือ JPG...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # อ่านและแสดงรูปภาพ
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)
    
    # Preprocess ภาพให้ตรงกับที่เทรนใน Notebook
    # 1) Resize เป็น 224x224
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    
    # 2) เพิ่ม Dimension ให้เป็น Batch (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 3) Normalize/Preprocess (ถ้าโมเดลของคุณใช้ preprocess_input ของ MobileNet/ResNet ให้ใส่เพิ่มตรงนี้)
    # เช่น img_batch = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
    
    with col2:
        with st.spinner('กำลังวิเคราะห์สเปกตรัม...'):
            # ทำนายผล (Output ออกมาเป็นค่าความน่าจะเป็น 0.0 - 1.0)
            prediction_prob = model.predict(img_batch)[0][0]
            
            # แปลผลลัพธ์
            # 0 = Susceptible, 1 = Resistant
            is_resistant = prediction_prob >= 0.5
            confidence = prediction_prob if is_resistant else (1 - prediction_prob)

            st.write("### ผลการวิเคราะห์")
            if is_resistant:
                st.error("🔴 **Resistant (ดื้อยา)**")
                st.write(f"โอกาสดื้อยา: **{prediction_prob*100:.2f}%**")
            else:
                st.success("🟢 **Susceptible (ไม่ดื้อยา)**")
                st.write(f"โอกาสไม่ดื้อยา: **{(1-prediction_prob)*100:.2f}%**")

            # แสดงหลอดสเกลความน่าจะเป็น
            st.progress(float(prediction_prob))