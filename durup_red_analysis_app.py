import streamlit as st
import cv2
import numpy as np
from PIL import Image

def calculate_red_ratio(image):
    image_np = np.array(image)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
lower_red1 = np.array([0, 20, 20])
upper_red1 = np.array([15, 255, 255])
lower_red2 = np.array([160, 20, 20])
upper_red2 = np.array([180, 255, 255])


    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    red_pixels = np.sum(mask > 0)
    total_pixels = image_np.shape[0] * image_np.shape[1]
    red_ratio = (red_pixels / total_pixels) * 100
    return red_ratio, mask

st.title("두릅 적색 비율 자동 분석기")

uploaded_file = st.file_uploader("두릅 사진을 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))  # 흰 배경
    image = Image.alpha_composite(background, image).convert("RGB")

    red_ratio, mask = calculate_red_ratio(image)
    st.markdown(f"### 적색 비율: {red_ratio:.2f}%")

    st.markdown("### 적색 영역 표시")
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    st.image(mask_rgb, caption="적색 필터링 결과", use_column_width=True)

    st.markdown("---")
    st.markdown("이 분석기는 최신 대화를 반영하여 HSV 방식으로 두릅의 적색 비율을 정확하게 자동 계산합니다.")
