import streamlit as st
import cv2
import numpy as np
from PIL import Image

def calculate_red_ratio(image):
    image_np = np.array(image)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # 더 넓은 적색 범위 설정
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 30, 30])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    red_pixels = np.sum(mask > 0)
    total_pixels = image_np.shape[0] * image_np.shape[1]
    red_ratio = (red_pixels / total_pixels) * 100
    return red_ratio, mask

st.title("두릅 적색 비율 자동 분석기")

uploaded_file = st.file_uploader("두릅 사진을 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # RGBA → RGB 변환 처리 (투명 배경 흰색으로 채우기)
    image = Image.open(uploaded_file).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, image).convert("RGB")

    st.image(image, caption="업로드한 두릅 이미지", use_column_width=True)

    red_ratio, mask = calculate_red_ratio(image)
    st.markdown(f"### 적색 비율: {red_ratio:.2f}%")

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    st.image(mask_rgb, caption="적색 필터링 결과", use_column_width=True)
