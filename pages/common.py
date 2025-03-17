import streamlit as st
from PIL import Image
import os
import hashlib


def _generate_color(token: int):
    """Generate a consistent color for a token"""
    hash_object = hashlib.md5(str(token).encode())
    hash_hex = hash_object.hexdigest()

    r = int(int(hash_hex[0:2], 16) * 0.6 + 256 * 0.4)
    g = int(int(hash_hex[2:4], 16) * 0.6 + 256 * 0.4)
    b = int(int(hash_hex[4:6], 16) * 0.6 + 256 * 0.4)

    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    if brightness > 200:
        r = min(r * 0.7, 255)
        g = min(g * 0.7, 255)
        b = min(b * 0.7, 255)

    return f"rgb({r},{g},{b})"


def create_colored_text_html(tokens):
    """Create HTML with color-coded tokens"""
    html_parts = []
    for i, token in enumerate(tokens):
        html_parts.append(
            f'<span style="background-color:{_generate_color(token)}; '
            f"padding: 0; margin-right: 1px; font-size: 1.2em; "
            f'display: inline-block;" '
            f'title="{i}-{token}">{token.replace(" ", "␣")}</span>'
        )
    return "".join(html_parts)


def add_bmc_footer():
    """
    Adds a Buy Me a Coffee footer to a Streamlit app with button, QR code, and thank you message.
    Place this at the bottom of your Streamlit app.
    """
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("<p style='font-size: 14px; color: #636363;'>Thank you for using this app! If you found it helpful, consider supporting my work. Any amout would be greatly appriciated!</p>", unsafe_allow_html=True)
        
        # Check if the image file exists and display button
        button_path = os.path.join("pages", "images", "bmc-button.png")
        if os.path.exists(button_path):
            button_img = Image.open(button_path)
            st.markdown(
                f"<a href='https://buymeacoffee.com/sieun' target='_blank'><img src='data:image/png;base64,{image_to_base64(button_img)}' alt='Buy Me A Coffee' style='height: 45px;'></a>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<a href='https://buymeacoffee.com/sieun' target='_blank' style='display: inline-block; background-color: #FFDD00; color: #000000; padding: 8px 20px; border-radius: 5px; text-decoration: none; font-weight: bold;'>Buy me a coffee</a>",
                unsafe_allow_html=True
            )
    
    with col3:
        # Display QR code if it exists
        qr_path = os.path.join("pages", "images", "bmc_qr.png")
        if os.path.exists(qr_path):
            qr_img = Image.open(qr_path)
            st.image(qr_img, width=120, caption="Scan to support")

def image_to_base64(img):
    """Convert an image to base64 string for embedding in HTML"""
    import io
    import base64
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str
