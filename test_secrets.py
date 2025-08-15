# test_secrets.py
import streamlit as st
import os

st.set_page_config(layout="wide")
st.title("בדיקת גישה ל-Secrets")

st.header("בדיקה באמצעות st.secrets")

# ננסה לגשת לסוד
api_key = st.secrets.get("DEEPSEEK_API_KEY")

if api_key:
    st.success("🎉 הצלחה! מפתח ה-API נמצא ב-st.secrets.")
    # נציג רק חלק קטן מהמפתח כדי לוודא שזה המפתח הנכון, לעולם לא את כולו!
    st.write(f"תחילת המפתח: `{api_key[:5]}...`")
else:
    st.error("❌ כישלון! מפתח ה-API לא נמצא ב-st.secrets.")
    st.warning("אנא ודא שהגדרת את הסוד נכון בממשק הניהול של האפליקציה.")