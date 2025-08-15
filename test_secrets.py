# debug_cloud_app.py
# A simple, step-by-step script to diagnose deployment issues on Streamlit Cloud.

import streamlit as st
import os
import requests
import time

st.set_page_config(layout="wide", page_title="Cloud Debugger")
st.markdown("""<style>.stApp { direction: rtl; }</style>""", unsafe_allow_html=True)
st.title("🕵️‍♂️ בודק תקינות לסביבת הענן")

st.info("סקריפט זה בודק את תקינות ההגדרות והגישה למשאבים חיצוניים בסביבת הענן.")

# --- Step 1: Check Environment Variable ---
st.header("שלב 1: בדיקת זיהוי סביבת ענן")
IS_CLOUD = os.environ.get('STREAMLIT_SERVER_RUNNING_IN_CLOUD', 'false').lower() == 'true'
if IS_CLOUD:
    st.success("✅ זוהתה ריצה בסביבת ענן של Streamlit.")
else:
    st.warning("⚠️ זוהתה ריצה בסביבה מקומית.")

# --- Step 2: Check Secrets Access ---
st.header("שלב 2: בדיקת גישה ל-Secrets")
api_key_found = False
try:
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if api_key and isinstance(api_key, str) and api_key.startswith("sk-"):
        st.success("✅ הצלחה! מפתח ה-API נמצא ב-st.secrets.")
        st.write(f"תחילת המפתח: `{api_key[:5]}...`")
        api_key_found = True
    else:
        st.error("❌ כישלון! המשתנה DEEPSEEK_API_KEY לא מוגדר או ריק ב-Secrets.")
        st.warning("אנא ודא שהוספת את הסוד נכון בממשק הניהול של האפליקציה תחת Settings -> Secrets.")
except Exception as e:
    st.error(f"❌ כישלון קריטי! אירעה שגיאה בזמן הניסיון לגשת ל-st.secrets.")
    st.exception(e) # This will print the full traceback to the screen

# --- Step 3: Check Database Download ---
if api_key_found:
    st.header("שלב 3: בדיקת הורדת בסיס הנתונים")
    DB_URL = "https://huggingface.co/datasets/sagytb/reports/resolve/main/reports.sqlite"
    LOCAL_DB_PATH = "reports.sqlite"
    
    st.write(f"מנסה להוריד את הקובץ מהכתובת: `{DB_URL}`")
    
    info_message = st.info("מתחיל ניסיון הורדה...")
    progress_bar = st.progress(0, text="...")

    try:
        r = requests.get(DB_URL, stream=True, timeout=30) # 30 second timeout
        r.raise_for_status()
        
        total_size = int(r.headers.get('content-length', 0))
        bytes_downloaded = 0
        
        with open(LOCAL_DB_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bytes_downloaded += len(chunk)
                if total_size > 0:
                    progress = bytes_downloaded / total_size
                    progress_bar.progress(progress, text=f"מוריד... {int(progress * 100)}%")
        
        progress_bar.empty()
        info_message.success("✅ הצלחה! בסיס הנתונים הורד ונשמר בהצלחה.")
        st.balloons()
        
    except requests.exceptions.RequestException as e:
        info_message.error("❌ כישלון קריטי! אירעה שגיאה במהלך הורדת בסיס הנתונים.")
        st.exception(e)
else:
    st.warning("מדלג על שלב 3 מכיוון שמפתח ה-API לא נמצא.")
