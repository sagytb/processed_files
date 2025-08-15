# analyzer.py
# FINAL COMPLETE HYBRID VERSION: With robust DB downloading and clear user feedback.

import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
import requests
from dotenv import load_dotenv

import sqlalchemy as db
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# --- Setup for Cloud and Local ---
load_dotenv()

IS_CLOUD = os.environ.get('STREAMLIT_SERVER_RUNNING_IN_CLOUD', 'false').lower() == 'true'
DEEPSEEK_API_KEY = ""

if IS_CLOUD:
    DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY")
else:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    st.error("מפתח API של DeepSeek לא הוגדר. אנא בדוק את הגדרות ה-Secrets בענן או את קובץ ה-.env במחשב המקומי.")
    st.stop()

# USE THE CORRECT DIRECT DOWNLOAD URL
DB_URL = "https://huggingface.co/datasets/sagytb/reports/resolve/main/reports.sqlite"
LOCAL_DB_PATH = "reports.sqlite"

# --- Database Setup & Download Function ---
@st.cache_resource(ttl=3600)
def setup_database():
    if IS_CLOUD and not os.path.exists(LOCAL_DB_PATH):
        # Display the message BEFORE starting the download
        info_message = st.info("קובץ בסיס הנתונים לא נמצא, מתחיל הורדה מ-Hugging Face... ☁️")
        progress_bar = st.progress(0)
        try:
            r = requests.get(DB_URL, stream=True)
            r.raise_for_status()
            
            total_size = int(r.headers.get('content-length', 0))
            bytes_downloaded = 0
            
            with open(LOCAL_DB_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(bytes_downloaded / total_size)

            progress_bar.empty() # Remove progress bar on completion
            info_message.success("הורדת בסיס הנתונים הושלמה בהצלחה!")
            time.sleep(2) # Give user time to read the message
            info_message.empty() # Clear the success message
            st.cache_data.clear()

        except requests.exceptions.RequestException as e:
            info_message.error(f"שגיאה קריטית בהורדת בסיס הנתונים: {e}")
            return None
            
    if not os.path.exists(LOCAL_DB_PATH):
        st.error(f"קובץ בסיס הנתונים '{LOCAL_DB_PATH}' לא נמצא. אנא הרץ תחילה את סקריפט העיבוד 'process_files.py'.")
        return None

    engine = db.create_engine(f'sqlite:///{LOCAL_DB_PATH}')
    Base = declarative_base()
    
    global Document, Finding, Contact, AutoContact
    # ... (Schema definition remains the same)
    
    Session = sessionmaker(bind=engine)
    return Session

# ... (The rest of the file is identical to the last complete version)
