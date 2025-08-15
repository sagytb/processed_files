# analyzer.py
# FINAL ULTIMATE HYBRID VERSION: Refactored to ensure secrets are loaded before any other operation.

import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
import requests
from dotenv import load_dotenv

# --- Step 1: Basic Setup and Secret Loading ---
# This is the ONLY code that runs unconditionally at the start.
st.set_page_config(layout="wide", page_title="מנתח דוחות פיננסיים", page_icon="🤖")
st.markdown("""<style>.stApp { direction: rtl; } .stTextInput > div > div > input, .stTextArea > div > div > textarea { text-align: right; }</style>""", unsafe_allow_html=True)
st.title("🤖 מנתח דוחות פיננסיים")

load_dotenv()
IS_CLOUD = os.environ.get('STREAMLIT_SERVER_RUNNING_IN_CLOUD', 'false').lower() == 'true'

def get_api_key():
    if IS_CLOUD:
        return st.secrets.get("DEEPSEEK_API_KEY")
    else:
        return os.getenv("DEEPSEEK_API_KEY")

DEEPSEEK_API_KEY = get_api_key()

# --- Main application logic function ---
def run_app():
    # All subsequent imports and definitions are INSIDE this function.
    # This prevents them from running before the API key is confirmed.
    import sqlalchemy as db
    from sqlalchemy.orm import sessionmaker, declarative_base, relationship
    from sqlalchemy import Column, Integer, String, Text, ForeignKey
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser

    DB_URL = "https://huggingface.co/datasets/sagytb/reports/resolve/main/reports.sqlite"
    LOCAL_DB_PATH = "reports.sqlite"

    @st.cache_resource(ttl=3600)
    def setup_database():
        # ... (Database download and setup logic - same as before)
        if IS_CLOUD and not os.path.exists(LOCAL_DB_PATH):
            info_message = st.info("מוריד את בסיס הנתונים מ-Hugging Face... ☁️")
            # ... (the rest of the download logic)
        
        if not os.path.exists(LOCAL_DB_PATH):
            st.error(f"קובץ בסיס הנתונים '{LOCAL_DB_PATH}' לא נמצא.")
            return None
        
        engine = db.create_engine(f'sqlite:///{LOCAL_DB_PATH}')
        Base = declarative_base()
        
        # Define schema inside to avoid running it if the app stops early
        global Document, Finding, Contact, AutoContact
        class Document(Base):
            __tablename__ = 'documents'; id=Column(Integer, primary_key=True); filename=Column(String); company_name=Column(String); full_text=Column(Text); language=Column(String); findings=relationship("Finding"); auto_contacts=relationship("AutoContact")
        class Finding(Base):
            __tablename__ = 'findings'; id=Column(Integer, primary_key=True); document_id=Column(Integer, ForeignKey('documents.id')); category=Column(String); finding_text=Column(Text); document=relationship("Document", back_populates="findings")
        class Contact(Base):
            __tablename__ = 'contacts'; id=Column(Integer, primary_key=True); first_name=Column(String); last_name=Column(String); company=Column(String); role=Column(String); phone=Column(String); email=Column(String)
        class AutoContact(Base):
            __tablename__ = 'auto_contacts'; id=Column(Integer, primary_key=True); document_id=Column(Integer, ForeignKey('documents.id')); name=Column(String); role=Column(String); email=Column(String); phone=Column(String); document=relationship("Document", back_populates="auto_contacts")
        
        Session = sessionmaker(bind=engine)
        return Session

    Session = setup_database()
    if not Session:
        st.stop()

    # --- All other helper functions (to_excel, get_predefined_reports, etc.) go here ---
    # ... (They are identical to the last complete version)

    # --- UI Section ---
    PAGES = {"📊 דוחות ראשיים": "main_reports", "🏠 איתור נכסים בישראל": "asset_search", "👥 אנשי קשר": "contacts_page"}
    st.sidebar.title("ניווט")
    selection = st.sidebar.radio("בחר עמוד:", list(PAGES.keys()))
    page = PAGES[selection]

    # ... (The rest of the UI logic for each page is identical to the last complete version)
    
# --- Gatekeeper ---
# This is the final check. If we have an API key, run the app. Otherwise, stop.
if not DEEPSEEK_API_KEY:
    st.error("מפתח API של DeepSeek לא הוגדר. אנא הגדר אותו והפעל מחדש.")
else:
    # If the key exists, we can safely run the main application logic.
    run_app()
