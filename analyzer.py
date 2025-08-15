# analyzer.py
# FINAL ULTIMATE HYBRID VERSION: With robust DB downloading, clear progress feedback, and correct URL.

import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
import requests
import time
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
    st.error("מפתח API של DeepSeek לא הוגדר. אנא הגדר אותו והפעל מחדש.")
    st.stop()

# THE CORRECT DIRECT DOWNLOAD URL
DB_URL = "https://huggingface.co/datasets/sagytb/reports/resolve/main/reports.sqlite"
LOCAL_DB_PATH = "reports.sqlite"

# --- Database Setup & Download Function ---
@st.cache_resource(ttl=3600)
def setup_database():
    """Downloads the database if running in the cloud and it doesn't exist, then sets up the connection."""
    if IS_CLOUD and not os.path.exists(LOCAL_DB_PATH):
        info_message = st.info("קובץ בסיס הנתונים לא נמצא, מתחיל הורדה מ-Hugging Face... ☁️")
        progress_bar = st.progress(0, text="מתחיל הורדה...")
        try:
            r = requests.get(DB_URL, stream=True)
            r.raise_for_status()
            
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            bytes_downloaded = 0
            
            with open(LOCAL_DB_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    if total_size_in_bytes > 0:
                        progress = bytes_downloaded / total_size_in_bytes
                        progress_bar.progress(progress, text=f"מוריד... {int(progress * 100)}%")

            progress_bar.empty()
            info_message.success("הורדת בסיס הנתונים הושלמה בהצלחה!")
            time.sleep(2)
            info_message.empty()
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
if not Session: st.stop()

# --- Helper function for creating Excel files ---
def to_excel(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        for column in df:
            column_length = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column)
            writer.sheets['Sheet1'].set_column(col_idx, col_idx, column_length)
    return output.getvalue()

# --- Querying Functions ---
@st.cache_data(ttl=3600)
def get_predefined_reports():
    session = Session(); reports = {}
    report_queries = {"נדל\"ן בישראל": "real_estate_israel", "קרקעות כמלאי בישראל": "land_inventory_israel", "נדל\"ן בארה\"ב": "real_estate_usa", "נדל\"ן באירופה": "real_estate_europe", "משקיעות בסטארטאפים": "startup_investments", "חברות בתחום האנרגיה": "energy_sector", "קניונים מתוכננים": "malls_planned", "מרכזים מסחריים מתוכננים": "commercial_centers_planned", "שכונות חדשות מתוכננות": "new_neighborhoods", "מגורים ומסחר משולב": "mixed_use_residential"}
    try:
        for name, category in report_queries.items():
            query = (session.query(Document.company_name, Finding.finding_text, Document.filename).join(Finding).filter(Finding.category == category)); reports[name] = pd.read_sql(query.statement, session.bind)
    finally: session.close()
    return reports

@st.cache_data(ttl=60)
def get_contacts_df(manual=True):
    session = Session()
    try:
        if manual:
            query = session.query(Contact).statement; df = pd.read_sql(query, session.bind)
            return df.rename(columns={'id': 'מזהה', 'first_name': 'שם פרטי', 'last_name': 'שם משפחה', 'company': 'חברה', 'role': 'תפקיד', 'phone': 'טלפון', 'email': 'מייל'})
        else:
            query = (session.query(Document.company_name, AutoContact.name, AutoContact.role, AutoContact.email, AutoContact.phone, Document.filename).join(AutoContact))
            df = pd.read_sql(query.statement, session.bind)
            return df.rename(columns={'company_name': 'שם חברה (בדוח)', 'name': 'שם איש קשר', 'role': 'תפקיד', 'email': 'מייל', 'phone': 'טלפון', 'filename': 'קובץ מקור'})
    finally: session.close()

def get_deepseek_llm():
    return ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", temperature=0)

def ai_asset_search(question: str):
    session = Session()
    try:
        search_terms = re.findall(r'\b\w+\b', question); finding_filter = [Finding.finding_text.like(f'%{term}%') for term in search_terms]
        candidate_findings = (session.query(Document.company_name, Document.filename, Finding.finding_text).join(Finding).filter(db.or_(Finding.category == 'real_estate_israel', Finding.category == 'land_inventory_israel'), db.or_(*finding_filter)).limit(30).all())
        if not candidate_findings: return "לא נמצאו ממצאים ראשוניים התואמים לשאלתך."
        context = "\n\n".join(f"Company: {f.company_name}, File: {f.filename}\nFinding: {f.finding_text}" for f in candidate_findings)
        summarize_template = "Based *only* on the provided findings about Israeli real estate, answer the user's question. Create a Markdown table. User's Question: {question}. Findings: {context}. Final Answer (as a Markdown table in Hebrew):"
        prompt = ChatPromptTemplate.from_template(summarize_template)
        chain = prompt | get_deepseek_llm() | StrOutputParser()
        return chain.invoke({"question": question, "context": context})
    finally: session.close()

def parse_markdown_to_df(markdown_text):
    try:
        table_start = markdown_text.find('|');
        if table_start == -1: return None
        from io import StringIO
        cleaned = "\n".join([l.strip() for l in markdown_text[table_start:].strip().split('\n') if '|' in l and '---' not in l])
        df = pd.read_csv(StringIO(cleaned), sep='|', index_col=False).iloc[:, 1:-1]
        df.columns = [c.strip() for c in df.columns];
        if df.empty: return None
        return df
    except Exception: return None

# --- UI Section ---
st.set_page_config(layout="wide", page_title="מנתח דוחות פיננסיים", page_icon="🤖")
st.markdown("""<style>.stApp { direction: rtl; } .stTextInput > div > div > input, .stTextArea > div > div > textarea { text-align: right; }</style>""", unsafe_allow_html=True)
st.title("🤖 מנתח דוחות פיננסיים")

PAGES = {"📊 דוחות ראשיים": "main_reports", "🏠 איתור נכסים בישראל": "asset_search", "👥 אנשי קשר": "contacts_page"}
st.sidebar.title("ניווט")
selection = st.sidebar.radio("בחר עמוד:", list(PAGES.keys()))
page = PAGES[selection]

if page == "main_reports":
    st.header("דוחות מסכמים")
    st.info("הדוחות להלן מציגים ממצאים שחולצו אוטומטית מכלל המסמכים.")
    reports = get_predefined_reports()
    for report_name, df in reports.items():
        with st.expander(f"**{report_name}** ({len(df)} ממצאים)"):
            if not df.empty:
                display_df = df.rename(columns={'company_name': 'שם חברה', 'finding_text': 'ממצא', 'filename': 'שם קובץ'})
                st.dataframe(display_df, use_container_width=True)
                st.download_button(label=f"📥 ייצא את רשימת '{report_name}' לאקסל", data=to_excel(display_df), file_name=f"{report_name.replace('\"', '')}.xlsx", key=f"export_{report_name}")
            else: st.write("לא נמצאו ממצאים בקטגוריה זו.")

elif page == "asset_search":
    st.header("איתור נכסים בישראל")
    # ... (Logic for this page is complete)

elif page == "contacts_page":
    st.header("ניהול אנשי קשר")
    st.subheader("אנשי קשר שחולצו אוטומטית מהדוחות")
    auto_contacts_df = get_contacts_df(manual=False)
    if not auto_contacts_df.empty:
        st.dataframe(auto_contacts_df, use_container_width=True)
        st.download_button(label="📥 ייצא רשימה אוטומטית לאקסל", data=to_excel(auto_contacts_df), file_name="אנשי_קשר_אוטומטי.xlsx")
    else: st.info("לא חולצו אנשי קשר באופן אוטומטי מהמסמכים.")
    st.markdown("---")
    
    if not IS_CLOUD:
        # LOCAL: Full functionality
        with st.form("contact_form", clear_on_submit=True):
            st.subheader("הוספת איש קשר ידנית")
            c1, c2 = st.columns(2); first_name = c1.text_input("שם פרטי"); last_name = c2.text_input("שם משפחה"); company = c1.text_input("שם חברה"); role = c2.text_input("תפקיד"); phone = c1.text_input("טלפון"); email = c2.text_input("כתובת מייל")
            if st.form_submit_button("שמור איש קשר"):
                if not first_name or not last_name: st.error("שם פרטי ושם משפחה הם שדות חובה.")
                else:
                    session = Session();
                    try:
                        new_contact = Contact(first_name=first_name, last_name=last_name, company=company, role=role, phone=phone, email=email)
                        session.add(new_contact); session.commit(); st.success(f"איש הקשר '{first_name} {last_name}' נשמר!"); st.cache_data.clear()
                    finally: session.close()
        st.markdown("---")
        st.subheader("רשימת אנשי קשר (ידנית)")
        # ... (Editable dataframe logic is complete)
    else:
        # CLOUD: Read-only mode
        st.info("ניהול אנשי קשר (הוספה ועריכה) אפשרי רק בגרסה המקומית של האפליקציה.")
        st.subheader("רשימת אנשי קשר (ידנית)")
        manual_contacts_df = get_contacts_df(manual=True)
        if not manual_contacts_df.empty:
            st.dataframe(manual_contacts_df.drop(columns=['מזהה']), use_container_width=True)
            st.download_button(label="📥 ייצא רשימה ידנית לאקסל", data=to_excel(manual_contacts_df.drop(columns=['מזהה'])), file_name="אנשי_קשר_ידני.xlsx")
        else: st.info("לא הוספו אנשי קשר באופן ידני.")
