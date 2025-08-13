# analyzer.py
# FINAL COMPLETE UI: Main dashboard, asset search, and a FULLY FUNCTIONAL contacts page with auto-extraction and export.

import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO

from dotenv import load_dotenv
import sqlalchemy as db
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# --- Setup ---
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DB_FILE = "reports.sqlite"

if not os.path.exists(DB_FILE):
    st.error(f"קובץ בסיס הנתונים '{DB_FILE}' לא נמצא. אנא הרץ תחילה את סקריפט העיבוד 'process_files.py' על תיקיית הקבצים שלך.")
    st.stop()

engine = db.create_engine(f'sqlite:///{DB_FILE}')
Base = declarative_base()

# --- Database Schema Definition ---
class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True); filename = Column(String); company_name = Column(String); full_text = Column(Text)
    findings = relationship("Finding", back_populates="document"); auto_contacts = relationship("AutoContact", back_populates="document")
class Finding(Base):
    __tablename__ = 'findings'
    id = Column(Integer, primary_key=True); document_id = Column(Integer, ForeignKey('documents.id')); category = Column(String); finding_text = Column(Text); document = relationship("Document", back_populates="findings")
class Contact(Base): # Manual contacts
    __tablename__ = 'contacts'
    id = Column(Integer, primary_key=True, autoincrement=True); first_name = Column(String); last_name = Column(String); company = Column(String); role = Column(String); phone = Column(String); email = Column(String)
class AutoContact(Base): # Auto-extracted contacts
    __tablename__ = 'auto_contacts'
    id = Column(Integer, primary_key=True); document_id = Column(Integer, ForeignKey('documents.id')); name = Column(String); role = Column(String); email = Column(String); phone = Column(String); document = relationship("Document", back_populates="auto_contacts")

Session = sessionmaker(bind=engine)

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
    session = Session()
    reports = {}
    report_queries = {
        "נדל\"ן בישראל": "real_estate_israel", "קרקעות כמלאי בישראל": "land_inventory_israel",
        "נדל\"ן בארה\"ב": "real_estate_usa", "נדל\"ן באירופה": "real_estate_europe",
        "משקיעות בסטארטאפים": "startup_investments", "חברות בתחום האנרגיה": "energy_sector",
        "קניונים מתוכננים": "malls_planned", "מרכזים מסחריים מתוכננים": "commercial_centers_planned",
        "שכונות חדשות מתוכננות": "new_neighborhoods", "מגורים ומסחר משולב": "mixed_use_residential"
    }
    try:
        for report_name, category in report_queries.items():
            query = (session.query(Document.company_name, Finding.finding_text, Document.filename).join(Finding).filter(Finding.category == category))
            reports[report_name] = pd.read_sql(query.statement, session.bind)
    finally: session.close()
    return reports

@st.cache_data(ttl=60)
def get_contacts_df(manual=True):
    session = Session()
    try:
        if manual:
            query = session.query(Contact).statement
            df = pd.read_sql(query, session.bind)
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
        search_terms = re.findall(r'\b\w+\b', question)
        finding_filter = [Finding.finding_text.like(f'%{term}%') for term in search_terms]
        candidate_findings = (session.query(Document.company_name, Document.filename, Finding.finding_text).join(Finding).filter(db.or_(Finding.category == 'real_estate_israel', Finding.category == 'land_inventory_israel'), db.or_(*finding_filter)).limit(30).all())
        if not candidate_findings: return "לא נמצאו ממצאים ראשוניים התואמים לשאלתך."
        context = "\n\n".join(f"Company: {f.company_name}, File: {f.filename}\nFinding: {f.finding_text}" for f in candidate_findings)
        summarize_template = "Based *only* on the provided findings about Israeli real estate, answer the user's question. Create a Markdown table. User's Question: {question}. Findings: {context}. Final Answer (as a Markdown table in Hebrew):"
        prompt = ChatPromptTemplate.from_template(summarize_template)
        chain = prompt | get_deepseek_llm() | StrOutputParser()
        return chain.invoke({"question": question, "context": context})
    finally: session.close()

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
            else:
                st.write("לא נמצאו ממצאים בקטגוריה זו.")

elif page == "asset_search":
    st.header("איתור נכסים בישראל")
    tab1, tab2 = st.tabs(["💬 שיחה עם AI", "🔍 חיפוש פשוט"])
    with tab1:
        st.info("שאל שאלה בשפה חופשית על נכסים בישראל. ה-AI יחפש בממצאים הרלוונטיים ויסכם את התשובה.")
        if "asset_messages" not in st.session_state: st.session_state.asset_messages = []
        for msg in st.session_state.asset_messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if prompt := st.chat_input("לדוגמה: אילו חברות מחזיקות קרקעות בחיפה?"):
            st.session_state.asset_messages.append({"role": "user", "content": prompt}); st.rerun()
        if st.session_state.asset_messages and st.session_state.asset_messages[-1]["role"] == "user":
            with st.chat_message("assistant"):
                with st.spinner("חושב..."):
                    response = ai_asset_search(st.session_state.asset_messages[-1]["content"])
                    st.markdown(response); st.session_state.asset_messages.append({"role": "assistant", "content": response}); st.rerun()
    with tab2:
        st.info("הזן מונח (עיר, רחוב, סוג נכס) לחיפוש מהיר בכל הטקסטים של הדוחות.")
        keyword = st.text_input("הזן מונח לחיפוש:", key="keyword_search")
        if st.button("חפש", key="search_button"):
            if keyword:
                with st.spinner("מחפש..."):
                    session = Session(); results = session.query(Document.company_name, Document.filename, Document.full_text).filter(Document.full_text.like(f'%{keyword}%')).limit(50).all(); session.close()
                    if results:
                        st.success(f"נמצאו {len(results)} מסמכים המכילים את המונח '{keyword}':")
                        for doc in results:
                            try:
                                safe_keyword = re.escape(keyword); match = re.search(safe_keyword, doc.full_text, re.IGNORECASE)
                                if match:
                                    snippet = doc.full_text[max(0, match.start() - 80):match.end() + 80]
                                    highlighted_snippet = re.sub(f'({safe_keyword})', r'**\1**', snippet, flags=re.IGNORECASE)
                                    st.info(f"**חברה:** {doc.company_name} | **קובץ:** {doc.filename}"); st.markdown(f'<div dir="rtl">...{highlighted_snippet}...</div>', unsafe_allow_html=True)
                            except Exception as e: st.warning(f"לא ניתן היה להציג קטע מתוך '{doc.filename}': {e}")
                    else: st.warning(f"המונח '{keyword}' לא נמצא באף מסמך.")
            else: st.warning("אנא הזן מונח לחיפוש.")

elif page == "contacts_page":
    st.header("ניהול אנשי קשר")
    st.subheader("אנשי קשר שחולצו אוטומטית מהדוחות")
    auto_contacts_df = get_contacts_df(manual=False)
    if not auto_contacts_df.empty:
        st.dataframe(auto_contacts_df, use_container_width=True)
        st.download_button(label="📥 ייצא רשימה אוטומטית לאקסל", data=to_excel(auto_contacts_df), file_name="אנשי_קשר_אוטומטי.xlsx")
    else:
        st.info("לא חולצו אנשי קשר באופן אוטומטי מהמסמכים.")
    st.markdown("---")
    
    with st.form("contact_form", clear_on_submit=True):
        st.subheader("הוספת איש קשר ידנית")
        c1, c2 = st.columns(2)
        first_name = c1.text_input("שם פרטי"); last_name = c2.text_input("שם משפחה")
        company = c1.text_input("שם חברה"); role = c2.text_input("תפקיד")
        phone = c1.text_input("טלפון"); email = c2.text_input("כתובת מייל")
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
    contacts_df = get_contacts_df(manual=True)
    search_term = st.text_input("חפש איש קשר:")
    if search_term:
        contacts_df = contacts_df[contacts_df.apply(lambda row: search_term.lower() in ' '.join(row.astype(str)).lower(), axis=1)]
    if not contacts_df.empty:
        if 'original_contacts' not in st.session_state: st.session_state.original_contacts = contacts_df.copy()
        edited_df = st.data_editor(contacts_df, key="contacts_editor", use_container_width=True, hide_index=True, disabled=["מזהה"])
        st.download_button(label="📥 ייצא רשימה ידנית לאקסל", data=to_excel(edited_df.drop(columns=['מזהה'])), file_name="אנשי_קשר_ידני.xlsx")
        if not st.session_state.original_contacts.equals(edited_df):
            session = Session()
            try:
                for idx, edited_row in edited_df.iterrows():
                    original_row = st.session_state.original_contacts.loc[st.session_state.original_contacts['מזהה'] == edited_row['מזהה']]
                    if not original_row.empty and not original_row.iloc[0].equals(edited_row):
                        contact_to_update = session.query(Contact).filter_by(id=edited_row['מזהה']).one()
                        contact_to_update.first_name, contact_to_update.last_name, contact_to_update.company, contact_to_update.role, contact_to_update.phone, contact_to_update.email = edited_row['שם פרטי'], edited_row['שם משפחה'], edited_row['חברה'], edited_row['תפקיד'], edited_row['טלפון'], edited_row['מייל']
                session.commit(); st.toast("השינויים נשמרו!"); st.cache_data.clear(); st.rerun()
            except Exception as e:
                session.rollback(); st.error(f"שגיאה בעדכון: {e}")
            finally: session.close()
    else:
        st.info("לא נמצאו אנשי קשר." if search_term else "עדיין לא נוספו אנשי קשר.")