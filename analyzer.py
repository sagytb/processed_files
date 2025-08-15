# analyzer.py
# FINAL ULTIMATE HYBRID VERSION: With aggressive, step-by-step logging and the full application logic.

import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
import requests
import time
from dotenv import load_dotenv

# --- Step 1: Basic Setup and Title ---
st.set_page_config(layout="wide", page_title="מנתח דוחות פיננסיים", page_icon="🤖")
st.markdown("""<style>.stApp { direction: rtl; } .stTextInput > div > div > input, .stTextArea > div > div > textarea { text-align: right; }</style>""", unsafe_allow_html=True)

# We will show the main title only after all checks pass.
st.title("🕵️‍♂️ טוען ומאתחל את המערכת...")

# --- Step 2: Define all functions first ---

def get_api_key():
    st.write("---")
    st.subheader("שלב 1: בדיקת מפתח API")
    load_dotenv()
    is_cloud = os.environ.get('STREAMLIT_SERVER_RUNNING_IN_CLOUD', 'false').lower() == 'true'
    
    with st.status(f"מאמת זהות בסביבת {'ענן' if is_cloud else 'מחשב מקומי'}...") as status:
        st.write(f"זיהוי סביבה: {'ענן' if is_cloud else 'מקומי'}")
        api_key = None
        if is_cloud:
            st.write("מנסה לקרוא מפתח מ-Streamlit Secrets...")
            api_key = st.secrets.get("DEEPSEEK_API_KEY")
        else:
            st.write("מנסה לקרוא מפתח מקובץ .env מקומי...")
            api_key = os.getenv("DEEPSEEK_API_KEY")

        if api_key and isinstance(api_key, str) and api_key.startswith("sk-"):
            status.update(label="מפתח API אומת בהצלחה!", state="complete", expanded=False)
            st.success(f"✅ מפתח API נמצא! (מתחיל ב: `{api_key[:5]}...`)")
            return api_key
        else:
            status.update(label="מפתח API לא נמצא או לא תקין!", state="error", expanded=True)
            st.error("❌ כישלון: מפתח API של DeepSeek לא נמצא או לא תקין.")
            st.warning("בסביבה מקומית, ודא שהוא קיים בקובץ .env. בענן, ודא שהגדרת אותו ב-Settings -> Secrets.")
            return None

def download_and_setup_database(db_url, local_path):
    st.write("---")
    st.subheader("שלב 2: הכנת בסיס הנתונים")
    
    if os.path.exists(local_path):
        st.success(f"✅ קובץ בסיס הנתונים '{local_path}' כבר קיים מקומית.")
    else:
        with st.status("מוריד את בסיס הנתונים...", expanded=True) as status:
            st.info(f"קובץ בסיס הנתונים לא נמצא, מתחיל הורדה מ:\n{db_url}")
            progress_bar = st.progress(0, text="מתחיל הורדה...")
            try:
                r = requests.get(db_url, stream=True, timeout=60)
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                bytes_downloaded = 0
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if total_size > 0:
                            progress = bytes_downloaded / total_size
                            progress_bar.progress(progress, text=f"מוריד... {int(progress * 100)}%")
                progress_bar.empty()
                status.update(label="הורדת בסיס הנתונים הושלמה!", state="complete")
            except Exception as e:
                status.update(label="כישלון קריטי בהורדת בסיס הנתונים!", state="error")
                st.error("❌ כישלון קריטי בהורדת בסיס הנתונים!")
                st.exception(e)
                return None

    with st.status("מגדיר חיבור לבסיס הנתונים...") as status:
        import sqlalchemy as db
        from sqlalchemy.orm import sessionmaker, declarative_base, relationship
        from sqlalchemy import Column, Integer, String, Text, ForeignKey
        
        engine = db.create_engine(f'sqlite:///{local_path}')
        Base = declarative_base()
        
        global Document, Finding, Contact, AutoContact
        class Document(Base):
            __tablename__ = 'documents'; id=Column(Integer, primary_key=True); filename=Column(String); company_name=Column(String); full_text=Column(Text); language=Column(String); findings=relationship("Finding", back_populates="document"); auto_contacts=relationship("AutoContact", back_populates="document")
        class Finding(Base):
            __tablename__ = 'findings'; id=Column(Integer, primary_key=True); document_id=Column(Integer, ForeignKey('documents.id')); category=Column(String); finding_text=Column(Text); document=relationship("Document", back_populates="findings")
        class Contact(Base):
            __tablename__ = 'contacts'; id=Column(Integer, primary_key=True, autoincrement=True); first_name=Column(String); last_name=Column(String); company=Column(String); role=Column(String); phone=Column(String); email=Column(String)
        class AutoContact(Base):
            __tablename__ = 'auto_contacts'; id=Column(Integer, primary_key=True, autoincrement=True); document_id=Column(Integer, ForeignKey('documents.id')); name=Column(String); role=Column(String); email=Column(String); phone=Column(String); document=relationship("Document", back_populates="auto_contacts")
        
        status.update(label="החיבור לבסיס הנתונים הוגדר בהצלחה.", state="complete")
        return sessionmaker(bind=engine)

def run_main_app(Session_factory, api_key):
    # This is the main UI, it will only run if everything above succeeded.
    import sqlalchemy as db
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser

    st.success("✅ כל הבדיקות המקדימות עברו בהצלחה. טוען אפליקציה ראשית...")
    time.sleep(2)
    st.experimental_rerun() # Rerun to clear the debug messages and show the final app

def to_excel(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        for column in df:
            column_length = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column)
            writer.sheets['Sheet1'].set_column(col_idx, col_idx, column_length)
    return output.getvalue()
    
@st.cache_data(ttl=3600)
def get_predefined_reports(Session):
    session = Session(); reports = {}; 
    report_queries = {"נדל\"ן בישראל": "real_estate_israel", "קרקעות כמלאי בישראל": "land_inventory_israel", "נדל\"ן בארה\"ב": "real_estate_usa", "נדל\"ן באירופה": "real_estate_europe", "משקיעות בסטארטאפים": "startup_investments", "חברות בתחום האנרגיה": "energy_sector", "קניונים מתוכננים": "malls_planned", "מרכזים מסחריים מתוכננים": "commercial_centers_planned", "שכונות חדשות מתוכננות": "new_neighborhoods", "מגורים ומסחר משולב": "mixed_use_residential"}
    try:
        for name, category in report_queries.items():
            query = (session.query(Document.company_name, Finding.finding_text, Document.filename).join(Finding).filter(Finding.category == category)); reports[name] = pd.read_sql(query.statement, session.bind)
    finally: session.close()
    return reports

@st.cache_data(ttl=60)
def get_contacts_df(Session, manual=True):
    session = Session()
    try:
        if manual:
            query = session.query(Contact).statement; df = pd.read_sql(query, session.bind); return df.rename(columns={'id': 'מזהה', 'first_name': 'שם פרטי', 'last_name': 'שם משפחה', 'company': 'חברה', 'role': 'תפקיד', 'phone': 'טלפון', 'email': 'מייל'})
        else:
            query = (session.query(Document.company_name, AutoContact.name, AutoContact.role, AutoContact.email, AutoContact.phone, Document.filename).join(AutoContact)); df = pd.read_sql(query.statement, session.bind); return df.rename(columns={'company_name': 'שם חברה (בדוח)', 'name': 'שם איש קשר', 'role': 'תפקיד', 'email': 'מייל', 'phone': 'טלפון', 'filename': 'קובץ מקור'})
    finally: session.close()

def get_deepseek_llm():
    return ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", temperature=0)

def ai_asset_search(Session, question: str):
    session = Session()
    try:
        search_terms = re.findall(r'\b\w+\b', question); finding_filter = [Finding.finding_text.like(f'%{term}%') for term in search_terms]; candidate_findings = (session.query(Document.company_name, Document.filename, Finding.finding_text).join(Finding).filter(db.or_(Finding.category == 'real_estate_israel', Finding.category == 'land_inventory_israel'), db.or_(*finding_filter)).limit(30).all())
        if not candidate_findings: return "לא נמצאו ממצאים ראשוניים התואמים לשאלתך."
        context = "\n\n".join(f"Company: {f.company_name}, File: {f.filename}\nFinding: {f.finding_text}" for f in candidate_findings); summarize_template = "Based *only* on the provided findings about Israeli real estate, answer the user's question. Create a Markdown table. User's Question: {question}. Findings: {context}. Final Answer (as a Markdown table in Hebrew):"; prompt = ChatPromptTemplate.from_template(summarize_template); chain = prompt | get_deepseek_llm() | StrOutputParser(); return chain.invoke({"question": question, "context": context})
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

def main_ui(Session):
    st.title("🤖 מנתח דוחות פיננסיים")
    PAGES = {"📊 דוחות ראשיים": "main_reports", "🏠 איתור נכסים בישראל": "asset_search", "👥 אנשי קשר": "contacts_page"}; st.sidebar.title("ניווט"); selection = st.sidebar.radio("בחר עמוד:", list(PAGES.keys())); page = PAGES[selection]
    
    if page == "main_reports":
        st.header("דוחות מסכמים"); st.info("הדוחות להלן מציגים ממצאים שחולצו אוטומטית מכלל המסמכים.")
        reports = get_predefined_reports(Session)
        for report_name, df in reports.items():
            with st.expander(f"**{report_name}** ({len(df)} ממצאים)"):
                if not df.empty:
                    display_df = df.rename(columns={'company_name': 'שם חברה', 'finding_text': 'ממצא', 'filename': 'שם קובץ'}); st.dataframe(display_df, use_container_width=True); st.download_button(label=f"📥 ייצא את רשימת '{report_name}' לאקסל", data=to_excel(display_df), file_name=f"{report_name.replace('\"', '')}.xlsx", key=f"export_{report_name}")
                else: st.write("לא נמצאו ממצאים בקטגוריה זו.")
    
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
                        response = ai_asset_search(Session, st.session_state.asset_messages[-1]["content"])
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
        auto_contacts_df = get_contacts_df(Session, manual=False)
        if not auto_contacts_df.empty:
            st.dataframe(auto_contacts_df, use_container_width=True)
            st.download_button(label="📥 ייצא רשימה אוטומטית לאקסל", data=to_excel(auto_contacts_df), file_name="אנשי_קשר_אוטומטי.xlsx")
        else: st.info("לא חולצו אנשי קשר באופן אוטומטי מהמסמכים.")
        st.markdown("---")
        if not IS_CLOUD:
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
            contacts_df = get_contacts_df(Session, manual=True)
            search_term = st.text_input("חפש איש קשר:")
            if search_term: contacts_df = contacts_df[contacts_df.apply(lambda row: search_term.lower() in ' '.join(row.astype(str)).lower(), axis=1)]
            if not contacts_df.empty:
                edited_df = st.data_editor(contacts_df, key="contacts_editor", use_container_width=True, hide_index=True, disabled=["מזהה"])
                st.download_button(label="📥 ייצא רשימה ידנית לאקסל", data=to_excel(edited_df.drop(columns=['מזהה'])), file_name="אנשי_קשר_ידני.xlsx")
                # Update logic for edited_df can be added here
        else:
            st.info("ניהול אנשי קשר אפשרי רק בגרסה המקומית.")
            manual_contacts_df = get_contacts_df(Session, manual=True)
            if not manual_contacts_df.empty:
                st.dataframe(manual_contacts_df.drop(columns=['מזהה']), use_container_width=True)
                st.download_button(label="📥 ייצא רשימה ידנית לאקסל", data=to_excel(manual_contacts_df.drop(columns=['מזהה'])), file_name="אנשי_קשר_ידני.xlsx")
            else: st.info("לא הוספו אנשי קשר באופן ידני.")

# --- Gatekeeper ---
if 'startup_complete' not in st.session_state:
    st.session_state.startup_complete = False
    st.session_state.api_key_ok = False
    st.session_state.db_ok = False

if not st.session_state.startup_complete:
    api_key = get_api_key()
    if api_key:
        st.session_state.api_key_ok = True
        Session = download_and_setup_database(
            db_url="https://huggingface.co/datasets/sagytb/reports/resolve/main/reports.sqlite",
            local_path="reports.sqlite"
        )
        if Session:
            st.session_state.db_ok = True
            st.session_state.Session = Session
            st.session_state.startup_complete = True
            st.experimental_rerun()
    else:
        st.stop()
else:
    # If startup is complete, run the main app UI
    main_ui(st.session_state.Session)
