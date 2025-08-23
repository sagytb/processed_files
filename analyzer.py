# analyzer.py
# FINAL COMPLETE & ROBUST HYBRID VERSION: Corrected all scope and caching errors.

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

# --- Global Setup (runs once) ---
st.set_page_config(layout="wide", page_title="מנתח דוחות פיננסיים", page_icon="🤖")
st.markdown("""<style>.stApp { direction: rtl; }</style>""", unsafe_allow_html=True)

load_dotenv()
IS_CLOUD = os.environ.get('STREAMLIT_SERVER_RUNNING_IN_CLOUD', 'false').lower() == 'true'
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY") if IS_CLOUD else os.getenv("DEEPSEEK_API_KEY")

DB_URL = "https://huggingface.co/datasets/sagytb/reports/resolve/main/reports.sqlite?download=true"
LOCAL_DB_PATH = "reports.sqlite"

# --- Database Schema (Defined Globally) ---
Base = declarative_base()
class Document(Base):
    __tablename__ = 'documents'; id=Column(Integer, primary_key=True); filename=Column(String); company_name=Column(String); report_year=Column(Integer); full_text=Column(Text); language=Column(String); findings=relationship("Finding", back_populates="document"); auto_contacts=relationship("AutoContact", back_populates="document")
class Finding(Base):
    __tablename__ = 'findings'; id=Column(Integer, primary_key=True); document_id=Column(Integer, ForeignKey('documents.id')); category=Column(String); finding_text=Column(Text); document=relationship("Document", back_populates="findings")
class Contact(Base):
    __tablename__ = 'contacts'; id=Column(Integer, primary_key=True, autoincrement=True); first_name=Column(String); last_name=Column(String); company=Column(String); role=Column(String); phone=Column(String); email=Column(String)
class AutoContact(Base):
    __tablename__ = 'auto_contacts'; id=Column(Integer, primary_key=True, autoincrement=True); document_id=Column(Integer, ForeignKey('documents.id')); name=Column(String); role=Column(String); email=Column(String); phone=Column(String); document=relationship("Document", back_populates="auto_contacts")

# --- Database Connection Function (Cached) ---
@st.cache_resource
def get_db_session_factory():
    if IS_CLOUD and not os.path.exists(LOCAL_DB_PATH):
        info_message = st.info("מוריד את בסיס הנתונים מ-Hugging Face... ☁️")
        try:
            r = requests.get(DB_URL, stream=True, timeout=120); r.raise_for_status()
            with open(LOCAL_DB_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            info_message.success("הורדת בסיס הנתונים הושלמה!"); time.sleep(2); info_message.empty()
        except requests.exceptions.RequestException as e:
            info_message.error(f"שגיאה בהורדת בסיס הנתונים: {e}"); return None
            
    if not os.path.exists(LOCAL_DB_PATH):
        st.error(f"קובץ בסיס הנתונים '{LOCAL_DB_PATH}' לא נמצא. אנא הרץ 'process_files.py'."); return None

    engine = db.create_engine(f'sqlite:///{LOCAL_DB_PATH}')
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

# --- Querying Functions (Now get the factory internally) ---
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
def get_available_years():
    Session = get_db_session_factory()
    if not Session: return []
    session = Session()
    try:
        years = session.query(Document.report_year).distinct().order_by(Document.report_year.desc()).all()
        return [y[0] for y in years if y[0] is not None]
    finally: session.close()

@st.cache_data(ttl=3600)
def get_predefined_reports(selected_years: list):
    Session = get_db_session_factory()
    if not selected_years or not Session: return {}
    session = Session(); reports = {}
    report_queries = {"נדל\"ן בישראל": "real_estate_israel", "קרקעות כמלאי בישראל": "land_inventory_israel", "נדל\"ן בארה\"ב": "real_estate_usa", "נדל\"ן באירופה": "real_estate_europe", "משקיעות בסטארטאפים": "startup_investments", "חברות בתחום האנרגיה": "energy_sector", "קניונים מתוכננים": "malls_planned", "מרכזים מסחריים מתוכננים": "commercial_centers_planned", "שכונות חדשות מתוכננות": "new_neighborhoods", "מגורים ומסחר משולב": "mixed_use_residential"}
    try:
        for name, category in report_queries.items():
            query = (session.query(Document.company_name, Document.report_year, Finding.finding_text, Document.filename).join(Finding).filter(Finding.category == category).filter(Document.report_year.in_(selected_years)))
            reports[name] = pd.read_sql(query.statement, session.bind)
    finally: session.close()
    return reports

@st.cache_data(ttl=3600)
def get_new_findings(year_to_check, base_year):
    Session = get_db_session_factory()
    if not Session: return pd.DataFrame(), pd.DataFrame()
    session = Session()
    try:
        query_new = (session.query(Document.company_name, Finding.finding_text, Finding.category).join(Finding).filter(Document.report_year == year_to_check))
        df_new = pd.read_sql(query_new.statement, session.bind)
        query_base = (session.query(Document.company_name, Finding.finding_text).join(Finding).filter(Document.report_year == base_year))
        df_base = pd.read_sql(query_base.statement, session.bind)
        if df_new.empty: return pd.DataFrame(), df_base.rename(columns={'company_name': 'שם חברה', 'finding_text': 'ממצא'})
        if df_base.empty: return df_new.rename(columns={'company_name': 'שם חברה', 'category': 'קטגוריה', 'finding_text': 'ממצא'}), pd.DataFrame()
        df_new['key'] = df_new['company_name'] + df_new['finding_text']; df_base['key'] = df_base['company_name'] + df_base['finding_text']
        new_findings_df = df_new[~df_new['key'].isin(df_base['key'])]; removed_findings_df = df_base[~df_base['key'].isin(df_new['key'])]
        return new_findings_df.drop(columns=['key']), removed_findings_df.drop(columns=['key'])
    finally: session.close()

@st.cache_data(ttl=60)
def get_contacts_df(manual=True):
    Session = get_db_session_factory()
    if not Session: return pd.DataFrame()
    session = Session()
    try:
        if manual:
            query = session.query(Contact).statement; df = pd.read_sql(query, session.bind); return df.rename(columns={'id': 'מזהה', 'first_name': 'שם פרטי', 'last_name': 'שם משפחה', 'company': 'חברה', 'role': 'תפקיד', 'phone': 'טלפון', 'email': 'מייל'})
        else:
            query = (session.query(Document.company_name, AutoContact.name, AutoContact.role, AutoContact.email, AutoContact.phone, Document.filename).join(AutoContact)); df = pd.read_sql(query.statement, session.bind); return df.rename(columns={'company_name': 'שם חברה (בדוח)', 'name': 'שם איש קשר', 'role': 'תפקיד', 'email': 'מייל', 'phone': 'טלפון', 'filename': 'קובץ מקור'})
    finally: session.close()

@st.cache_data(ttl=60)
def get_documents_for_editing():
    Session = get_db_session_factory()
    if not Session: return pd.DataFrame()
    session = Session()
    try:
        query = session.query(Document.id, Document.filename, Document.company_name).statement; df = pd.read_sql(query, session.bind)
        return df.rename(columns={'id': 'מזהה', 'filename': 'שם קובץ', 'company_name': 'שם חברה (ניתן לעריכה)'})
    finally: session.close()

def get_deepseek_llm(api_key):
    return ChatOpenAI(model="deepseek-chat", api_key=api_key, base_url="https://api.deepseek.com/v1", temperature=0)

def ai_asset_search(api_key, question: str, selected_years: list):
    Session = get_db_session_factory()
    if not Session: return "שגיאה: לא ניתן להתחבר לבסיס הנתונים."
    session = Session()
    try:
        search_terms = re.findall(r'\b\w+\b', question); finding_filter = [Finding.finding_text.like(f'%{term}%') for term in search_terms]; candidate_findings = (session.query(Document.company_name, Document.filename, Finding.finding_text).join(Finding).filter(db.or_(Finding.category == 'real_estate_israel', Finding.category == 'land_inventory_israel'), Document.report_year.in_(selected_years), db.or_(*finding_filter)).limit(30).all())
        if not candidate_findings: return "לא נמצאו ממצאים ראשוניים התואמים לשאלתך בשנים שנבחרו."
        context = "\n\n".join(f"Company: {f.company_name}, File: {f.filename}\nFinding: {f.finding_text}" for f in candidate_findings); summarize_template = "Based *only* on the provided findings about Israeli real estate, answer the user's question. Create a Markdown table. User's Question: {question}. Findings: {context}. Final Answer (as a Markdown table in Hebrew):"; prompt = ChatPromptTemplate.from_template(summarize_template); chain = prompt | get_deepseek_llm(api_key) | StrOutputParser(); return chain.invoke({"question": question, "context": context})
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

# --- Main App Logic ---
def main():
    st.title("🤖 מנתח דוחות פיננסיים")
    if not DEEPSEEK_API_KEY: 
        st.error("מפתח API של DeepSeek לא הוגדר."); 
        st.stop()

    Session_factory = get_db_session_factory()
    if not Session_factory:
        st.stop()

    available_years = get_available_years()
    if available_years:
        selected_years = st.sidebar.multiselect("בחר שנות דוח להצגה:", options=available_years, default=available_years)
        if not selected_years: st.sidebar.warning("יש לבחור לפחות שנת דוח אחת."); st.stop()
    else:
        selected_years = []

    PAGES = {"📊 דוחות ראשיים": "main_reports", "✨ מה חדש?": "whats_new", "🏠 איתור נכסים בישראל": "asset_search", "👥 אנשי קשר": "contacts_page", "📝 ניהול ועריכת נתונים": "data_management"}
    st.sidebar.title("ניווט"); 
    selection = st.sidebar.radio("בחר עמוד:", list(PAGES.keys())); 
    page = PAGES[selection]
    
    if page == "main_reports":
        st.header(f"דוחות מסכמים עבור השנים: {', '.join(map(str, sorted(selected_years)))}")
        st.info("הדוחות להלן מציגים ממצאים שחולצו אוטומטית מכלל המסמכים.")
        reports = get_predefined_reports(selected_years)
        for report_name, df in reports.items():
            with st.expander(f"**{report_name}** ({len(df)} ממצאים)"):
                if not df.empty:
                    display_df = df.rename(columns={'company_name': 'שם חברה', 'report_year': 'שנת דוח', 'finding_text': 'ממצא', 'filename': 'שם קובץ'}); 
                    st.dataframe(display_df, use_container_width=True, hide_index=True); 
                    st.download_button(label=f"📥 ייצא את רשימת '{report_name}' לאקסל", data=to_excel(display_df), file_name=f"{report_name.replace('\"', '')}.xlsx", key=f"export_{report_name}")
                else: 
                    st.write("לא נמצאו ממצאים בקטגוריה זו עבור השנים שנבחרו.")
    
    elif page == "whats_new":
        st.header("מה חדש? - השוואה בין שנים")
        if len(available_years) < 2: st.info("נדרשות לפחות שתי שנות נתונים כדי לבצע השוואה.")
        else:
            st.info("כלי זה מציג ממצאים שהופיעו בשנה החדשה אך לא הופיעו בשנה הישנה, וממצאים שהוסרו.")
            c1, c2 = st.columns(2); 
            compare_year = c1.selectbox("הצג שינויים משנת:", available_years, index=0); 
            base_year = c2.selectbox("בהשוואה לשנת:", available_years, index=1 if len(available_years)>1 else 0)
            if st.button("בצע השוואה"):
                if compare_year == base_year: st.warning("יש לבחור שתי שנים שונות להשוואה.")
                else:
                    with st.spinner(f"משווה את {compare_year} מול {base_year}..."):
                        new, removed = get_new_findings(compare_year, base_year)
                        st.subheader(f"ממצאים חדשים בשנת {compare_year}"); 
                        st.dataframe(new.rename(columns={'company_name': 'שם חברה', 'category': 'קטגוריה', 'finding_text': 'ממצא'}), use_container_width=True) if not new.empty else st.write("לא נמצאו ממצאים חדשים.")
                        st.subheader(f"ממצאים שהוסרו (היו ב-{base_year}, נעלמו ב-{compare_year})"); 
                        st.dataframe(removed.rename(columns={'company_name': 'שם חברה', 'finding_text': 'ממצא'}), use_container_width=True) if not removed.empty else st.write("לא נמצאו ממצאים שהוסרו.")

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
                        response = ai_asset_search(st.session_state.asset_messages[-1]["content"], selected_years)
                        st.markdown(response); st.session_state.asset_messages.append({"role": "assistant", "content": response}); st.rerun()
        with tab2:
            st.info("הזן מונח (עיר, רחוב, סוג נכס) לחיפוש מהיר בכל הטקסטים של הדוחות.")
            keyword = st.text_input("הזן מונח לחיפוש:", key="keyword_search")
            if st.button("חפש", key="search_button"):
                if keyword:
                    with st.spinner("מחפש..."):
                        session = Session_factory(); results = session.query(Document.company_name, Document.filename, Document.full_text).filter(Document.full_text.like(f'%{keyword}%'), Document.report_year.in_(selected_years)).limit(50).all(); session.close()
                        if results:
                            st.success(f"נמצאו {len(results)} מסמכים המכילים את המונח '{keyword}' בשנים שנבחרו:")
                            for doc in results:
                                try:
                                    safe_keyword = re.escape(keyword); match = re.search(safe_keyword, doc.full_text, re.IGNORECASE)
                                    if match:
                                        snippet = doc.full_text[max(0, match.start() - 80):match.end() + 80]
                                        highlighted_snippet = re.sub(f'({safe_keyword})', r'**\1**', snippet, flags=re.IGNORECASE)
                                        st.info(f"**חברה:** {doc.company_name} | **קובץ:** {doc.filename}"); st.markdown(f'<div dir="rtl">...{highlighted_snippet}...</div>', unsafe_allow_html=True)
                                except Exception as e: st.warning(f"לא ניתן היה להציג קטע מתוך '{doc.filename}': {e}")
                        else: st.warning(f"המונח '{keyword}' לא נמצא באף מסמך בשנים שנבחרו.")
                else: st.warning("אנא הזן מונח לחיפוש.")
    
    elif page == "contacts_page":
        st.header("ניהול אנשי קשר")
        st.subheader("אנשי קשר שחולצו אוטומטית מהדוחות")
        auto_contacts_df = get_contacts_df(manual=False)
        if not auto_contacts_df.empty:
            st.dataframe(auto_contacts_df, use_container_width=True, hide_index=True)
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
                        session = Session_factory();
                        try:
                            new_contact = Contact(first_name=first_name, last_name=last_name, company=company, role=role, phone=phone, email=email)
                            session.add(new_contact); session.commit(); st.success(f"איש הקשר '{first_name} {last_name}' נשמר!"); st.cache_data.clear()
                        finally: session.close()
            st.markdown("---")
            st.subheader("רשימת אנשי קשר (ידנית)")
            contacts_df = get_contacts_df(manual=True)
            search_term = st.text_input("חפש איש קשר:")
            if search_term: contacts_df = contacts_df[contacts_df.apply(lambda row: search_term.lower() in ' '.join(row.astype(str)).lower(), axis=1)]
            if not contacts_df.empty:
                edited_df = st.data_editor(contacts_df, key="contacts_editor", use_container_width=True, hide_index=True, disabled=["מזהה"])
                st.download_button(label="📥 ייצא רשימה ידנית לאקסל", data=to_excel(edited_df.drop(columns=['מזהה'])), file_name="אנשי_קשר_ידני.xlsx")
                if not contacts_df.equals(edited_df):
                    session = Session_factory()
                    try:
                        changed_rows = pd.concat([contacts_df, edited_df]).drop_duplicates(keep=False)
                        for _, row in changed_rows.iterrows():
                            if row['מזהה'] in edited_df['מזהה'].values:
                                contact_to_update = session.query(Contact).filter_by(id=row['מזהה']).one()
                                contact_to_update.first_name, contact_to_update.last_name, contact_to_update.company, contact_to_update.role, contact_to_update.phone, contact_to_update.email = row['שם פרטי'], row['שם משפחה'], row['חברה'], row['תפקיד'], row['טלפון'], row['מייל']
                        session.commit(); st.toast("השינויים נשמרו!"); st.cache_data.clear(); st.rerun()
                    except Exception as e:
                        session.rollback(); st.error(f"שגיאה בעדכון: {e}")
                    finally: session.close()
            else: st.info("לא נמצאו אנשי קשר." if search_term else "עדיין לא נוספו אנשי קשר.")
        else:
            st.info("ניהול אנשי קשר (הוספה ועריכה) אפשרי רק בגרסה המקומית של האפליקציה.")
            st.subheader("רשימת אנשי קשר (ידנית)")
            manual_contacts_df = get_contacts_df(manual=True)
            if not manual_contacts_df.empty:
                st.dataframe(manual_contacts_df.drop(columns=['מזהה']), use_container_width=True, hide_index=True)
                st.download_button(label="📥 ייצא רשימה ידנית לאקסל", data=to_excel(manual_contacts_df.drop(columns=['מזהה'])), file_name="אנשי_קשר_ידני.xlsx")
            else: st.info("לא הוספו אנשי קשר באופן ידני.")

    elif page == "data_management":
        st.header("ניהול ועריכת נתונים")
        if IS_CLOUD: st.warning("עריכת נתונים אפשרית רק בגרסה המקומית של האפליקציה.")
        else:
            st.subheader("עריכת שמות חברות")
            st.info("כאן ניתן לתקן את שמות החברות שזוהו אוטומטית. לחץ על תא בטבלה, הקלד את השם החדש ולחץ Enter. השינוי יישמר אוטומטית.")
            docs_df = get_documents_for_editing()
            if not docs_df.empty:
                if 'original_docs' not in st.session_state or not st.session_state.original_docs.equals(docs_df):
                    st.session_state.original_docs = docs_df.copy()
                edited_df = st.data_editor(docs_df, key="docs_editor", use_container_width=True, hide_index=True, disabled=["מזהה", "שם קובץ"])
                if not st.session_state.original_docs.equals(edited_df):
                    session = Session_factory()
                    try:
                        diff = pd.concat([st.session_state.original_docs, edited_df]).drop_duplicates(keep=False)
                        for _, row in diff.iterrows():
                            if row['מזהה'] in edited_df['מזהה'].values:
                                new_name = edited_df.loc[edited_df['מזהה'] == row['מזהה'], 'שם חברה (ניתן לעריכה)'].iloc[0]
                                doc_to_update = session.query(Document).filter_by(id=row['מזהה']).one()
                                doc_to_update.company_name = new_name
                        session.commit(); st.toast("השינויים נשמרו!", icon="✔️"); st.cache_data.clear(); st.rerun()
                    except Exception as e:
                        session.rollback(); st.error(f"שגיאה בעדכון שם החברה: {e}")
                    finally: session.close()
            else: st.info("אין מסמכים בבסיס הנתונים לעריכה.")

if __name__ == "__main__":
    main()
