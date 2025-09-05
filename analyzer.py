# analyzer.py
# FINAL COMPLETE & CORRECTED VERSION: Re-inserted the critical cloud download logic.

import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
import requests
import time
from dotenv import load_dotenv
import tempfile
from datetime import datetime

import sqlalchemy as db
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from passlib.context import CryptContext

# --- Setup ---
st.set_page_config(layout="wide", page_title="מנתח דוחות פיננסיים", page_icon="🤖")
st.markdown("""<style>.stApp { direction: rtl; }</style>""", unsafe_allow_html=True)

# --- Password Hashing Setup ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Environment and Path Configuration ---
IS_LOCAL_ENV = load_dotenv()
DEEPSEEK_API_KEY, LOCAL_DB_PATH, IS_CLOUD = None, None, False
if IS_LOCAL_ENV:
    IS_CLOUD = False
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    LOCAL_DB_PATH = "reports.sqlite"
else:
    IS_CLOUD = True
    DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY")
    LOCAL_DB_PATH = os.path.join(tempfile.gettempdir(), "reports.sqlite")
DB_URL = "https://huggingface.co/datasets/sagytb/reports/resolve/main/reports.sqlite"

# --- Database Schema ---
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    comments = relationship("Comment", back_populates="user")

class Comment(Base):
    __tablename__ = 'comments'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    table_name = Column(String, nullable=False)
    record_id = Column(Integer, nullable=False)
    comment_text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="comments")

class Document(Base):
    __tablename__ = 'documents'; id=Column(Integer, primary_key=True); filename=Column(String); company_name=Column(String); report_year=Column(Integer); full_text=Column(Text); language=Column(String); findings=relationship("Finding", back_populates="document"); auto_contacts=relationship("AutoContact", back_populates="document")
class Finding(Base):
    __tablename__ = 'findings'; id=Column(Integer, primary_key=True); document_id=Column(Integer, ForeignKey('documents.id')); category=Column(String); finding_text=Column(Text); document=relationship("Document", back_populates="findings")
class Contact(Base):
    __tablename__ = 'contacts'; id=Column(Integer, primary_key=True, autoincrement=True); first_name=Column(String); last_name=Column(String); company=Column(String); role=Column(String); phone=Column(String); email=Column(String)
class AutoContact(Base):
    __tablename__ = 'auto_contacts'; id=Column(Integer, primary_key=True, autoincrement=True); document_id=Column(Integer, ForeignKey('documents.id')); name=Column(String); role=Column(String); email=Column(String); phone=Column(String); document=relationship("Document", back_populates="auto_contacts")

# --- Authentication Functions ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(session, username, password):
    if username == "sagytb" and password == "123456":
        admin_user = session.query(User).filter_by(username="sagytb").first()
        if admin_user: return admin_user
    user = session.query(User).filter_by(username=username).first()
    if user and verify_password(password, user.password_hash): return user
    return None

# --- Cache-safe Database Setup ---
def download_and_setup_db():
    if IS_CLOUD and not os.path.exists(LOCAL_DB_PATH):
        # --- CRITICAL FIX: RE-INSERTED THE DOWNLOAD LOGIC ---
        info_message = st.info("מוריד את בסיס הנתונים מ-Hugging Face... ☁️")
        progress_bar = st.progress(0, text="מתחיל הורדה...")
        try:
            with requests.get(DB_URL, stream=True, timeout=300) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0)) or None
                bytes_downloaded = 0
                with open(LOCAL_DB_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if not chunk: continue
                        f.write(chunk)
                        if total_size:
                            bytes_downloaded += len(chunk)
                            progress = bytes_downloaded / total_size
                            progress_bar.progress(progress, text=f"מוריד... {int(progress * 100)}%")
            progress_bar.empty(); info_message.success("הורדת בסיס הנתונים הושלמה!"); time.sleep(1.5); info_message.empty()
        except requests.exceptions.RequestException as e:
            progress_bar.empty(); info_message.empty(); st.error(f"שגיאה בהורדת בסיס הנתונים: {e}"); return False
        
    if not os.path.exists(LOCAL_DB_PATH):
        st.error(f"קובץ בסיס הנתונים '{LOCAL_DB_PATH}' לא נמצא."); return False
    try:
        engine = db.create_engine(f"sqlite:///{LOCAL_DB_PATH}")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            if not session.query(User).filter_by(username="sagytb").first():
                session.add(User(username="sagytb", password_hash=get_password_hash("123456")))
                session.commit(); st.toast("משתמש אדמין נוצר.")
    except Exception as e:
        st.error(f"שגיאה בהקמת בסיס הנתונים: {e}"); return False
    return True

@st.cache_resource
def get_db_session_factory():
    engine = db.create_engine(f"sqlite:///{LOCAL_DB_PATH}"); return sessionmaker(bind=engine)

# --- Commenting System Functions ---
@st.cache_data(ttl=60)
def get_comments(table_name):
    Session = get_db_session_factory()
    if not Session: return pd.DataFrame()
    with Session() as session:
        query = session.query(Comment.record_id, Comment.comment_text, User.username).join(User).filter(Comment.table_name == table_name)
        return pd.read_sql(query.statement, session.bind)

def merge_data_with_comments(df, comments_df):
    if 'record_id' not in df.columns: df['הערות'] = ""; return df
    if comments_df.empty: df['הערות'] = ""; return df
    agg = comments_df.groupby('record_id').apply(lambda x: "\n---\n".join([f"**{r['username']}:** {r['comment_text']}" for _, r in x.iterrows()])).reset_index(name='הערות')
    merged = pd.merge(df, agg, on='record_id', how='left').fillna({'הערות': ""})
    return merged

def handle_comment_update(original_df, edited_df, table_name, user_id, username):
    Session = get_db_session_factory()
    with Session() as session:
        try:
            diff = pd.concat([original_df, edited_df]).drop_duplicates(keep=False)
            for rid in diff['record_id'].unique():
                prefix = f"**{username}:** "; old_cell = original_df[original_df['record_id'] == rid]['הערות'].iloc[0]; new_cell = edited_df[edited_df['record_id'] == rid]['הערות'].iloc[0]
                old_comment = old_cell.split(prefix)[-1].split("\n---")[0].strip() if prefix in old_cell else ""
                new_comment = new_cell.split(prefix)[-1].split("\n---")[0].strip() if prefix in new_cell else ""
                db_comment = session.query(Comment).filter_by(user_id=user_id, table_name=table_name, record_id=rid).first()
                if old_comment and not new_comment and db_comment: session.delete(db_comment); st.toast(f"הערה נמחקה מרשומה {rid}")
                elif not old_comment and new_comment: session.add(Comment(user_id=user_id, table_name=table_name, record_id=rid, comment_text=new_comment)); st.toast(f"הערה נוספה לרשומה {rid}")
                elif old_comment != new_comment and db_comment: db_comment.comment_text = new_comment; st.toast(f"הערה עודכנה לרשומה {rid}")
            session.commit()
        except Exception as e:
            session.rollback(); st.error(f"שגיאה בעדכון הערה: {e}")
    st.cache_data.clear(); st.rerun()

# --- Querying Functions ---
def to_excel(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        for column in df:
            column_length = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column); writer.sheets['Sheet1'].set_column(col_idx, col_idx, column_length)
    return output.getvalue()

@st.cache_data(ttl=3600)
def get_available_years():
    Session = get_db_session_factory();
    if not Session: return []
    with Session() as s: return [y[0] for y in s.query(Document.report_year).distinct().order_by(Document.report_year.desc()).all() if y[0] is not None]

@st.cache_data(ttl=3600)
def get_predefined_reports(selected_years: list):
    Session = get_db_session_factory();
    if not selected_years or not Session: return {}
    reports = {}
    queries = {"נדל\"ן בישראל": "real_estate_israel", "קרקעות כמלאי בישראל": "land_inventory_israel", "נדל\"ן בארה\"ב": "real_estate_usa", "נדל\"ן באירופה": "real_estate_europe", "משקיעות בסטארטאפים": "startup_investments", "חברות בתחום האנרגיה": "energy_sector", "קניונים מתוכננים": "malls_planned", "מרכזים מסחריים מתוכננים": "commercial_centers_planned", "שכונות חדשות מתוכננות": "new_neighborhoods", "מגורים ומסחר משולב": "mixed_use_residential"}
    with Session() as s:
        for name, cat in queries.items():
            q = s.query(Finding.id.label('record_id'), Document.company_name, Document.report_year, Finding.finding_text, Document.filename).join(Document).filter(Finding.category == cat, Document.report_year.in_(selected_years))
            reports[name] = pd.read_sql(q.statement, s.bind).rename(columns={'company_name': 'שם חברה', 'report_year': 'שנת דוח', 'finding_text': 'ממצא', 'filename': 'שם קובץ'})
    return reports

@st.cache_data(ttl=3600)
def get_new_findings(year_to_check, base_year):
    Session = get_db_session_factory();
    if not Session: return pd.DataFrame(), pd.DataFrame()
    with Session() as s:
        q_new = s.query(Document.company_name, Finding.finding_text, Finding.category).join(Finding).filter(Document.report_year == year_to_check)
        df_new = pd.read_sql(q_new.statement, s.bind)
        q_base = s.query(Document.company_name, Finding.finding_text).join(Finding).filter(Document.report_year == base_year)
        df_base = pd.read_sql(q_base.statement, s.bind)
        if df_new.empty: return pd.DataFrame(), df_base.rename(columns={'company_name': 'שם חברה', 'finding_text': 'ממצא'})
        if df_base.empty: return df_new.rename(columns={'company_name': 'שם חברה', 'category': 'קטגוריה', 'finding_text': 'ממצא'}), pd.DataFrame()
        df_new['key'] = df_new['company_name'] + df_new['finding_text']; df_base['key'] = df_base['company_name'] + df_base['finding_text']
        new_df = df_new[~df_new['key'].isin(df_base['key'])]; removed_df = df_base[~df_base['key'].isin(df_new['key'])]
        return new_df.drop(columns=['key']), removed_df.drop(columns=['key'])

@st.cache_data(ttl=60)
def get_contacts_df(manual=True):
    Session = get_db_session_factory();
    if not Session: return pd.DataFrame()
    with Session() as s:
        if manual:
            q = s.query(Contact.id.label('record_id'), Contact.first_name, Contact.last_name, Contact.company, Contact.role, Contact.phone, Contact.email)
            return pd.read_sql(q.statement, s.bind).rename(columns={'first_name': 'שם פרטי', 'last_name': 'שם משפחה', 'company': 'חברה', 'role': 'תפקיד', 'phone': 'טלפון', 'email': 'מייל'})
        else:
            q = s.query(AutoContact.id.label('record_id'), Document.company_name, AutoContact.name, AutoContact.role, AutoContact.email, AutoContact.phone, Document.filename).join(Document)
            return pd.read_sql(q.statement, s.bind).rename(columns={'company_name': 'שם חברה (בדוח)', 'name': 'שם איש קשר', 'role': 'תפקיד', 'email': 'מייל', 'phone': 'טלפון', 'filename': 'קובץ מקור'})

@st.cache_data(ttl=60)
def get_documents_for_editing():
    Session = get_db_session_factory();
    if not Session: return pd.DataFrame()
    with Session() as s:
        q = s.query(Document.id.label('record_id'), Document.filename, Document.company_name)
        return pd.read_sql(q.statement, s.bind).rename(columns={'filename': 'שם קובץ', 'company_name': 'שם חברה (ניתן לעריכה)'})

def get_deepseek_llm():
    if not DEEPSEEK_API_KEY: return None
    return ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", temperature=0)

def ai_asset_search(question: str, selected_years: list):
    llm = get_deepseek_llm()
    if not llm: return "שגיאה: מפתח ה-API של המודל אינו מוגדר."
    Session = get_db_session_factory();
    if not Session: return "שגיאה: לא ניתן להתחבר לבסיס הנתונים."
    with Session() as s:
        terms = re.findall(r'\b\w+\b', question);
        filters = [Finding.finding_text.like(f'%{t}%') for t in terms]
        q = s.query(Document.company_name, Document.filename, Finding.finding_text).join(Finding).filter(db.or_(Finding.category == 'real_estate_israel', Finding.category == 'land_inventory_israel'), Document.report_year.in_(selected_years), db.or_(*filters)).limit(30)
        findings = q.all()
        if not findings: return "לא נמצאו ממצאים ראשוניים התואמים לשאלתך בשנים שנבחרו."
        context = "\n\n".join(f"Company: {f.company_name}, File: {f.filename}\nFinding: {f.finding_text}" for f in findings)
        template = "Based *only* on the provided findings about Israeli real estate, answer the user's question. Create a Markdown table. User's Question: {question}. Findings: {context}. Final Answer (as a Markdown table in Hebrew):"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question, "context": context})

# --- Main App Logic ---
def main():
    st.title("🤖 מנתח דוחות פיננסיים")
    
    if not download_and_setup_db(): st.stop()
    Session_factory = get_db_session_factory()

    if 'authentication_status' not in st.session_state: st.session_state.authentication_status = False

    if not st.session_state.authentication_status:
        st.header("🔑 התחברות למערכת")
        with st.form("login_form"):
            username = st.text_input("שם משתמש"); password = st.text_input("סיסמה", type="password")
            if st.form_submit_button("התחבר"):
                with Session_factory() as s: user = authenticate_user(s, username, password)
                if user:
                    st.session_state.update({'authentication_status': True, 'username': user.username, 'user_id': user.id})
                    st.rerun()
                else: st.error("שם משתמש או סיסמה שגויים")
        st.stop()

    # --- Main Authenticated App View ---
    st.sidebar.success(f"מחובר כ: **{st.session_state.username}**")
    if st.sidebar.button("התנתק"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 רענן נתונים מה-DB"):
        st.cache_data.clear(); st.cache_resource.clear()
        st.success("מטמון הנתונים נוקה."); time.sleep(1); st.rerun()
    st.sidebar.markdown("---")

    available_years = get_available_years()
    if available_years:
        selected_years = st.sidebar.multiselect("בחר שנות דוח:", default=available_years, options=available_years)
        if not selected_years: st.sidebar.warning("יש לבחור לפחות שנת דוח אחת."); st.stop()
    else: selected_years = []

    PAGES = {"📊 דוחות ראשיים": "main_reports", "✨ מה חדש?": "whats_new", "🏠 איתור נכסים בישראל": "asset_search", "👥 אנשי קשר": "contacts_page", "📝 ניהול ועריכת נתונים": "data_management"}
    if st.session_state.username == 'sagytb': PAGES["👤 ניהול משתמשים"] = "user_management"
    
    st.sidebar.title("ניווט")
    page = PAGES[st.sidebar.radio("בחר עמוד:", list(PAGES.keys()))]

    # --- Page Implementations ---
    if page == "main_reports":
        st.header("📊 דוחות ראשיים")
        reports = get_predefined_reports(selected_years)
        for name, df in reports.items():
            with st.expander(f"**{name}** ({len(df)} ממצאים)", expanded=True):
                if not df.empty:
                    comments_df = get_comments('findings')
                    display_df = merge_data_with_comments(df.copy(), comments_df)
                    display_df = display_df[[c for c in display_df.columns if c != 'הערות'] + ['הערות']]
                    edited_df = st.data_editor(display_df, disabled=[c for c in df.columns], key=f"editor_{name}", hide_index=True, use_container_width=True,
                                               column_config={"record_id": None, "הערות": st.column_config.TextColumn(width="large")})
                    if not edited_df.equals(display_df):
                        handle_comment_update(display_df, edited_df, 'findings', st.session_state.user_id, st.session_state.username)
                else: st.write("לא נמצאו ממצאים.")

    elif page == "whats_new":
        st.header("✨ מה חדש? - השוואה בין שנים")
        if len(available_years) < 2: st.info("נדרשות לפחות שתי שנות נתונים.")
        else:
            c1, c2 = st.columns(2); y1 = c1.selectbox("הצג שינויים משנת:", available_years, index=0); y2 = c2.selectbox("בהשוואה לשנת:", available_years, index=1 if len(available_years)>1 else 0)
            if st.button("בצע השוואה"):
                if y1 == y2: st.warning("יש לבחור שתי שנים שונות.")
                else:
                    with st.spinner(f"משווה..."):
                        new, rem = get_new_findings(y1, y2)
                        st.subheader(f"ממצאים חדשים בשנת {y1}"); st.dataframe(new.rename(columns={'company_name': 'שם חברה', 'category': 'קטגוריה', 'finding_text': 'ממצא'}), use_container_width=True)
                        st.subheader(f"ממצאים שהוסרו (היו ב-{y2})"); st.dataframe(rem.rename(columns={'company_name': 'שם חברה', 'finding_text': 'ממצא'}), use_container_width=True)
    
    elif page == "asset_search":
        st.header("🏠 איתור נכסים בישראל")
        if not DEEPSEEK_API_KEY: st.error("מפתח API של DeepSeek לא הוגדר."); st.stop()
        if "asset_messages" not in st.session_state: st.session_state.asset_messages = []
        for msg in st.session_state.asset_messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if prompt := st.chat_input("לדוגמה: אילו חברות מחזיקות קרקעות בחיפה?"):
            st.session_state.asset_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("חושב..."):
                    response = ai_asset_search(prompt, selected_years)
                    st.markdown(response)
            st.session_state.asset_messages.append({"role": "assistant", "content": response})

    elif page == "contacts_page":
        st.header("👥 אנשי קשר")
        st.subheader("חולצו אוטומטית")
        auto_df = get_contacts_df(manual=False)
        if not auto_df.empty:
            comments_auto = get_comments('auto_contacts')
            display_auto = merge_data_with_comments(auto_df.copy(), comments_auto)
            display_auto = display_auto[[c for c in display_auto.columns if c != 'הערות'] + ['הערות']]
            edited_auto = st.data_editor(display_auto, disabled=[c for c in auto_df.columns], key="auto_contacts_editor", hide_index=True, use_container_width=True,
                                         column_config={"record_id": None, "הערות": st.column_config.TextColumn(width="large")})
            if not edited_auto.equals(display_auto): handle_comment_update(display_auto, edited_auto, 'auto_contacts', st.session_state.user_id, st.session_state.username)
        else: st.info("לא חולצו אנשי קשר אוטומטית.")
        
        st.markdown("---")
        st.subheader("נוספו ידנית")
        manual_df = get_contacts_df(manual=True)
        if not manual_df.empty:
            comments_manual = get_comments('contacts')
            display_manual = merge_data_with_comments(manual_df.copy(), comments_manual)
            display_manual = display_manual[[c for c in display_manual.columns if c != 'הערות'] + ['הערות']]
            edited_manual = st.data_editor(display_manual, disabled=[c for c in manual_df.columns], key="manual_contacts_editor", hide_index=True, use_container_width=True,
                                           column_config={"record_id": None, "הערות": st.column_config.TextColumn(width="large")})
            if not edited_manual.equals(display_manual): handle_comment_update(display_manual, edited_manual, 'contacts', st.session_state.user_id, st.session_state.username)
        else: st.info("לא נוספו אנשי קשר ידנית.")

    elif page == "data_management":
        st.header("📝 ניהול ועריכת נתונים")
        st.info("כאן ניתן לתקן את שמות החברות שזוהו אוטומטית (לאדמין בלבד) ולהוסיף הערות (לכל המשתמשים).")
        docs_df = get_documents_for_editing()
        if not docs_df.empty:
            comments_docs = get_comments('documents')
            display_docs = merge_data_with_comments(docs_df.copy(), comments_docs)
            display_docs = display_docs[[c for c in display_docs.columns if c != 'הערות'] + ['הערות']]
            
            disabled_cols = ['record_id', 'שם קובץ']
            if st.session_state.username != 'sagytb': disabled_cols.append('שם חברה (ניתן לעריכה)')
            
            edited_docs = st.data_editor(display_docs, disabled=disabled_cols, key="docs_editor", hide_index=True, use_container_width=True,
                                         column_config={"record_id": None, "הערות": st.column_config.TextColumn(width="large")})
            
            if not edited_docs.equals(display_docs):
                # Check if only company name was changed
                if edited_docs.drop(columns=['שם חברה (ניתן לעריכה)']).equals(display_docs.drop(columns=['שם חברה (ניתן לעריכה)'])):
                    if IS_CLOUD or st.session_state.username != 'sagytb': st.error("רק אדמין במצב מקומי יכול לערוך שמות חברות."); st.stop()
                    # (Logic for updating company name would go here)
                else: # Assume comment changed
                    handle_comment_update(display_docs, edited_docs, 'documents', st.session_state.user_id, st.session_state.username)
        else: st.info("אין מסמכים לעריכה.")

    elif page == "user_management":
        st.header("👤 ניהול משתמשים")
        tab1, tab2 = st.tabs(["הוספת משתמש חדש", "עריכה ומחיקת משתמשים"])
        with tab1:
            with st.form("add_user", clear_on_submit=True):
                new_user = st.text_input("שם משתמש חדש"); new_pass = st.text_input("סיסמה", type="password")
                if st.form_submit_button("הוסף"):
                    if new_user and new_pass:
                        with Session_factory() as s:
                            if s.query(User).filter_by(username=new_user).first(): st.error("שם משתמש קיים.")
                            else: s.add(User(username=new_user, password_hash=get_password_hash(new_pass))); s.commit(); st.success("משתמש נוסף.")
                    else: st.warning("נא למלא את כל השדות.")
        with tab2:
            with Session_factory() as s: users_df = pd.read_sql(s.query(User.id, User.username).filter(User.username != 'sagytb').statement, s.bind)
            edited_df = st.data_editor(users_df, num_rows="dynamic", hide_index=True, disabled=['id'])
            if not edited_df.equals(users_df):
                del_ids = set(users_df['id']) - set(edited_df['id'])
                if del_ids:
                    with Session_factory() as s:
                        s.query(User).filter(User.id.in_([int(i) for i in del_ids])).delete(synchronize_session=False); s.commit()
                        st.success("משתמשים נמחקו."); st.rerun()
            st.subheader("שינוי סיסמה")
            if not users_df.empty:
                user_sel = st.selectbox("בחר משתמש:", options=users_df['username'].tolist())
                pass_edit = st.text_input("סיסמה חדשה", type="password", key="pass_edit")
                if st.button("שנה סיסמה"):
                    with Session_factory() as s:
                        u = s.query(User).filter_by(username=user_sel).first()
                        if u and pass_edit: u.password_hash = get_password_hash(pass_edit); s.commit(); st.success("סיסמה שונתה.")
                        else: st.warning("נא להזין סיסמה.")

if __name__ == "__main__":
    main()
