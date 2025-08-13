# process_files.py
# FINAL COMPLETE & ROBUST VERSION: Full document analysis via internal batching to ensure 100% coverage.

import fitz
import os
import re
import json
from pathlib import Path
import time
import logging
from dotenv import load_dotenv

import pytesseract
from PIL import Image
from langdetect import detect, LangDetectException

import sqlalchemy as db
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configure Tesseract Path ---
TESSERACT_CMD_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(TESSERACT_CMD_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
else:
    print("WARNING: Tesseract executable not found at the default path. OCR will fail.")
    print(f"Please install Tesseract or update TESSERACT_CMD_PATH in {__file__}")

# --- Logging Setup ---
# Console will show INFO level messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
# A separate file handler for detailed ERROR logs with tracebacks.
error_log_handler = logging.FileHandler("processing_errors.log", mode='w', encoding='utf-8')
error_log_handler.setLevel(logging.ERROR)
error_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - FILE: %(name)s - MESSAGE: %(message)s\n%(exc_info)s\n' + '-'*80))
logging.getLogger('').addHandler(error_log_handler)

# --- Setup and Constants ---
load_dotenv(); DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DB_FILE = "reports.sqlite"; MANIFEST_FILE = "processed_files_manifest.json"
# BATCH SIZE FOR INTERNAL DOCUMENT PROCESSING
FINDINGS_BATCH_SIZE = 15 # Process 15 chunks at a time per document

# --- Database Schema ---
engine = db.create_engine(f'sqlite:///{DB_FILE}')
Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, unique=True, nullable=False)
    company_name = Column(String)
    full_text = Column(Text)
    language = Column(String)
    findings = relationship("Finding", back_populates="document", cascade="all, delete-orphan")
    auto_contacts = relationship("AutoContact", back_populates="document", cascade="all, delete-orphan")

class Finding(Base):
    __tablename__ = 'findings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    category = Column(String, nullable=False)
    finding_text = Column(Text, nullable=False)
    document = relationship("Document", back_populates="findings")

class Contact(Base):
    __tablename__ = 'contacts'
    id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String)
    last_name = Column(String)
    company = Column(String)
    role = Column(String)
    phone = Column(String)
    email = Column(String)

class AutoContact(Base):
    __tablename__ = 'auto_contacts'
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    name = Column(String)
    role = Column(String)
    email = Column(String)
    phone = Column(String)
    document = relationship("Document", back_populates="auto_contacts")

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# --- Helper Functions ---
def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
            try: return json.load(f)
            except json.JSONDecodeError: return {}
    return {}

def save_manifest(manifest_data):
    with open(MANIFEST_FILE, 'w', encoding='utf-8') as f: json.dump(manifest_data, f, indent=4, ensure_ascii=False)

def get_file_identifier(file_path: Path):
    stat = file_path.stat(); return f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"

def detect_language(text_snippet: str) -> str:
    try:
        lang = detect(text_snippet); return {'he': 'hebrew', 'en': 'english'}.get(lang, 'mixed')
    except LangDetectException: return 'unknown'

def extract_company_name_multilingual(text_to_search: str, lang: str):
    if lang in ['hebrew', 'mixed', 'unknown']:
        patterns_he = [
            re.compile(r'([\w\s."\'()–-]+?)\s*\((?:להלן|להלן:)\s*["\']החברה["\']\)', re.IGNORECASE), 
            re.compile(r'^\s*([\w\s."\'()-]+?)\s+בע"מ', re.MULTILINE), 
            re.compile(r'([\w\s."\'()–-]+?)\s+(?:דוח|דו"ח)\s+(?:תקופתי|שנתי)', re.IGNORECASE), 
            re.compile(r'בשם\s+דירקטוריון\s+([\w\s."\'-]+?)\s+אני\s+מתכבד', re.IGNORECASE)
        ]
        for pattern in patterns_he:
            if match := pattern.search(text_to_search):
                name = match.group(1).strip().strip('."\'() \t\n');
                if len(name) > 2: return name
    if lang in ['english', 'mixed', 'unknown']:
        patterns_en = [
            re.compile(r'report\s+of\s+([\w\s.&,-]+?)(?:,?\s+(?:inc|ltd|llc))', re.IGNORECASE), 
            re.compile(r'^\s*([\w\s.&,-]+?)(?:,?\s+(?:inc|ltd|llc))\s*$', re.MULTILINE | re.IGNORECASE)
        ]
        for pattern in patterns_en:
            if match := pattern.search(text_to_search):
                name = match.group(1).strip();
                if len(name) > 2: return name
    return "חברה לא מזוהה"

def get_deepseek_llm():
    return ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", temperature=0, request_timeout=240)

def extract_findings_from_text(full_text: str) -> dict:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(full_text)
    num_chunks = len(chunks)
    logging.info(f"    - פוצל ל-{num_chunks} מקטעים (chunks) לניתוח פנימי.")

    aggregated_findings = {key: [] for key in ["real_estate_israel", "land_inventory_israel", "real_estate_usa", "real_estate_europe", "startup_investments", "energy_sector", "malls_planned", "commercial_centers_planned", "new_neighborhoods", "mixed_use_residential", "contacts_extracted"]}
    
    extraction_template = """
    You are an expert financial analyst. Your task is to analyze the provided text and extract specific findings for the categories below.
    For each category, return a list of strings (for findings) or a list of objects (for contacts).
    If no findings exist in THIS CHUNK of text for a category, return an empty list [].
    Respond ONLY with a valid JSON object.

    **CRITICAL RULES for GEOGRAPHIC CATEGORIES:**
    - For "real_estate_israel" and "land_inventory_israel", only include findings where the ASSET ITSELF is explicitly located in Israel.
    - For "real_estate_usa", only include findings where the ASSET ITSELF is explicitly located in the USA.
    - For "real_estate_europe", only include findings where the ASSET ITSELF is explicitly located in Europe.
    - Be extremely strict. If the location of the asset is not mentioned or doesn't match the category, DO NOT include the finding.

    Categories:
    1. "real_estate_israel": Findings about real estate assets LOCATED IN ISRAEL.
    2. "land_inventory_israel": Findings about LAND held as INVENTORY LOCATED IN ISRAEL.
    3. "real_estate_usa": Findings about real estate assets LOCATED IN THE USA.
    4. "real_estate_europe": Findings about real estate assets LOCATED IN EUROPE (e.g., UK, Germany, Spain, Poland).
    5. "startup_investments": Findings about investments in startups, high-tech, or venture capital.
    6. "energy_sector": Findings about activities or investments in the energy sector.
    7. "malls_planned": Findings about land designated for SHOPPING MALLS.
    8. "commercial_centers_planned": Findings about COMMERCIAL CENTERS planned for construction.
    9. "new_neighborhoods": Findings about land for NEW NEIGHBORHOODS.
    10. "mixed_use_residential": Findings about land for RESIDENTIAL WITH COMMERCIAL use.
    11. "contacts_extracted": A list of key personnel mentioned. Each item must be an object with keys "name", "role", "email", and "phone".
    
    Text chunk to analyze: --- {text_chunk} ---
    Your JSON response:
    """
    prompt = ChatPromptTemplate.from_template(extraction_template)
    llm = get_deepseek_llm()
    chain = prompt | llm | StrOutputParser()

    for i in range(0, num_chunks, FINDINGS_BATCH_SIZE):
        batch_chunks = chunks[i:i + FINDINGS_BATCH_SIZE]
        combined_chunk_text = "\n\n".join(batch_chunks)
        
        logging.info(f"      - מעבד קבוצת מקטעים {i//FINDINGS_BATCH_SIZE + 1} / {-(num_chunks // -FINDINGS_BATCH_SIZE)}...")
        
        response_str = chain.invoke({"text_chunk": combined_chunk_text})
        try:
            data = json.loads(response_str.strip().replace("```json", "").replace("```", ""))
            for key, value in data.items():
                if key in aggregated_findings and isinstance(value, list):
                    aggregated_findings[key].extend(value)
        except json.JSONDecodeError:
            logging.warning(f"Could not parse JSON from batch {i//FINDINGS_BATCH_SIZE + 1}. Response: {response_str}")

    # Remove duplicates from each list
    for key, value in aggregated_findings.items():
        if value and isinstance(value[0], dict): # Handle list of dicts (contacts)
             aggregated_findings[key] = [dict(t) for t in {tuple(sorted(d.items())) for d in value}]
        else: # Handle list of strings
            aggregated_findings[key] = list(dict.fromkeys(value))

    return aggregated_findings

def run_processing_pipeline(folder_path_str: str):
    if not DEEPSEEK_API_KEY: 
        logging.error("מפתח API של DeepSeek לא נמצא בקובץ .env. התהליך נעצר.")
        return
    root_path = Path(folder_path_str)
    if not root_path.is_dir(): 
        logging.error(f"הנתיב שסופק אינו תיקייה תקינה: '{folder_path_str}'. התהליך נעצר.")
        return

    logging.info("--- מתחיל תהליך עיבוד PDF מתקדם ---")
    
    all_pdf_paths = list(root_path.rglob('*.pdf'))
    manifest = load_manifest()
    files_to_process = [p for p in all_pdf_paths if get_file_identifier(p) not in manifest or manifest.get(get_file_identifier(p), {}).get('status') == 'failed']
    
    if not files_to_process: 
        logging.info("✅ כל הקבצים כבר מעודכנים. אין קבצים חדשים לעיבוד.")
        return

    logging.info(f"⏳ נמצאו {len(files_to_process)} קבצים הדורשים עיבוד (חדשים או שנכשלו בעבר).")
    
    session, start_time, successful_files, failed_files = Session(), time.time(), [], []
    for i, file_path in enumerate(files_to_process):
        identifier = get_file_identifier(file_path)
        try:
            logging.info(f"--- [{i+1}/{len(files_to_process)}] מעבד את: {file_path.name} ---")
            
            logging.info("  - שלב 1/4: חילוץ טקסט...")
            doc = fitz.open(file_path)
            full_text = "".join(page.get_text() for page in doc)
            if len(full_text.strip()) < 200 * doc.page_count:
                logging.warning(f"  - כמות טקסט נמוכה. מנסה לבצע OCR...")
                full_text = ""
                for page_num, page in enumerate(doc):
                    logging.info(f"    - מבצע OCR על עמוד {page_num + 1}/{doc.page_count}...")
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    full_text += pytesseract.image_to_string(img, lang='heb+eng')
            doc.close()
            if not full_text.strip(): 
                raise ValueError("הקובץ ריק או אינו מכיל טקסט שניתן לחלץ.")
            
            lang = detect_language(full_text[:2000])
            logging.info(f"  - שלב 2/4: זוהתה שפה: {lang.upper()}")
            
            company_name = extract_company_name_multilingual(full_text[:5000], lang)
            logging.info(f"  - שלב 3/4: ניתוח תוכן מלא באמצעות AI (חברה: {company_name})...")
            
            all_findings_and_contacts = extract_findings_from_text(full_text)
            
            logging.info("    - תוצאות ניתוח AI מסכמות:")
            for category, findings_list in all_findings_and_contacts.items():
                if findings_list: 
                    logging.info(f"      - קטגוריה '{category}': {len(findings_list)} ממצאים")

            logging.info("  - שלב 4/4: שמירה בבסיס הנתונים...")
            existing_doc = session.query(Document).filter_by(filename=file_path.name).first()
            if existing_doc:
                session.query(Finding).filter_by(document_id=existing_doc.id).delete()
                session.query(AutoContact).filter_by(document_id=existing_doc.id).delete()
                doc_to_update = existing_doc
            else:
                doc_to_update = Document(filename=file_path.name)
                session.add(doc_to_update)

            doc_to_update.company_name = company_name
            doc_to_update.language = lang
            doc_to_update.full_text = full_text
            
            findings_data = {k: v for k, v in all_findings_and_contacts.items() if k != 'contacts_extracted'}
            contacts_data = all_findings_and_contacts.get('contacts_extracted', [])

            for category, findings_list in findings_data.items():
                if findings_list:
                    for finding_text in findings_list:
                        doc_to_update.findings.append(Finding(category=category, finding_text=finding_text))
            
            if contacts_data:
                for contact_info in contacts_data:
                    doc_to_update.auto_contacts.append(AutoContact(name=contact_info.get('name'), role=contact_info.get('role'), email=contact_info.get('email'), phone=contact_info.get('phone')))

            session.commit()
            
            manifest[identifier] = {"filename": file_path.name, "company_name": company_name, "status": "success", "error": None}
            save_manifest(manifest)
            logging.info(f"  - ✔️ הקובץ '{file_path.name}' עובד ונשמר בהצלחה")
            successful_files.append(file_path.name)
        except Exception as e:
            error_message = str(e).splitlines()[0]
            logging.error(f"❌ נכשל בעיבוד הקובץ '{file_path.name}'. סיבה: {error_message}", exc_info=False)
            logging.getLogger(file_path.name).error(f"Traceback מלא עבור {file_path.name}:", exc_info=True)
            failed_files.append((file_path.name, error_message))
            session.rollback()
            manifest[identifier] = {"filename": file_path.name, "status": "failed", "error": error_message}
            save_manifest(manifest)
    session.close()
    
    end_time = time.time()
    logging.info("\n" + "="*50 + "\n--- תהליך העיבוד הסתיים ---")
    logging.info(f"זמן כולל: {end_time - start_time:.2f} שניות.")
    logging.info(f"✔️ עובדו בהצלחה: {len(successful_files)} קבצים.")
    if not failed_files:
        if successful_files: 
            logging.info("✅ --- כל הקבצים החדשים/שנכשלו עובדו בהצלחה! --- ✅")
    else:
        logging.warning(f"❌ --- הסתיים עם {len(failed_files)} שגיאות --- ❌")
        logging.warning("הקבצים הבאים נכשלו ויטופלו בריצה הבאה:")
        for filename, error in failed_files:
            logging.warning(f"  - קובץ: {filename}\n    סיבה: {error}")
        logging.warning("לפרטים טכניים מלאים, יש לבדוק את הקובץ 'processing_errors.log'.")
    logging.info("="*50 + "\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("שימוש: python process_files.py \"<הנתיב המלא לתיקיית ה-PDF>\"")
        print("דוגמה: python process_files.py \"C:\\Users\\MyUser\\Documents\\Financial Reports\"")
    else:
        run_processing_pipeline(sys.argv[1])