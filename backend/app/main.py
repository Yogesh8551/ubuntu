from fastapi.responses import StreamingResponse
import io
from fastapi import HTTPException
import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session

from .database import Base, engine, SessionLocal
from . import utils, crud, schemas

from fastapi.middleware.cors import CORSMiddleware

import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


logging.getLogger("uvicorn").setLevel(logging.DEBUG)
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)   # shows DB activity
logging.getLogger("CRUD_LOGS").setLevel(logging.DEBUG)          # our CRUD logs



# --------------------------------------
# LOGGING SETUP
# --------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("🚀 Backend starting...")


# --------------------------------------
# DB INIT
# --------------------------------------
Base.metadata.create_all(bind=engine)

def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


# --------------------------------------
# FASTAPI APP
# --------------------------------------
app = FastAPI()


origins = [
    "https://test.phylon.in"   #frontend url
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],    # allow GET, POST, etc.
    allow_headers=["*"],    # allow all headers
)

logger.info("✅ CORS configured successfully")


# --------------------------------------
# CONNECT CHROMA
# --------------------------------------
logger.info("🔗 Connecting to ChromaDB at /chroma_data ...")
client = PersistentClient(path="/chroma_data")

COLLECTION = "resumes_collection"
try:
    col = client.get_collection(COLLECTION)
    logger.info(f"📁 Using existing Chroma collection: {COLLECTION}")
except:
    col = client.create_collection(COLLECTION)
    logger.info(f"📁 Created new Chroma collection: {COLLECTION}")


# Load embedding model
logger.info("🔤 Loading SentenceTransformer model (MiniLM)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logger.info("✅ Model loaded successfully")


# --------------------------------------
# ROUTES
# --------------------------------------

@app.get("/")
def read_root():
    logger.info("➡️ Root endpoint accessed")
    return {"status": "Backend running successfully"}

@app.post("/ingest")
async def ingest_resume(
    file: UploadFile = File(...),
    name: str = Form(None),
    resumetype: str = Form(None),
    occupation: str = Form(None),
    db: Session = Depends(db)
):

    logger.info(f"📥 Received file: {file.filename}")
    logger.info(f"➡️ Metadata | name={name}, resumetype={resumetype}, occupation={occupation}")

    data = await file.read()
    logger.info(f"📄 File size received: {len(data)} bytes")

    # Generate Chroma ID immediately
    chroma_id = str(uuid.uuid4())
    logger.info(f"🆔 Generated Chroma ID: {chroma_id}")

    # Extract text
    try:
        text = utils.extract_text(file.filename, data)
        logger.info(f"📝 Extracted text length: {len(text)} characters")
    except Exception as e:
        logger.error(f"❌ Error extracting text: {e}")
        raise

    # Remove NUL bytes before saving
    cleaned_text = text.replace("\x00", "")
    snippet = cleaned_text[:300]

    # Embedding
    try:
        logger.info("🔢 Generating embedding...")
        embedding = model.encode(cleaned_text).tolist()
        logger.info(f"🔢 Embedding generated | vector size = {len(embedding)}")
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise

    # Chroma DB insert
    try:
        col.add(
            ids=[chroma_id],
            documents=[cleaned_text],
            embeddings=[embedding],
            metadatas=[{
                "name": name,
                "resumetype": resumetype,
                "occupation": occupation,
                "filename": file.filename
            }]
        )
        logger.info(f"📚 Added document to Chroma collection: {chroma_id}")
    except Exception as e:
        logger.error(f"❌ Failed to insert into ChromaDB: {e}")
        raise

    # Insert metadata in PostgreSQL
    try:
        logger.info("🗄 Saving metadata in PostgreSQL...")
        obj = crud.create_resume(
            db,
            name=name,
            resumetype=resumetype,
            occupation=occupation,
            filename=file.filename,
            chroma_id=chroma_id,
            snippet=snippet
        )
        logger.info(f"✅ Metadata stored in PostgreSQL | resume_id = {obj.id}")
    except Exception as e:
        logger.error(f"❌ Failed to insert into PostgreSQL: {e}")
        raise

    return obj

from fastapi import Form

@app.post("/search", response_model=list[schemas.ResumeMeta])
def search(
    name: str = Form(None),
    resumetype: str = Form(None),
    occupation: str = Form(None),
    db: Session = Depends(db)
):

        # ⏳ Start timer
    start_time = time.time()

    results = crud.query_resumes(db, name, resumetype, occupation)

    # ⏱ Calculate time taken in ms
    time_taken = (time.time() - start_time) * 1000  

    # 📝 Log output cleanly
    logger.info(f"🔍 Search Query -> name={name}, type={resumetype}, occupation={occupation}")
    logger.info(f"⚡ Query returned {len(results)} result(s) in {time_taken:.2f} ms")


    results = crud.query_resumes(db, name, resumetype, occupation)
    return results

# @app.get("/document/{chroma_id}")
# def document(chroma_id: str):
#     res = col.get(ids=[chroma_id], include=["documents", "metadatas"])

#     if not res or not res["documents"] or not res["documents"][0]:
#         raise HTTPException(status_code=404, detail="Document not found")

#     return {
#         "chroma_id": chroma_id,
#         "metadata": res["metadatas"][0],
#         "document": res["documents"][0]
#     }

import re
from fastapi import HTTPException

def extract_section(text, title):
    pattern = rf"{title}[\s\S]*?(?=\n[A-Z][A-Z ]+:|\Z)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0).replace(title, "").strip() if match else None

@app.get("/document/{chroma_id}")
def document(chroma_id: str):
    res = col.get(ids=[chroma_id], include=["documents", "metadatas"])

    if not res or not res["documents"]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    full_text = res["documents"][0]

    # ----------- SECTION EXTRACTION ----------- #
    contacts = extract_section(full_text, "CONTACT INFORMATION")
    summary = extract_section(full_text, "SUMMARY")
    skills = extract_section(full_text, "SKILLS AND TOOLS")
    experience = extract_section(full_text, "PROFESSIONAL EXPERIENCE")
    certifications = extract_section(full_text, "Certifications and Badges")

    # Bullet points extract clean array
    contact_list = re.findall(r"[•\-]\s*(.*)", contacts or "")
    skill_list = re.findall(r"•\s*(.*)", skills or "")
    exp_points = re.findall(r"•\s*(.*)", experience or "")
    cert_list = re.findall(r"•\s*(.*)", certifications or "")

    final_output = {
        "1. Basic Details": {
            "Name": res["metadatas"][0].get("name"),
            "Resume Type": res["metadatas"][0].get("resumetype"),
            "Occupation": res["metadatas"][0].get("occupation"),
            "Filename": res["metadatas"][0].get("filename")
        },
        "2. Summary": summary,
        "3. Technical Skills": skill_list,
        "4. Professional Experience": exp_points,
        "5. Certifications": cert_list,
        "6. Contact Information": contact_list
    }

    return final_output
