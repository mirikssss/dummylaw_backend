import os
import requests
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, List
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from models import User
from passlib.context import CryptContext
import PyPDF2
import docx
from io import BytesIO
import re

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Добавляем отдачу статики
app.mount("/", StaticFiles(directory="frontend_build", html=True), name="static")

Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

class DocumentResponse(BaseModel):
    explanation: str
    summary: str
    key_points: List[str]
    risks: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    risk: Optional[int] = None
    full_text: str

class RegisterRequest(BaseModel):
    full_name: str
    phone: str
    email: EmailStr
    password: str

def extract_text_from_file(content: bytes, filename: str) -> str:
    if filename.endswith('.txt'):
        return content.decode('utf-8', errors='ignore')
    elif filename.endswith('.docx'):
        return extract_text_from_docx(content)
    elif filename.endswith('.pdf'):
        return extract_text_from_pdf(content)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {filename}")

def extract_text_from_docx(docx_file: bytes) -> str:
    try:
        doc = docx.Document(BytesIO(docx_file))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        print(f"Ошибка при чтении DOCX: {str(e)}")
        raise ValueError("Ошибка при чтении DOCX файла")

def extract_text_from_pdf(pdf_file: bytes) -> str:
    try:
        pdf = PyPDF2.PdfReader(BytesIO(pdf_file))
        return "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception as e:
        print(f"Ошибка при чтении PDF: {str(e)}")
        raise ValueError("Ошибка при чтении PDF файла")

def gemini_explain(text: str) -> DocumentResponse:
    url = GEMINI_URL
    headers = {"Content-Type": "application/json"}
    prompt = (
        "You are a highly qualified legal expert specializing in contract analysis under the laws of Uzbekistan.\n"
        "Analyze the following legal document in detail, using the Civil Code and other relevant legislation of the Republic of Uzbekistan.\n"
        "Your goal is to protect the client's interests, identify all possible legal risks, unfair or ambiguous terms, and provide practical, actionable recommendations.\n"
        "Be specific: cite relevant articles of law or legal practice where possible.\n"
        "Structure your answer clearly and concisely.\n\n"
        "Please provide the following sections (use clear headings):\n"
        "1. Explanation: Explain the document in simple terms for a non-lawyer.\n"
        "2. Summary: Give a concise summary of the document's main points.\n"
        "3. Key Points: List the most important provisions and obligations.\n"
        "4. Risks: Identify all potential legal and practical risks for the client (with severity and references to law if possible).\n"
        "5. Recommendations: Give practical, actionable recommendations to the client (including what to negotiate, clarify, or refuse).\n"
        "6. Legal References: List relevant articles of the Civil Code or other laws that apply.\n\n"
        f"Document:\n{text}"
    )

    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    params = {"key": GEMINI_API_KEY}
    try:
        response = requests.post(url, headers=headers, params=params, json=data, timeout=90)
        response.raise_for_status()
        result = response.json()
        answer = result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print("Error calling Gemini API:", e)
        traceback.print_exc()
        raise

    # Простейший парсер секций (можно доработать под формат ответа Gemini)
    sections = re.split(r"\*\*\d+\.\s*([^\*]+):\*\*", answer)
    data_map = {
        "explanation": "",
        "summary": "",
        "key_points": [],
        "risks": [],
        "recommendations": []
    }
    for i in range(1, len(sections), 2):
        section_name = sections[i].strip().lower()
        section_text = sections[i+1].strip()
        if "explanation" in section_name or "explain" in section_name:
            data_map["explanation"] = section_text
        elif "summary" in section_name:
            data_map["summary"] = section_text
        elif "key point" in section_name:
            data_map["key_points"] = [line.strip("* ").strip() for line in section_text.splitlines() if line.strip().startswith("*")]
        elif "risk" in section_name:
            data_map["risks"] = [line.strip("* ").strip() for line in section_text.splitlines() if line.strip().startswith("*")]
        elif "recommendation" in section_name:
            data_map["recommendations"] = [line.strip("* ").strip() for line in section_text.splitlines() if line.strip().startswith("*")]

    # Оценка риска отдельным запросом
    risk_prompt = (
        "Assess the risk for the client in this legal document on a scale from 0 to 100, where 0 means no risk and 100 means maximum risk. Respond with a single number only.\n\n"
        f"Document:\n{text}"
    )
    risk_data = {
        "contents": [
            {"parts": [{"text": risk_prompt}]}
        ]
    }
    try:
        risk_response = requests.post(url, headers=headers, params=params, json=risk_data, timeout=30)
        risk_response.raise_for_status()
        risk_result = risk_response.json()
        risk_text = risk_result["candidates"][0]["content"]["parts"][0]["text"]
        try:
            data_map["risk"] = int(risk_text.strip())
        except ValueError:
            data_map["risk"] = 50
    except Exception as e:
        print("Error calling Gemini API for risk assessment:", e)
        data_map["risk"] = 50

    return DocumentResponse(
        explanation=data_map["explanation"],
        summary=data_map["summary"],
        key_points=data_map["key_points"],
        risks=data_map["risks"],
        recommendations=data_map["recommendations"],
        risk=data_map["risk"],
        full_text=text
    )

@app.post("/api/analyze-document", response_model=DocumentResponse)
async def analyze_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = extract_text_from_file(content, file.filename)
        return gemini_explain(text)
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chat")
async def chat_with_ai(question: str = Form(...), document: str = Form(...)):
    url = GEMINI_URL
    headers = {"Content-Type": "application/json"}
    prompt = (
        "You are a legal expert specializing in Uzbek law. Answer the following question about the legal document.\n"
        "Be specific and cite relevant laws when possible.\n\n"
        f"Document:\n{document}\n\n"
        f"Question: {question}"
    )
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    params = {"key": GEMINI_API_KEY}
    try:
        response = requests.post(url, headers=headers, params=params, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return {"answer": result["candidates"][0]["content"]["parts"][0]["text"]}
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/register")
async def register_user(data: RegisterRequest):
    db: Session = SessionLocal()
    try:
        existing_user = db.query(User).filter(
            (User.email == data.email) | (User.phone == data.phone)
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Пользователь с таким email или телефоном уже существует"
            )
        hashed_password = pwd_context.hash(data.password)
        new_user = User(
            full_name=data.full_name,
            phone=data.phone,
            email=data.email,
            hashed_password=hashed_password
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"message": "Пользователь успешно зарегистрирован"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/health")
async def health_check():
    return {"status": "ok"} 