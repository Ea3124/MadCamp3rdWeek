# main_server.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI()

# CORS 설정
origins = [
    "*",  # 프론트엔드 원본    
    # 필요시 추가
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
app.mount("/assets", StaticFiles(directory="public/assets"), name="static")

# Svelte의 index.html 반환
@app.get("/", response_class=FileResponse)
async def main():
    return "public/index.html"

# 예시 API 엔드포인트
@app.get("/api/hello")
async def read_root():
    return {"message": "Hello from FastAPI"}

# 이미지 생성 요청 스키마 정의
class GenerateRequest(BaseModel):
    prompt: str
    num_images: int = 1  # 기본값은 1장

# gpu_server를 통해 이미지 생성하는 엔드포인트
@app.post("/api/generate")
async def generate_image(request: GenerateRequest):
    gpu_server_url = "http://127.0.0.1:8001/generate"  # 실제 gpu_server URL과 포트로 업데이트

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(gpu_server_url, json=request.dict())
            response.raise_for_status()
            data = response.json()
            return JSONResponse(content=data)
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"GPU 서버 요청 실패: {e}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
