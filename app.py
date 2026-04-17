from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from openai import OpenAI
import tempfile, os, requests

app = FastAPI()

# OpenAI 客户端（用于 Whisper 语音转文字 + GPT 对话）
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ElevenLabs 配置
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")  # ElevenLabs API Key
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")  # Voice ID

messages = [
    {"role": "system", "content": "你是一个有趣的语音助手，回答简洁，100字以内。"}
]

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html") as f:
        return f.read()

@app.post("/chat")
async def chat(audio: UploadFile = File(...)):
    # 1. 保存上传的音频
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(await audio.read())
        tmp_path = f.name

    # 2. Whisper 语音转文字
    with open(tmp_path, "rb") as f:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1", file=f, language="zh"
        )
    os.unlink(tmp_path)

    # 3. GPT 生成回复
    user_text = transcript.text
    messages.append({"role": "user", "content": user_text})
    response = openai_client.chat.completions.create(
        model="gpt-4o", messages=messages
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    # 4. ElevenLabs TTS 生成语音（用你克隆的声音）
    tts_response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "text": reply,
            "model_id": "eleven_multilingual_v2",  # 支持中文
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
    )

    # 5. 保存音频文件
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir="/tmp") as f:
        f.write(tts_response.content)
        audio_path = f.name

    return {
        "user_text": user_text,
        "reply": reply,
        "audio_url": f"/audio/{os.path.basename(audio_path)}"
    }

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    return FileResponse(f"/tmp/{filename}", media_type="audio/mpeg")
