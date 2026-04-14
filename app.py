from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from openai import OpenAI
import tempfile, os

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

messages = [
    {"role": "system", "content": "你是一个有趣的语音助手，回答简洁，100字以内。"}
]

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html") as f:
        return f.read()

@app.post("/chat")
async def chat(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(await audio.read())
        tmp_path = f.name
    with open(tmp_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=f, language="zh"
        )
    os.unlink(tmp_path)
    user_text = transcript.text
    messages.append({"role": "user", "content": user_text})
    response = client.chat.completions.create(
        model="gpt-4o", messages=messages
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    tts = client.audio.speech.create(
        model="tts-1", voice="nova", input=reply
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir="/tmp") as f:
        f.write(tts.content)
        audio_path = f.name
    return {
        "user_text": user_text,
        "reply": reply,
        "audio_url": f"/audio/{os.path.basename(audio_path)}"
    }

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    return FileResponse(f"/tmp/{filename}", media_type="audio/mpeg")