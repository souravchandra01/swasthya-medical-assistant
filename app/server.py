from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from llama_cpp import Llama
import re

# CONFIG
GGUF_PATH    = "../models/swasthya-1b-q4km.gguf"
N_CTX        = 2048
N_THREADS    = 4
N_GPU_LAYERS = 0

# EMERGENCY KEYWORDS
EMERGENCY_EN = [
    "chest pain", "left arm pain", "heart attack", "cannot breathe",
    "difficulty breathing", "unconscious", "not breathing", "stroke",
    "sudden weakness", "severe bleeding", "coughing blood",
    "vomiting blood", "seizure", "collapsed", "fainted"
]

EMERGENCY_BN = [
    "বুকে ব্যথা", "বুকে তীব্র ব্যথা", "শ্বাস নিতে পারছি না",
    "শ্বাস নিতে কষ্ট", "হার্ট অ্যাটাক", "অজ্ঞান", "রক্ত বমি",
    "খিঁচুনি", "স্ট্রোক", "হঠাৎ দুর্বল", "তীব্র রক্তক্ষরণ",
    "বাম হাতে ব্যথা", "শ্বাসকষ্ট"
]

def check_emergency(text: str):
    is_bengali = bool(re.search(r'[\u0980-\u09FF]', text))
    
    if is_bengali:
        if any(kw in text for kw in EMERGENCY_BN):
            return (
                "🚨 জরুরি সতর্কতা: আপনার উপসর্গগুলো জীবনঘাতী হতে পারে। "
                "এখনই ১০৮ নম্বরে কল করুন অথবা নিকটস্থ জরুরি বিভাগে যান। "
                "দেরি করবেন না।\n\n---\n\n"
            )
    else:
        if any(kw in text.lower() for kw in EMERGENCY_EN):
            return (
                "🚨 EMERGENCY WARNING: Your symptoms may indicate a life-threatening condition. "
                "Call 108 (ambulance) or go to the nearest emergency room IMMEDIATELY.\n\n---\n\n"
            )
    return None

# LOAD MODEL
print(f"Loading model: {GGUF_PATH}")
llm = Llama(
    model_path=GGUF_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False
)
print("Model ready.")

# APP
app = FastAPI(title="Swasthya API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# SCHEMAS
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 1024
    model: Optional[str] = "local"
    stream: Optional[bool] = False

# PROMPT BUILDER
def build_prompt(messages: List[Message]) -> str:
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += "<start_of_turn>system\n" + msg.content + "<end_of_turn>\n"
        elif msg.role == "user":
            prompt += "<start_of_turn>user\n" + msg.content + "<end_of_turn>\n"
        elif msg.role == "assistant":
            prompt += "<start_of_turn>model\n" + msg.content + "<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return prompt

# ROUTES
@app.get("/")
def root():
    return {"status": "ok", "app": "Swasthya"}

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    user_msgs = [m for m in req.messages if m.role == "user"]
    last_input = user_msgs[-1].content if user_msgs else ""

    prompt = build_prompt(req.messages)
    output = llm(
        prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.1,
        stop=["<end_of_turn>", "<start_of_turn>"]
    )

    reply = output["choices"][0]["text"].strip()

    warning = check_emergency(last_input)
    if warning:
        reply = warning + reply

    return {
        "id": "swasthya",
        "object": "chat.completion",
        "model": "local",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop"
        }]
    }

# MAIN
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)