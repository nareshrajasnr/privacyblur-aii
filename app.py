"""
PrivacyBlur AI - Local Version
Run this on your own computer for best performance (30-80ms latency)

Setup:
1. Install: pip install flask ultralytics opencv-python numpy torch
2. Place model files in 'models/' folder
3. Run: python app.py
4. Open: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import base64
import os

app = Flask(__name__)

# Check if models exist
MODEL_PATH_FACE = 'models/yolov8n-face-lindevs.pt'
MODEL_PATH_ID = 'models/best.pt'

if not os.path.exists(MODEL_PATH_FACE):
    print(f"❌ ERROR: Face model not found at {MODEL_PATH_FACE}")
    print("   Please place 'yolov8n-face-lindevs.pt' in the 'models/' folder")
    exit(1)

if not os.path.exists(MODEL_PATH_ID):
    print(f"❌ ERROR: ID card model not found at {MODEL_PATH_ID}")
    print("   Please place 'best.pt' in the 'models/' folder")
    exit(1)

# Load models
print("Loading models...")
model_face = YOLO(MODEL_PATH_FACE)
model_idcard = YOLO(MODEL_PATH_ID)
print("✅ Models loaded successfully!")

# Same settings that work well
FACE_CONFIDENCE = 0.45
ID_CONFIDENCE = 0.35
BLUR_STRENGTH = 15
PROCESS_SIZE = 320

# Exact same HTML (works perfectly)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PrivacyBlur AI - Local</title>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        :root { --bg:#0a0a0f;--surface:#12121a;--accent:#00ff88;--danger:#ff4455;--text:#e8e8f0;--muted:#6b6b80;--border:rgba(255,255,255,0.07); }
        *{margin:0;padding:0;box-sizing:border-box;}
        body{font-family:'DM Mono',monospace;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;}
        body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,255,136,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,136,0.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0;}
        .app{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:40px 24px;}
        header{display:flex;align-items:center;gap:16px;margin-bottom:40px;flex-wrap:wrap;}
        .logo-mark{width:48px;height:48px;border-radius:12px;background:var(--accent);display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 0 30px rgba(0,255,136,0.4);}
        h1{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;}
        h1 span{color:var(--accent);}
        .tagline{font-size:11px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-top:2px;}
        .status-bar{display:flex;align-items:center;gap:8px;padding:10px 18px;background:var(--surface);border:1px solid var(--border);border-radius:100px;font-size:12px;color:var(--muted);margin-left:auto;}
        .status-dot{width:8px;height:8px;border-radius:50%;background:var(--muted);transition:all 0.3s;}
        .status-dot.live{background:var(--accent);box-shadow:0 0 8px var(--accent);animation:pulse 2s infinite;}
        @keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.4;}}
        .controls{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:20px 24px;margin-bottom:20px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;}
        .btn{font-family:'Syne',sans-serif;font-weight:700;font-size:13px;padding:11px 26px;border-radius:10px;border:none;cursor:pointer;transition:all 0.2s;display:flex;align-items:center;gap:8px;}
        .btn-primary{background:var(--accent);color:#000;}
        .btn-primary:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,255,136,0.35);}
        .btn-danger{background:rgba(255,68,85,0.12);color:var(--danger);border:1px solid rgba(255,68,85,0.25);}
        .btn-danger:hover:not(:disabled){background:var(--danger);color:white;}
        .btn:disabled{opacity:0.35;cursor:not-allowed;}
        .fps-info{margin-left:auto;font-size:12px;color:var(--muted);display:flex;gap:24px;}
        .fps-val{color:var(--accent);}
        .main-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}
        @media(max-width:700px){.main-grid{grid-template-columns:1fr;}}
        .panel{background:var(--surface);border:1px solid var(--border);border-radius:16px;overflow:hidden;}
        .panel-header{display:flex;align-items:center;justify-content:space-between;padding:13px 18px;border-bottom:1px solid var(--border);}
        .panel-title{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);}
        .badge{font-size:10px;padding:3px 10px;border-radius:100px;font-weight:500;letter-spacing:1px;text-transform:uppercase;}
        .badge-raw{background:rgba(107,107,128,0.15);color:var(--muted);border:1px solid var(--border);}
        .badge-ai{background:rgba(0,255,136,0.1);color:var(--accent);border:1px solid rgba(0,255,136,0.25);}
        .video-wrap{position:relative;aspect-ratio:4/3;background:#000;}
        video,.processed-img{width:100%;height:100%;object-fit:cover;display:block;}
        .placeholder{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;color:var(--muted);}
        .placeholder-icon{font-size:40px;opacity:0.3;}
        .placeholder-text{font-size:13px;}
        .placeholder-sub{font-size:11px;color:#333;margin-top:2px;}
        .scan-line{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:0;pointer-events:none;transition:opacity 0.3s;}
        .scan-line.active{opacity:1;animation:scan 2s linear infinite;}
        @keyframes scan{0%{top:0%;}100%{top:100%;}}
        .legend{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;}
        @media(max-width:600px){.legend{grid-template-columns:1fr;}}
        .legend-item{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px;display:flex;align-items:center;gap:12px;}
        .legend-dot{width:12px;height:12px;border-radius:3px;flex-shrink:0;}
        .legend-label{font-family:'Syne',sans-serif;font-weight:700;font-size:12px;margin-bottom:2px;}
        .legend-desc{color:var(--muted);font-size:11px;}
        .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}
        @media(max-width:600px){.stats{grid-template-columns:repeat(2,1fr);}}
        .stat-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;}
        .stat-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;}
        .stat-value{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:var(--accent);}
        .stat-unit{font-size:11px;color:var(--muted);margin-left:3px;}
        .toast{position:fixed;bottom:24px;right:24px;background:var(--danger);color:white;padding:14px 20px;border-radius:10px;font-size:13px;transform:translateY(80px);opacity:0;transition:all 0.3s;z-index:999;max-width:320px;}
        .toast.show{transform:translateY(0);opacity:1;}
        canvas{display:none;}
    </style>
</head>
<body>
<div class="app">
    <header>
        <div class="logo-mark">🔒</div>
        <div>
            <h1>Privacy<span>Blur</span> AI</h1>
            <div class="tagline">Real-time face &amp; ID card protection</div>
        </div>
        <div class="status-bar">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">Idle</span>
        </div>
    </header>
    <div class="controls">
        <button class="btn btn-primary" id="startBtn" onclick="startCamera()">▶ Start Camera</button>
        <button class="btn btn-danger" id="stopBtn" onclick="stopCamera()" disabled>■ Stop</button>
        <div class="fps-info">
            <span>Frames: <span class="fps-val" id="frameCount">0</span></span>
            <span>Latency: <span class="fps-val" id="latencyDisplay">—</span> ms</span>
        </div>
    </div>
    <div class="main-grid">
        <div class="panel">
            <div class="panel-header"><span class="panel-title">Raw Feed</span><span class="badge badge-raw">Input</span></div>
            <div class="video-wrap">
                <video id="localVideo" autoplay muted playsinline style="display:none;"></video>
                <div class="placeholder" id="rawHolder">
                    <div class="placeholder-icon">📷</div>
                    <div class="placeholder-text">Camera not started</div>
                    <div class="placeholder-sub">Click Start Camera above</div>
                </div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-header"><span class="panel-title">AI Processed</span><span class="badge badge-ai">Output</span></div>
            <div class="video-wrap">
                <img class="processed-img" id="processedImg" style="display:none;" alt="AI Output">
                <div class="scan-line" id="scanLine"></div>
                <div class="placeholder" id="aiHolder">
                    <div class="placeholder-icon">🤖</div>
                    <div class="placeholder-text">Waiting for stream</div>
                    <div class="placeholder-sub">AI output appears here</div>
                </div>
            </div>
        </div>
    </div>
    <div class="legend">
        <div class="legend-item">
            <div class="legend-dot" style="background:#00ff88;box-shadow:0 0 8px #00ff88;"></div>
            <div><div class="legend-label">Speaker</div><div class="legend-desc">Green box — largest face, no blur</div></div>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#ff4455;box-shadow:0 0 8px #ff4455;"></div>
            <div><div class="legend-label">Background Face</div><div class="legend-desc">Red box + blurred</div></div>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#4488ff;box-shadow:0 0 8px #4488ff;"></div>
            <div><div class="legend-label">ID Card</div><div class="legend-desc">Blue box + blurred</div></div>
        </div>
    </div>
    <div class="stats">
        <div class="stat-card"><div class="stat-label">Faces Detected</div><div class="stat-value" id="statFaces">0</div></div>
        <div class="stat-card"><div class="stat-label">IDs Blurred</div><div class="stat-value" id="statIds">0</div></div>
        <div class="stat-card"><div class="stat-label">Total Frames</div><div class="stat-value" id="statFrames">0</div></div>
        <div class="stat-card"><div class="stat-label">Avg Latency</div><div class="stat-value" id="statAvgLat">—<span class="stat-unit">ms</span></div></div>
    </div>
</div>
<canvas id="captureCanvas"></canvas>
<div class="toast" id="toast"></div>
<script>
    let stream=null,running=false,frameCount=0,totalLatency=0,processing=false;
    const video=document.getElementById('localVideo');
    const canvas=document.getElementById('captureCanvas');
    const ctx=canvas.getContext('2d');
    const outImg=document.getElementById('processedImg');

    function toast(msg){const el=document.getElementById('toast');el.textContent=msg;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),4000);}
    function setStatus(cls,text){document.getElementById('statusDot').className='status-dot '+cls;document.getElementById('statusText').textContent=text;}

    async function startCamera(){
        try{
            stream=await navigator.mediaDevices.getUserMedia({video:{width:320,height:240},audio:false});
            video.srcObject=stream;video.style.display='block';
            document.getElementById('rawHolder').style.display='none';
            document.getElementById('startBtn').disabled=true;
            document.getElementById('stopBtn').disabled=false;
            document.getElementById('scanLine').classList.add('active');
            setStatus('live','Live');running=true;loop();
        }catch(e){toast('Camera access denied!');setStatus('','Error');}
    }

    function stopCamera(){
        running=false;processing=false;
        if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
        video.style.display='none';outImg.style.display='none';
        document.getElementById('rawHolder').style.display='flex';
        document.getElementById('aiHolder').style.display='flex';
        document.getElementById('startBtn').disabled=false;
        document.getElementById('stopBtn').disabled=true;
        document.getElementById('scanLine').classList.remove('active');
        setStatus('','Idle');
    }

    async function loop(){
        while(running){
            if(processing||video.readyState<2){await sleep(30);continue;}
            canvas.width=320;canvas.height=240;
            ctx.drawImage(video,0,0,320,240);
            const frameData=canvas.toDataURL('image/jpeg',0.5);
            const t0=performance.now();
            processing=true;
            try{
                const res=await fetch('/process_frame',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frame:frameData})});
                const data=await res.json();
                const ms=Math.round(performance.now()-t0);
                if(data.status==='ok'){
                    outImg.src='data:image/jpeg;base64,'+data.frame;
                    outImg.style.display='block';
                    document.getElementById('aiHolder').style.display='none';
                    frameCount++;totalLatency+=ms;
                    document.getElementById('frameCount').textContent=frameCount;
                    document.getElementById('latencyDisplay').textContent=ms;
                    document.getElementById('statFaces').textContent=data.faces??0;
                    document.getElementById('statIds').textContent=data.ids??0;
                    document.getElementById('statFrames').textContent=frameCount;
                    document.getElementById('statAvgLat').innerHTML=Math.round(totalLatency/frameCount)+'<span class="stat-unit">ms</span>';
                }
            }catch(e){}
            processing=false;
        }
    }
    const sleep=ms=>new Promise(r=>setTimeout(r,ms));
</script>
</body>
</html>
"""


def draw_box(img, x1, y1, x2, y2, color, label):
    """Draw bounding box with label - same as working version"""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.5, 2
    (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
    ly = max(y1 - 4, th + 8)
    cv2.rectangle(img, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), color, -1)
    cv2.putText(img, label, (x1 + 4, ly - 2), font, fs, (0, 0, 0), ft, cv2.LINE_AA)


def run_detection(frame):
    """Same detection logic that works perfectly"""
    output = frame.copy()
    face_count = 0
    id_count = 0

    GREEN = (0, 255, 136)
    RED = (50, 50, 255)
    BLUE = (255, 100, 0)

    # Face detection
    results_face = model_face.predict(
        source=frame, conf=FACE_CONFIDENCE,
        verbose=False, imgsz=PROCESS_SIZE, device=0)

    face_boxes = []
    for r in results_face:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_boxes.append((x1, y1, x2, y2))

    largest = None
    if face_boxes:
        largest = max(face_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))

    for (x1, y1, x2, y2) in face_boxes:
        face_count += 1
        if (x1, y1, x2, y2) == largest:
            draw_box(output, x1, y1, x2, y2, GREEN, "Speaker")
        else:
            roi = output[y1:y2, x1:x2]
            if roi.size > 0:
                output[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (BLUR_STRENGTH, BLUR_STRENGTH), 15)
            draw_box(output, x1, y1, x2, y2, RED, "Face [blurred]")

    # ID card detection
    results_id = model_idcard.predict(
        source=frame, conf=ID_CONFIDENCE,
        verbose=False, imgsz=PROCESS_SIZE, device=0)

    for r in results_id:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w < 30 or h < 20:
                continue
            id_count += 1
            roi = output[y1:y2, x1:x2]
            if roi.size > 0:
                output[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 30)
            draw_box(output, x1, y1, x2, y2, BLUE, "ID Card [blurred]")

    return output, face_count, id_count


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    try:
        data = request.json
        frame_data = data.get('frame', '')
        _, encoded = frame_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'status': 'error'})
        output, faces, ids = run_detection(frame)
        _, buf = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64 = base64.b64encode(buf).decode('utf-8')
        return jsonify({'status': 'ok', 'frame': b64, 'faces': faces, 'ids': ids})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🔒 PrivacyBlur AI - Local Server Starting...")
    print("=" * 60)
    print("➡️  Open your browser and go to: http://localhost:5000")
    print("⚡ Expected latency: 30-80ms (depending on your GPU)")
    print("🛑 Press CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    # Run Flask on localhost
    app.run(host='127.0.0.1', port=5000, debug=False)
