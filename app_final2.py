"""
RTIOC - Real-Time Identity and Object Concealment
Professional version with clean UI
Detection logic remains UNCHANGED
"""

from flask import Flask, render_template_string, request, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import base64
import os
import torch

app = Flask(__name__)

DEVICE = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {'GPU' if DEVICE == 0 else 'CPU'}")

# Model paths
MODEL_PATH_FACE = 'models/yolov8n-face-lindevs.pt'
MODEL_PATH_ID   = 'models/best.pt'
MODEL_PATH_ID2  = 'models/best2.pt'   # ← new trained model

if not os.path.exists(MODEL_PATH_FACE):
    print(f"❌ ERROR: Face model not found at {MODEL_PATH_FACE}")
    exit(1)

if not os.path.exists(MODEL_PATH_ID):
    print(f"❌ ERROR: ID card model not found at {MODEL_PATH_ID}")
    exit(1)

# Load models
print("Loading models...")
model_face    = YOLO(MODEL_PATH_FACE)
model_idcard  = YOLO(MODEL_PATH_ID)
model_idcard2 = None
if os.path.exists(MODEL_PATH_ID2):
    model_idcard2 = YOLO(MODEL_PATH_ID2)
    print("✅ Models loaded (face + 2x ID card models)!")
else:
    print("✅ Models loaded (face + 1x ID card model)")

# Detection parameters
FACE_CONFIDENCE = 0.45
ID_CONFIDENCE   = 0.50   # Balanced — catches distant cards too
BLUR_STRENGTH   = 15
PROCESS_SIZE    = 320

# Upscale factor for distant card detection
# Frame is enlarged before being sent to model — makes small/far cards bigger
ID_UPSCALE      = 2.0    # 2x upscale — increase to 3.0 if still missing far cards

# Temporal smoothing
ID_CONFIRM_FRAMES = 4    # frames needed to confirm (higher = less flicker)
ID_FORGET_FRAMES  = 8    # frames to keep blur after card disappears
ID_SMOOTH_ALPHA   = 0.6  # box position smoothing 0=very smooth/slow, 1=instant/jumpy

# Smoothing state
id_tracker = {
    'boxes': [],        # confirmed box positions (smoothed)
    'candidates': [],
    'hit_counts': [],
    'miss_counts': [],
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RTIOC - Real-Time Identity and Object Concealment</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #1e3a5f 0%, #2d5a7b 50%, #1e3a5f 100%);
            min-height: 100vh;
            color: #ffffff;
            position: relative;
            overflow-x: hidden;
        }
        
        /* Subtle dot pattern like Infracorp */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            right: 0;
            width: 300px;
            height: 300px;
            background-image: 
                radial-gradient(circle, rgba(255,255,255,0.15) 1px, transparent 1px);
            background-size: 20px 20px;
            pointer-events: none;
            z-index: 0;
        }
        
        body::after {
            content: '';
            position: fixed;
            bottom: 0;
            left: 0;
            width: 200px;
            height: 200px;
            background-image: 
                radial-gradient(circle, rgba(255,255,255,0.15) 1px, transparent 1px);
            background-size: 20px 20px;
            pointer-events: none;
            z-index: 0;
        }
        
        .app {
            position: relative;
            z-index: 1;
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 24px;
        }
        
        /* Header */
        header {
            text-align: center;
            margin-bottom: 50px;
            padding: 30px 0;
        }
        
        .brand {
            font-size: 42px;
            font-weight: 300;
            letter-spacing: 8px;
            margin-bottom: 16px;
            text-transform: uppercase;
            background: linear-gradient(90deg, #ffffff, #a8d5ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .tagline {
            font-size: 16px;
            color: rgba(255,255,255,0.7);
            font-weight: 300;
            letter-spacing: 2px;
        }
        
        .status-bar {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 8px 20px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 100px;
            font-size: 13px;
            margin-top: 20px;
            backdrop-filter: blur(10px);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: rgba(255,255,255,0.5);
            transition: all 0.3s;
        }
        
        .status-dot.live {
            background: #10b981;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }
        
        /* Controls */
        .controls {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }
        
        .btn {
            font-weight: 600;
            font-size: 14px;
            padding: 14px 32px;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            transition: all 0.3s;
            font-family: inherit;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: white;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        }
        
        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
        }
        
        .btn-danger {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            border: 2px solid #ef4444;
        }
        
        .btn-danger:hover:not(:disabled) {
            background: #ef4444;
            color: white;
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .fps-info {
            margin-left: auto;
            font-size: 13px;
            color: rgba(255,255,255,0.8);
            display: flex;
            gap: 24px;
        }
        
        .fps-val {
            color: #60a5fa;
            font-weight: 700;
        }
        
        /* Video Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }
        
        @media(max-width: 900px) {
            .main-grid { grid-template-columns: 1fr; }
        }
        
        .panel {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 20px;
            overflow: hidden;
        }
        
        .panel-header {
            padding: 16px 24px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .panel-title {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: rgba(255,255,255,0.8);
        }
        
        .badge {
            font-size: 10px;
            padding: 4px 12px;
            border-radius: 100px;
            font-weight: 600;
            background: rgba(255,255,255,0.15);
            color: rgba(255,255,255,0.9);
        }
        
        .video-wrap {
            position: relative;
            aspect-ratio: 4/3;
            background: #000;
        }
        
        video, .processed-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        
        .placeholder {
            position: absolute;
            inset: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 12px;
            background: rgba(0,0,0,0.6);
            color: rgba(255,255,255,0.7);
        }
        
        .placeholder-icon {
            font-size: 48px;
            opacity: 0.5;
        }
        
        .placeholder-text {
            font-size: 15px;
            font-weight: 500;
        }
        
        /* Legend */
        .legend {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        
        @media(max-width: 768px) {
            .legend { grid-template-columns: 1fr; }
        }
        
        .legend-item {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 16px;
            padding: 20px;
            display: flex;
            align-items: center;
            gap: 14px;
        }
        
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 3px;
            flex-shrink: 0;
        }
        
        .legend-label {
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 4px;
        }
        
        .legend-desc {
            color: rgba(255,255,255,0.6);
            font-size: 12px;
        }
        
        /* Stats */
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
        }
        
        @media(max-width: 768px) {
            .stats { grid-template-columns: repeat(2, 1fr); }
        }
        
        .stat-card {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
        }
        
        .stat-label {
            font-size: 11px;
            color: rgba(255,255,255,0.6);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        
        .stat-value {
            font-size: 36px;
            font-weight: 700;
            color: #60a5fa;
            line-height: 1;
        }
        
        .stat-unit {
            font-size: 14px;
            color: rgba(255,255,255,0.5);
            font-weight: 500;
        }
        
        /* Bottom dots indicator */
        .dots-indicator {
            position: fixed;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 8px;
            z-index: 100;
        }
        
        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: rgba(239, 68, 68, 0.6);
            transition: all 0.3s;
        }
        
        .dot.active {
            background: #ef4444;
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.8);
        }
        
        canvas { display: none; }
    </style>
</head>
<body>
<div class="app">
    <header>
        <div class="brand">RTIOC</div>
        <div class="tagline">Real-Time Identity and Object Concealment</div>
        <div class="status-bar">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">System Idle</span>
        </div>
    </header>
    
    <div class="controls">
        <button class="btn btn-primary" id="startBtn" onclick="startCamera()">▶ Start System</button>
        <button class="btn btn-danger" id="stopBtn" onclick="stopCamera()" disabled>■ Stop</button>
        <div class="fps-info">
            <span>Frames: <span class="fps-val" id="frameCount">0</span></span>
            <span>Latency: <span class="fps-val" id="latencyDisplay">—</span> ms</span>
        </div>
    </div>
    
    <div class="main-grid">
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">Input Stream</span>
                <span class="badge">RAW</span>
            </div>
            <div class="video-wrap">
                <video id="localVideo" autoplay muted playsinline style="display:none;"></video>
                <div class="placeholder" id="rawHolder">
                    <div class="placeholder-icon">📷</div>
                    <div class="placeholder-text">Camera Inactive</div>
                </div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">Processed Output</span>
                <span class="badge">AI</span>
            </div>
            <div class="video-wrap">
                <img class="processed-img" id="processedImg" style="display:none;">
                <div class="placeholder" id="aiHolder">
                    <div class="placeholder-icon">🤖</div>
                    <div class="placeholder-text">Awaiting Input</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="legend">
        <div class="legend-item">
            <div class="legend-dot" style="background:#10b981;"></div>
            <div>
                <div class="legend-label">Primary Subject</div>
                <div class="legend-desc">Unobscured</div>
            </div>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#ef4444;"></div>
            <div>
                <div class="legend-label">Background Identity</div>
                <div class="legend-desc">Concealed</div>
            </div>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#3b82f6;"></div>
            <div>
                <div class="legend-label">Sensitive Document</div>
                <div class="legend-desc">Protected</div>
            </div>
        </div>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-label">Identities</div>
            <div class="stat-value" id="statFaces">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Documents</div>
            <div class="stat-value" id="statIds">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Processed</div>
            <div class="stat-value" id="statFrames">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Response</div>
            <div class="stat-value" id="statAvgLat">—<span class="stat-unit">ms</span></div>
        </div>
    </div>
</div>

<!-- Three red dots at bottom -->
<div class="dots-indicator">
    <div class="dot" id="dot1"></div>
    <div class="dot" id="dot2"></div>
    <div class="dot" id="dot3"></div>
</div>

<canvas id="captureCanvas"></canvas>

<script>
    let stream=null,running=false,frameCount=0,totalLatency=0,processing=false;
    const video=document.getElementById('localVideo');
    const canvas=document.getElementById('captureCanvas');
    const ctx=canvas.getContext('2d');
    const outImg=document.getElementById('processedImg');

    function setStatus(cls,text){
        document.getElementById('statusDot').className='status-dot '+cls;
        document.getElementById('statusText').textContent=text;
    }
    
    // Animate dots
    function animateDots(){
        const dots=['dot1','dot2','dot3'];
        let idx=0;
        setInterval(()=>{
            dots.forEach(d=>document.getElementById(d).classList.remove('active'));
            document.getElementById(dots[idx]).classList.add('active');
            idx=(idx+1)%3;
        },800);
    }
    animateDots();

    async function startCamera(){
        try{
            stream=await navigator.mediaDevices.getUserMedia({video:{width:320,height:240},audio:false});
            video.srcObject=stream;
            video.style.display='block';
            document.getElementById('rawHolder').style.display='none';
            document.getElementById('startBtn').disabled=true;
            document.getElementById('stopBtn').disabled=false;
            setStatus('live','System Active');
            running=true;
            loop();
        }catch(e){
            setStatus('','Camera Error');
        }
    }

    function stopCamera(){
        running=false;
        processing=false;
        if(stream){
            stream.getTracks().forEach(t=>t.stop());
            stream=null;
        }
        video.style.display='none';
        outImg.style.display='none';
        document.getElementById('rawHolder').style.display='flex';
        document.getElementById('aiHolder').style.display='flex';
        document.getElementById('startBtn').disabled=false;
        document.getElementById('stopBtn').disabled=true;
        setStatus('','System Idle');
    }

    async function loop(){
        while(running){
            if(processing||video.readyState<2){
                await new Promise(r=>setTimeout(r,30));
                continue;
            }
            canvas.width=320;
            canvas.height=240;
            ctx.drawImage(video,0,0,320,240);
            const frameData=canvas.toDataURL('image/jpeg',0.5);
            const t0=performance.now();
            processing=true;
            try{
                const res=await fetch('/process_frame',{
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({frame:frameData})
                });
                const data=await res.json();
                const ms=Math.round(performance.now()-t0);
                if(data.status==='ok'){
                    outImg.src='data:image/jpeg;base64,'+data.frame;
                    outImg.style.display='block';
                    document.getElementById('aiHolder').style.display='none';
                    frameCount++;
                    totalLatency+=ms;
                    document.getElementById('frameCount').textContent=frameCount;
                    document.getElementById('latencyDisplay').textContent=ms;
                    document.getElementById('statFaces').textContent=data.faces||0;
                    document.getElementById('statIds').textContent=data.ids||0;
                    document.getElementById('statFrames').textContent=frameCount;
                    document.getElementById('statAvgLat').innerHTML=Math.round(totalLatency/frameCount)+'<span class="stat-unit">ms</span>';
                }
            }catch(e){}
            processing=false;
        }
    }
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════
# DETECTION LOGIC - COMPLETELY UNCHANGED (WORKING PERFECTLY!)
# ═══════════════════════════════════════════════════════════════

def draw_box(img, x1, y1, x2, y2, color, label):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.5, 2
    (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
    ly = max(y1 - 4, th + 8)
    cv2.rectangle(img, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), color, -1)
    cv2.putText(img, label, (x1 + 4, ly - 2), font, fs, (0, 0, 0), ft, cv2.LINE_AA)


def box_iou(a, b):
    """Intersection over Union — measures how much two boxes overlap"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def update_id_tracker(raw_boxes):
    t = id_tracker
    IOU_THRESH = 0.3

    def smooth_box(old, new):
        """Lerp box position — reduces jumping/flickering of the drawn box"""
        a = ID_SMOOTH_ALPHA
        return (
            int(old[0] * (1-a) + new[0] * a),
            int(old[1] * (1-a) + new[1] * a),
            int(old[2] * (1-a) + new[2] * a),
            int(old[3] * (1-a) + new[3] * a),
        )

    matched_candidates = set()
    matched_raw = set()
    for i, cand in enumerate(t['candidates']):
        for j, raw in enumerate(raw_boxes):
            if j in matched_raw:
                continue
            if box_iou(cand, raw) > IOU_THRESH:
                t['candidates'][i] = smooth_box(cand, raw)
                t['hit_counts'][i] += 1
                matched_candidates.add(i)
                matched_raw.add(j)
                break

    for i in range(len(t['candidates'])):
        if i not in matched_candidates:
            t['hit_counts'][i] = 0

    for j, raw in enumerate(raw_boxes):
        if j not in matched_raw:
            t['candidates'].append(raw)
            t['hit_counts'].append(1)

    new_candidates, new_hits = [], []
    for box, hits in zip(t['candidates'], t['hit_counts']):
        if hits >= ID_CONFIRM_FRAMES:
            if not any(box_iou(box, cb) > IOU_THRESH for cb in t['boxes']):
                t['boxes'].append(box)
                t['miss_counts'].append(0)
        elif hits > 0:
            new_candidates.append(box)
            new_hits.append(hits)
    t['candidates'] = new_candidates
    t['hit_counts'] = new_hits

    matched_confirmed = set()
    for j, raw in enumerate(raw_boxes):
        for i, conf in enumerate(t['boxes']):
            if box_iou(raw, conf) > IOU_THRESH:
                t['boxes'][i] = smooth_box(conf, raw)  # smooth position update
                t['miss_counts'][i] = 0
                matched_confirmed.add(i)
                break

    for i in range(len(t['boxes'])):
        if i not in matched_confirmed:
            t['miss_counts'][i] += 1

    # Remove confirmed boxes that have been missing too long
    surviving = [(b, m) for b, m in zip(t['boxes'], t['miss_counts'])
                 if m <= ID_FORGET_FRAMES]
    t['boxes'] = [x[0] for x in surviving]
    t['miss_counts'] = [x[1] for x in surviving]

    return t['boxes']


def run_detection(frame):
    output = frame.copy()
    face_count = 0

    GREEN = (0, 255, 136)
    RED   = (50, 50, 255)
    BLUE  = (255, 100, 0)

    # ── Face detection (unchanged) ───────────────────────────────
    results_face = model_face.predict(
        source=frame, conf=FACE_CONFIDENCE,
        verbose=False, imgsz=PROCESS_SIZE, device=DEVICE)

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

    # ── ID card detection with upscaling + dual model + temporal smoothing ──
    h, w = frame.shape[:2]
    upscaled = cv2.resize(frame, (int(w * ID_UPSCALE), int(h * ID_UPSCALE)),
                          interpolation=cv2.INTER_CUBIC)

    raw_id_boxes = []

    def collect_boxes(model):
        results = model.predict(
            source=upscaled, conf=ID_CONFIDENCE,
            verbose=False, imgsz=PROCESS_SIZE, device=DEVICE)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = int(x1 / ID_UPSCALE); y1 = int(y1 / ID_UPSCALE)
                x2 = int(x2 / ID_UPSCALE); y2 = int(y2 / ID_UPSCALE)
                bw, bh = x2 - x1, y2 - y1
                if bw < 30 or bh < 20:
                    continue
                raw_id_boxes.append((x1, y1, x2, y2))

    # Run primary model
    collect_boxes(model_idcard)

    # Run second model if loaded
    if model_idcard2 is not None:
        collect_boxes(model_idcard2)

    confirmed_boxes = update_id_tracker(raw_id_boxes)
    for (x1, y1, x2, y2) in confirmed_boxes:
        roi = output[y1:y2, x1:x2]
        if roi.size > 0:
            output[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 30)
        draw_box(output, x1, y1, x2, y2, BLUE, "ID Card [blurred]")

    return output, face_count, len(confirmed_boxes)


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
    print("\n" + "=" * 70)
    print("🔒 RTIOC - Real-Time Identity and Object Concealment")
    print("=" * 70)
    print("➡️  Open: http://localhost:5000")
    print("=" * 70 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

