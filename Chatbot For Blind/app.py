"""
NAIN AI — Fixed Final Version
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixes in this version:
  ✓ Web search ALWAYS works — keyword-based routing, not tool calling
  ✓ Allu Arjun / movies / actors / celebrities → forced web search
  ✓ Stock market / prices / news / weather → forced web search
  ✓ Spotify opens when asked
  ✓ "Close youtube" → closes only the YouTube tab (not whole browser)
  ✓ "Close spotify" → closes Spotify app or its browser tab only
  ✓ "Close browser" → closes the entire browser window
  ✓ Fresh start on every run — no stale chat, no stale lock file
  ✓ Audio works reliably on every restart
  ✓ Double voice fixed
  ✓ Camera close works
  ✓ Live vision via Moondream — say "open camera" → auto object detection starts
  ✓ Voice commands work from background threads — persistent state stored in
    sys.modules (process-level), NOT session_state (main-thread only)

Run:
  streamlit run app.py
"""

import streamlit as st
import speech_recognition as sr
import subprocess
import webbrowser
import urllib.parse
import requests
import ollama
import datetime
import wikipedia
import cv2
import os
import sys
import json
import types
import threading
import time
import base64

# ── YOLO — primary object detector ──
try:
    from ultralytics import YOLO as _YOLO
    _yolo_model = _YOLO("yolov8n.pt")   # nano — fastest, auto-downloads on first run
    YOLO_AVAILABLE = True
    print("[VISION] YOLOv8 loaded ✓")
except Exception as _ye:
    _yolo_model    = None
    YOLO_AVAILABLE = False
    print(f"[VISION] YOLO not available ({_ye}) — will use Moondream fallback")

# ─────────────────────────────────────────────────────────
#  PERSISTENT STATE — stored in sys.modules
#  ─────────────────────────────────────────────────────────
#  Streamlit re-executes this entire script on every rerender
#  (every 1-second autorefresh).  Module-level variables like
#  threading.Event() would be recreated as NEW objects each
#  time — background threads hold references to the OLD objects
#  and stop responding.
#
#  st.session_state is NOT accessible from background threads.
#
#  Solution: store shared objects in sys.modules once, then
#  bind them to local names.  sys.modules is process-level and
#  survives rerenders AND is visible from any thread.
# ─────────────────────────────────────────────────────────
if "_naina_state" not in sys.modules:
    _ns = types.ModuleType("_naina_state")
    _ns.cam_running           = threading.Event()   # camera on/off signal
    _ns.speak_lock            = threading.Lock()    # TTS mutex
    _ns.speak_process         = None                # current PowerShell TTS process
    _ns.frame_lock            = threading.Lock()    # latest_frame mutex
    _ns.latest_frame          = None                # last camera frame (numpy array)
    _ns.voice_started         = False               # guard — start voice thread once
    _ns.vision_thread_running = False               # moondream loop thread guard
    sys.modules["_naina_state"] = _ns

# Bind to short names used throughout the file
_ns            = sys.modules["_naina_state"]
_cam_running   = _ns.cam_running
_speak_lock    = _ns.speak_lock
_frame_lock    = _ns.frame_lock

# ── Defensive patch: add any attributes that old cached state may be missing ──
if not hasattr(_ns, 'vision_thread_running'):
    _ns.vision_thread_running = False

# ─────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────
MODEL        = "qwen2.5:0.5b"
VISION_MODEL = "moondream"
TRANSCRIPT_F = "naina_transcript.json"
STATUS_F     = "naina_status.txt"
CAMERA_F     = "naina_camera.txt"
LOOP_LOCK_F  = "naina_loop.lock"

# ── Keywords that ALWAYS trigger web search ──
WEB_SEARCH_KW = [
    # people / celebrities
    'who is','who was','who are','actor','actress','movie','film',
    'singer','director','celebrity','allu arjun','hero','heroine',
    # current events
    'news','today','latest','current','right now','recently',
    'happened','breaking','update','2024','2025','2026',
    # stocks / finance
    'stock','share price','market','nifty','sensex','nasdaq',
    'bitcoin','crypto','rupee','dollar','gold price','oil price',
    # sports
    'score','match','ipl','cricket','football','winner','lost',
    # weather
    'weather','temperature','forecast','rain','sunny',
    # president / PM / positions
    'president','prime minister','ceo','governor','minister',
    # anything about movies / shows
    'release date','box office','ott','streaming','trailer',
]

STOP_WORDS       = {'stop','wait','shut up','enough','quiet','silence','pause','cancel'}
CAM_OPEN_KW      = [
    'open camera','open the camera','start camera','start the camera',
    'turn on camera','turn on the camera','show camera','show the camera',
    'camera on','launch camera','activate camera','enable camera',
    'open my camera','camera please','use camera',
]
CAM_CLOSE_KW     = [
    'close camera','close the camera','stop camera','stop the camera',
    'turn off camera','turn off the camera','hide camera','hide the camera',
    'camera off','shut camera','disable camera','end camera',
]
DESCRIBE_KW      = ['describe','what do you see','look around',
                    'what is in front','what can you see','tell me what you see']
YOUTUBE_KW       = ['play','open youtube','youtube']
SPOTIFY_KW       = ['open spotify','spotify','play spotify','music on spotify']
YOUTUBE_CLOSE_KW = ['close youtube','stop youtube','close the youtube','close youtube tab']
SPOTIFY_CLOSE_KW = ['close spotify','stop spotify','close the spotify','close spotify tab']
BROWSER_CLOSE_KW = ['close browser','close tab','stop video',
                     'close the video','close it','stop it','close window']
WIKI_KW          = ['tell me about','wikipedia','explain what is','explain']
TIME_KW          = ['what time','current time','time now']
DATE_KW          = ['what date','today date','what day','current date']

# ─────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NAIN AI",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@800&display=swap');
:root{
  --bg:#06090f;--surface:#0b1018;--border:#182333;
  --accent:#29d9ff;--green:#00f090;--orange:#ff8c30;
  --red:#ff2d55;--muted:#334455;--text:#aabccc;
  --mono:'IBM Plex Mono',monospace;--title:'Syne',sans-serif;
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{
  background:var(--bg)!important;font-family:var(--mono);color:var(--text);}
#MainMenu,footer,header,
[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],[data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important;}

.hdr{text-align:center;padding:1.8rem 0 1.1rem;border-bottom:1px solid var(--border);}
.hdr-title{font-family:var(--title);font-size:2.8rem;letter-spacing:.3em;color:var(--accent);line-height:1;}
.hdr-sub{font-size:.58rem;color:var(--muted);letter-spacing:.35em;margin-top:.4rem;text-transform:uppercase;}

.ss{display:flex;align-items:center;gap:.8rem;padding:.85rem 1.5rem;
    background:var(--surface);border-bottom:1px solid var(--border);}
.dot{width:12px;height:12px;border-radius:50%;flex-shrink:0;}
.dot-listening{background:var(--accent);box-shadow:0 0 10px var(--accent);animation:blink .8s infinite;}
.dot-thinking{background:var(--orange);animation:blink .55s infinite;}
.dot-speaking{background:var(--green);box-shadow:0 0 10px var(--green);animation:blink 1.1s infinite;}
.dot-stopped{background:var(--red);box-shadow:0 0 14px var(--red);}
.dot-camera{background:var(--orange);box-shadow:0 0 10px var(--orange);animation:blink .9s infinite;}
.dot-searching{background:#a855f7;box-shadow:0 0 10px #a855f7;animation:blink .7s infinite;}
.dot-boot,.dot-ready{background:var(--muted);}
.dot-error{background:var(--red);}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}

.slbl{font-size:.72rem;font-weight:600;letter-spacing:.22em;text-transform:uppercase;}
.slbl.listening{color:var(--accent);}
.slbl.thinking{color:var(--orange);}
.slbl.speaking{color:var(--green);}
.slbl.stopped{color:var(--red);}
.slbl.camera{color:var(--orange);}
.slbl.searching{color:#a855f7;}
.slbl.boot,.slbl.ready{color:var(--muted);}
.slbl.error{color:var(--red);}
.shint{margin-left:auto;font-size:.58rem;color:var(--muted);letter-spacing:.1em;}

.osc{display:flex;align-items:center;justify-content:center;gap:2px;height:46px;
     background:var(--surface);border-bottom:1px solid var(--border);}
.ob{width:3px;border-radius:2px;background:var(--green);
    animation:oa .6s ease-in-out infinite alternate;}
.ob:nth-child(1){height:5px;animation-delay:.00s}
.ob:nth-child(2){height:9px;animation-delay:.03s}
.ob:nth-child(3){height:15px;animation-delay:.06s}
.ob:nth-child(4){height:22px;animation-delay:.09s;background:var(--accent)}
.ob:nth-child(5){height:29px;animation-delay:.12s}
.ob:nth-child(6){height:34px;animation-delay:.15s;background:var(--accent)}
.ob:nth-child(7){height:36px;animation-delay:.18s}
.ob:nth-child(8){height:34px;animation-delay:.21s;background:var(--accent)}
.ob:nth-child(9){height:29px;animation-delay:.24s}
.ob:nth-child(10){height:22px;animation-delay:.27s;background:var(--orange)}
.ob:nth-child(11){height:22px;animation-delay:.30s;background:var(--orange)}
.ob:nth-child(12){height:29px;animation-delay:.33s}
.ob:nth-child(13){height:34px;animation-delay:.36s;background:var(--accent)}
.ob:nth-child(14){height:36px;animation-delay:.39s}
.ob:nth-child(15){height:34px;animation-delay:.42s;background:var(--accent)}
.ob:nth-child(16){height:29px;animation-delay:.45s}
.ob:nth-child(17){height:22px;animation-delay:.48s;background:var(--accent)}
.ob:nth-child(18){height:15px;animation-delay:.51s}
.ob:nth-child(19){height:9px;animation-delay:.54s}
.ob:nth-child(20){height:5px;animation-delay:.57s}
@keyframes oa{from{transform:scaleY(.2);opacity:.4}to{transform:scaleY(1.7);opacity:1}}

.mw{display:flex;align-items:center;justify-content:center;gap:4px;height:46px;
    background:var(--surface);border-bottom:1px solid var(--border);}
.mb{width:4px;border-radius:2px;background:var(--accent);
    animation:ma .75s ease-in-out infinite alternate;}
.mb:nth-child(1){height:8px;animation-delay:.00s}
.mb:nth-child(2){height:18px;animation-delay:.10s;background:var(--orange)}
.mb:nth-child(3){height:28px;animation-delay:.20s}
.mb:nth-child(4){height:18px;animation-delay:.30s;background:var(--orange)}
.mb:nth-child(5){height:8px;animation-delay:.40s}
.mb:nth-child(6){height:22px;animation-delay:.50s;background:var(--green)}
.mb:nth-child(7){height:12px;animation-delay:.60s}
.mb:nth-child(8){height:6px;animation-delay:.70s}
@keyframes ma{from{transform:scaleY(.35)}to{transform:scaleY(2.3)}}

.sw{display:flex;align-items:center;justify-content:center;gap:3px;height:46px;
    background:var(--surface);border-bottom:1px solid var(--border);}
.sb{width:3px;border-radius:2px;background:#a855f7;
    animation:sa .5s ease-in-out infinite alternate;}
.sb:nth-child(odd){height:12px;}.sb:nth-child(even){height:24px;}
@keyframes sa{from{transform:scaleY(.3)}to{transform:scaleY(1.8)}}

.cw{display:flex;align-items:center;justify-content:center;gap:4px;height:46px;
    background:var(--surface);border-bottom:1px solid var(--border);}
.cb{width:4px;border-radius:2px;background:var(--orange);
    animation:ca .9s ease-in-out infinite alternate;}
.cb:nth-child(odd){height:8px;}.cb:nth-child(even){height:18px;}
@keyframes ca{from{transform:scaleY(.3)}to{transform:scaleY(2.0)}}

.stop-banner{display:flex;align-items:center;justify-content:center;gap:.7rem;
  height:46px;background:#180008;border-bottom:1px solid var(--red);
  color:var(--red);font-size:.72rem;font-weight:600;
  letter-spacing:.28em;text-transform:uppercase;}
.sq{width:14px;height:14px;background:var(--red);border-radius:2px;
    box-shadow:0 0 12px var(--red);}

.tx{background:var(--surface);border-left:1px solid var(--border);
    border-right:1px solid var(--border);
    padding:1.2rem 1.5rem;min-height:320px;max-height:320px;
    overflow-y:auto;display:flex;flex-direction:column;gap:1rem;}
.tx::-webkit-scrollbar{width:2px;}
.tx::-webkit-scrollbar-thumb{background:var(--border);}
.sys{text-align:center;font-size:.6rem;color:var(--muted);letter-spacing:.18em;}
.ur{display:flex;flex-direction:column;align-items:flex-end;}
.ur .lb{font-size:.56rem;color:var(--muted);letter-spacing:.2em;margin-bottom:.25rem;}
.ur .bb{background:#091c2e;border:1px solid #1a4560;
        border-radius:14px 14px 2px 14px;padding:.65rem 1.1rem;
        color:var(--accent);font-size:.82rem;line-height:1.55;
        max-width:88%;word-break:break-word;}
.ar{display:flex;flex-direction:column;align-items:flex-start;}
.ar .lb{font-size:.56rem;color:var(--muted);letter-spacing:.2em;margin-bottom:.25rem;}
.ar .bb{background:#061410;border:1px solid #0c3828;
        border-radius:14px 14px 14px 2px;padding:.65rem 1.1rem;
        color:var(--green);font-size:.82rem;line-height:1.6;
        max-width:88%;word-break:break-word;}
.ft{background:var(--surface);border:1px solid var(--border);border-top:none;
    border-radius:0 0 10px 10px;padding:.6rem 1.5rem;
    display:flex;justify-content:space-between;
    font-size:.58rem;color:var(--muted);letter-spacing:.12em;}
.lg{background:var(--surface);border:1px solid var(--border);
    border-radius:10px;padding:1rem 1.5rem;margin-top:1.2rem;
    display:grid;grid-template-columns:1fr 1fr;gap:.4rem 2rem;
    font-size:.63rem;color:var(--muted);letter-spacing:.06em;}
.lg-t{grid-column:1/-1;font-size:.57rem;letter-spacing:.28em;text-transform:uppercase;
      padding-bottom:.5rem;border-bottom:1px solid var(--border);margin-bottom:.1rem;}
.lg-i span{color:var(--accent);font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  FILE STATE
# ─────────────────────────────────────────────────────────
_file_lock = threading.Lock()

def read_transcript():
    try:
        with open(TRANSCRIPT_F,"r",encoding="utf-8") as f: return json.load(f)
    except: return []

def append_transcript(role, text):
    with _file_lock:
        try:
            data = read_transcript()
            data.append({"role":role,"text":text,
                         "ts":datetime.datetime.now().strftime("%H:%M:%S")})
            with open(TRANSCRIPT_F,"w",encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except: pass

def read_status():
    try:
        with open(STATUS_F,"r") as f: return f.read().strip().lower()
    except: return "boot"

def write_status(s):
    try:
        with open(STATUS_F,"w") as f: f.write(s.upper())
    except: pass

def read_camera_state():
    try:
        with open(CAMERA_F,"r") as f: return f.read().strip().upper()
    except: return "OFF"

def write_camera_state(s):
    try:
        with open(CAMERA_F,"w") as f: f.write(s.upper())
    except: pass

# ─────────────────────────────────────────────────────────
#  TTS — Windows PowerShell SAPI
#  Uses _ns.speak_lock and _ns.speak_process from sys.modules
#  so these survive rerenders and are visible from threads.
# ─────────────────────────────────────────────────────────
def speak(text: str):
    if not text or text.startswith("__"): return
    clean = (text.replace('"',"'").replace('\n',' ')
                 .replace('*','').replace('#','').replace('`','')
                 .replace('[','').replace(']','').strip())
    if not clean: return
    with _ns.speak_lock:
        try:
            if _ns.speak_process and _ns.speak_process.poll() is None:
                _ns.speak_process.terminate()
                _ns.speak_process.wait(timeout=1)
        except: pass
        try:
            ps = (
                'Add-Type -AssemblyName System.Speech;'
                '$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;'
                '$s.Rate=1;$s.Volume=100;'
                f'$s.Speak("{clean}");'
            )
            proc = subprocess.Popen(
                ["powershell","-NoProfile","-Command",ps],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _ns.speak_process = proc
            proc.wait()
        except Exception as e:
            print(f"[TTS ERROR] {e}")

def stop_tts():
    with _ns.speak_lock:
        try:
            if _ns.speak_process and _ns.speak_process.poll() is None:
                _ns.speak_process.terminate()
                _ns.speak_process.wait(timeout=1)
        except: pass

def kill_old_speech_processes():
    """Kill any leftover PowerShell SAPI processes from a previous run."""
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "powershell.exe"],
            capture_output=True
        )
    except:
        pass

# ─────────────────────────────────────────────────────────
#  WEB SEARCH — DuckDuckGo (no API key)
# ─────────────────────────────────────────────────────────
def web_search(query: str) -> str:
    print(f"[WEB SEARCH] {query}")
    try:
        encoded = urllib.parse.quote(query)
        url     = (f"https://api.duckduckgo.com/?q={encoded}"
                   f"&format=json&no_html=1&skip_disambig=1")
        resp    = requests.get(url, timeout=7,
                               headers={"User-Agent":"NainaAI/1.0"})
        data    = resp.json()

        parts = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"])
        for t in data.get("RelatedTopics",[])[:3]:
            if isinstance(t,dict) and t.get("Text"):
                parts.append(t["Text"])
        if not parts and data.get("Answer"):
            parts.append(data["Answer"])
        if not parts and data.get("Definition"):
            parts.append(data["Definition"])

        result = " ".join(parts[:2]).strip()
        if result:
            print(f"[WEB RESULT] {result[:100]}...")
            return result

    except Exception as e:
        print(f"[SEARCH ERROR] {e}")

    try:
        wikipedia.set_lang("en")
        return wikipedia.summary(query, sentences=3, auto_suggest=True)
    except:
        pass

    return f"I searched for {query} but couldn't find a clear result right now."


def needs_web_search(cmd: str) -> bool:
    return any(kw in cmd for kw in WEB_SEARCH_KW)


def search_and_answer(question: str) -> str:
    write_status("SEARCHING")
    speak("Let me search that for you.")
    search_result = web_search(question)
    try:
        system = (
            "You are Nain, a friendly AI voice assistant. "
            "Using the search result below, answer in ONE or TWO short spoken sentences. "
            "No bullet points, no lists, no asterisks, no markdown. "
            "Sound warm and human. Speak directly to the user."
        )
        prompt = (
            f"Question: {question}\n\n"
            f"Search result: {search_result}\n\n"
            "Give a short spoken answer based on the search result."
        )
        resp = ollama.chat(
            model=MODEL,
            messages=[
                {"role":"system","content":system},
                {"role":"user",  "content":prompt},
            ]
        )
        answer = resp["message"]["content"].strip()
        answer = (answer.replace('*','').replace('#','')
                        .replace('`','').replace('\n',' ').strip())
        return answer if answer else search_result
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return search_result

# ─────────────────────────────────────────────────────────
#  BROWSER CONTROL
# ─────────────────────────────────────────────────────────
def _close_tab_by_title(title_keyword: str) -> bool:
    """Find the first browser window whose title contains title_keyword
    and close just that tab with Ctrl+W. Returns True if found."""
    ps_script = f"""
    Add-Type -AssemblyName System.Windows.Forms
    $shell = New-Object -ComObject WScript.Shell
    $procs = Get-Process | Where-Object {{ $_.MainWindowTitle -like '*{title_keyword}*' }}
    if ($procs) {{
        $shell.AppActivate($procs[0].MainWindowTitle) | Out-Null
        Start-Sleep -Milliseconds 400
        [System.Windows.Forms.SendKeys]::SendWait('^w')
        Write-Output 'closed'
    }} else {{
        Write-Output 'not_found'
    }}
    """
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, text=True, timeout=6)
        return "closed" in result.stdout
    except Exception as e:
        print(f"[BROWSER TAB CLOSE ERROR] {e}")
        return False

def close_youtube_tab() -> bool:
    """Close only the YouTube tab, not the whole browser."""
    return _close_tab_by_title("YouTube")

def close_spotify_app() -> bool:
    """Close the Spotify desktop app; fall back to closing the browser tab."""
    # Try desktop app first
    try:
        result = subprocess.run(
            ["taskkill", "/F", "/IM", "Spotify.exe"],
            capture_output=True, text=True)
        if result.returncode == 0:
            print("[SPOTIFY] Closed Spotify.exe")
            return True
    except: pass
    # Fall back to closing the browser tab
    return _close_tab_by_title("Spotify")

def close_browser_windows():
    """Close the entire browser — used only for generic 'close browser' commands."""
    closed = False
    for proc in ["chrome.exe","msedge.exe","firefox.exe","brave.exe"]:
        try:
            result = subprocess.run(
                ["taskkill","/F","/IM",proc],
                capture_output=True, text=True)
            if result.returncode == 0:
                closed = True
                print(f"[BROWSER] Closed {proc}")
        except: pass
    return closed

# ─────────────────────────────────────────────────────────
#  SPOTIFY
# ─────────────────────────────────────────────────────────
def open_spotify(query: str = "") -> str:
    try:
        spotify_path = (
            r"C:\Users\\" + os.getenv("USERNAME","") +
            r"\AppData\Roaming\Spotify\Spotify.exe"
        )
        if os.path.exists(spotify_path):
            if query:
                encoded = urllib.parse.quote(query)
                subprocess.Popen([spotify_path])
                time.sleep(2)
                webbrowser.open(f"https://open.spotify.com/search/{encoded}")
            else:
                subprocess.Popen([spotify_path])
            return f"Opening Spotify{' and searching for ' + query if query else ''}."
    except: pass
    if query:
        encoded = urllib.parse.quote(query)
        webbrowser.open(f"https://open.spotify.com/search/{encoded}")
        return f"Opening Spotify search for {query}."
    else:
        webbrowser.open("https://open.spotify.com")
        return "Opening Spotify."

# ─────────────────────────────────────────────────────────
#  YOUTUBE
# ─────────────────────────────────────────────────────────
def open_youtube(query: str) -> str:
    try:
        from youtubesearchpython import VideosSearch
        r = VideosSearch(query, limit=1).result()
        if r and r["result"]:
            webbrowser.open(r["result"][0]["link"])
            return f"Playing {r['result'][0]['title']} on YouTube."
    except: pass
    encoded = urllib.parse.quote(query)
    webbrowser.open(f"https://www.youtube.com/results?search_query={encoded}")
    return f"Opening YouTube search for {query}."

# ─────────────────────────────────────────────────────────
#  CAMERA + VISION
#  Primary detector  : YOLOv8  (ultralytics)  — fast, precise
#  Fallback detector : Moondream (ollama)      — if YOLO missing
# ─────────────────────────────────────────────────────────
def _is_good_frame(frame_bgr) -> bool:
    """
    Returns True only if the frame has actual content.
    Rejects black frames (mean brightness < 10) and
    corrupt frames (near-zero standard deviation).
    """
    if frame_bgr is None:
        return False
    try:
        import numpy as np
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean  = float(gray.mean())
        stdv  = float(gray.std())
        ok = mean > 10 and stdv > 5
        if not ok:
            print(f"[CAM] Bad frame — mean={mean:.1f} std={stdv:.1f} — skipping")
        return ok
    except Exception:
        return False


def _open_camera_device():
    """
    Try every backend in order until one gives a good frame.
    Returns an opened cv2.VideoCapture or None.
    """
    import numpy as np
    backends = []
    if hasattr(cv2, 'CAP_DSHOW'):   backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, 'CAP_MSMF'):    backends.append(cv2.CAP_MSMF)
    backends.append(None)  # default / no flag

    for backend in backends:
        try:
            cap = cv2.VideoCapture(0, backend) if backend is not None else cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.release(); continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # Drain the buffer and warm up — discard first 20 frames
            for _ in range(20):
                cap.read()

            # Verify we get a real frame
            ret, test = cap.read()
            if ret and _is_good_frame(test):
                label = {cv2.CAP_DSHOW: "DSHOW",
                         cv2.CAP_MSMF:  "MSMF"}.get(backend, "DEFAULT") if backend else "DEFAULT"
                print(f"[CAM] Opened with backend={label} ✓")
                return cap
            cap.release()
        except Exception as e:
            print(f"[CAM] Backend {backend} failed: {e}")

    return None


def _camera_loop():
    """Captures live frames into _ns.latest_frame (RGB numpy array)."""
    cam = _open_camera_device()
    if cam is None:
        append_transcript("system", "── Camera not found or gives black frames ──")
        write_camera_state("OFF")
        _ns.cam_running.clear()
        return

    print("[CAM] Camera loop running ✓")
    while _ns.cam_running.is_set():
        ret, frame = cam.read()
        if ret and frame is not None:
            # Store EVERY frame for live display — no quality filter here
            with _ns.frame_lock:
                _ns.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        time.sleep(0.05)

    cam.release()
    with _ns.frame_lock:
        _ns.latest_frame = None
    print("[CAM] Camera loop stopped ✓")


def _get_frame_bgr():
    """Return latest good BGR frame from stream."""
    with _ns.frame_lock:
        rgb = _ns.latest_frame
    if rgb is not None:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return None


def _detect_yolo(frame_bgr) -> str:
    """
    Run YOLOv8 and return a spoken sentence, e.g.
    "I can see 1 person, 2 laptops, and 1 bottle in front of you."
    Returns "" if nothing detected or YOLO unavailable.
    """
    if not YOLO_AVAILABLE or _yolo_model is None:
        return ""
    try:
        from collections import Counter
        results  = _yolo_model(frame_bgr, verbose=False)[0]
        counts   = Counter(results.names[int(b.cls[0])] for b in results.boxes)
        if not counts:
            return "I don't see any recognisable objects right now."
        parts = []
        for label, n in counts.most_common():
            parts.append(f"{n} {label if n == 1 else label + 's'}")
        if len(parts) == 1:
            listed = parts[0]
        elif len(parts) == 2:
            listed = f"{parts[0]} and {parts[1]}"
        else:
            listed = ", ".join(parts[:-1]) + f", and {parts[-1]}"
        sentence = f"I can see {listed} in front of you."
        print(f"[YOLO] {sentence}")
        return sentence
    except Exception as e:
        print(f"[YOLO ERROR] {e}")
        return ""


def _detect_moondream(frame_bgr) -> str:
    """Fallback: send frame to Moondream and return its description."""
    try:
        ret, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return ""
        img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        print(f"[MOONDREAM] Sending frame ({len(img_b64)} chars)…")
        resp = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": ("List every object visible in this image in one short sentence. "
                            "Start with: I can see…"),
                "images": [img_b64]
            }]
        )
        result = resp["message"]["content"].strip()
        print(f"[MOONDREAM] {result}")
        return result
    except Exception as e:
        print(f"[MOONDREAM ERROR] {e}")
        return ""


def describe_with_vision(question: str = "") -> str:
    """
    Detect objects in the current camera frame.
    Uses YOLO if installed, otherwise Moondream.
    """
    frame_bgr = _get_frame_bgr()
    if frame_bgr is None:
        return "I could not access the camera right now."

    # Try YOLO first
    result = _detect_yolo(frame_bgr)
    if result:
        return result

    # Moondream fallback
    result = _detect_moondream(frame_bgr)
    if result:
        return result

    return "I can see the camera feed but could not identify any objects."


def _vision_loop():
    """
    Daemon thread: detects objects every 4 seconds and speaks them aloud.
    Waits for the first real frame before starting.
    """
    _ns.vision_thread_running = True
    print("[VISION] Object-detection loop started ✓")

    # Wait up to 10 s for the first GOOD frame
    for _ in range(100):
        if not _ns.cam_running.is_set():
            _ns.vision_thread_running = False
            return
        frame_check = _get_frame_bgr()
        if frame_check is not None and _is_good_frame(frame_check):
            break
        time.sleep(0.1)

    engine = "YOLO" if YOLO_AVAILABLE else "Moondream"
    print(f"[VISION] First frame ready — scanning with {engine}")

    try:
        while _ns.cam_running.is_set():
            result = describe_with_vision()
            bad_phrases = ("could not", "error", "unavailable", "please ensure")
            if result and not any(p in result.lower() for p in bad_phrases):
                append_transcript("ai", result)
                speak(result)
            else:
                print(f"[VISION] Skipped: {result}")

            # 4-second gap between scans
            for _ in range(40):
                if not _ns.cam_running.is_set():
                    break
                time.sleep(0.1)

    except Exception as e:
        print(f"[VISION LOOP ERROR] {e}")
    finally:
        _ns.vision_thread_running = False
        print("[VISION] Object-detection loop stopped ✓")


def open_camera() -> str:
    """Open camera and start live object detection immediately."""
    try:
        _ns.cam_running.set()
        write_camera_state("ON")
        threading.Thread(target=_camera_loop, daemon=True).start()
        if not getattr(_ns, 'vision_thread_running', False):
            threading.Thread(target=_vision_loop, daemon=True).start()
        engine = "YOLO" if YOLO_AVAILABLE else "Moondream"
        return (f"Camera is open. I am using {engine} to detect objects. "
                "I will tell you what I see every few seconds. Say close camera to stop.")
    except Exception as e:
        print(f"[CAMERA OPEN ERROR] {e}")
        write_camera_state("OFF")
        return "Sorry, I could not open the camera. Please check if it is connected."


def close_camera() -> str:
    _ns.cam_running.clear()
    write_camera_state("OFF")
    return "Camera closed. Listening again."

# ─────────────────────────────────────────────────────────
#  DIRECT OLLAMA — for simple conversational questions
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Nain, a friendly AI voice assistant with live camera and vision capabilities. "
    "You can open the camera, detect objects, and describe what you see using Moondream. "
    "Introduce yourself as Nain when asked. "
    "Reply in ONE or TWO short spoken sentences only. "
    "No bullet points, no lists, no asterisks, no markdown, no symbols. "
    "Sound warm and human like a caring friend."
)

def ask_ollama_direct(command: str) -> str:
    try:
        full = ""
        buf  = ""
        stream = ollama.chat(
            model=MODEL,
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user",  "content":command},
            ],
            stream=True
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            full += token
            buf  += token
            if any(buf.endswith(p) for p in [". ","? ","! ","\n"]):
                s = buf.strip(); buf = ""
                if s: speak(s)
        if buf.strip(): speak(buf.strip())
        return full.strip() or "I am not sure about that."
    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        msg = "I had trouble answering that."
        speak(msg)
        return msg

# ─────────────────────────────────────────────────────────
#  PROCESS COMMAND — clear priority order
# ─────────────────────────────────────────────────────────
def process_command(command: str) -> str:
    cmd = command.lower().strip()

    # ── 0. Broad camera safety-net ──
    # Catches "open the camera", "camera chalao", "can you open camera" etc.
    # even if the exact keyword list misses due to filler words.
    if 'camera' in cmd:
        open_words  = {'open','start','turn on','show','launch','activate',
                       'enable','use','please','on','chalao','kholo'}
        close_words = {'close','stop','turn off','hide','off','shut','disable',
                       'end','band','bund'}
        if any(w in cmd for w in close_words):
            r = close_camera(); speak(r); return r
        if any(w in cmd for w in open_words):
            r = open_camera(); speak(r); return r

    # ── 1. Stop ──
    if any(w in cmd for w in STOP_WORDS):
        stop_tts(); return "__STOP__"

    # ── 2. Exit ──
    if any(w in cmd for w in ["exit","goodbye","bye","quit","shut down"]):
        return "__EXIT__"

    # ── 3. Close YouTube tab only ──
    if any(kw in cmd for kw in YOUTUBE_CLOSE_KW):
        closed = close_youtube_tab()
        r = "YouTube tab closed." if closed else "No YouTube tab found to close."
        speak(r); return r

    # ── 3b. Close Spotify only ──
    if any(kw in cmd for kw in SPOTIFY_CLOSE_KW):
        closed = close_spotify_app()
        r = "Spotify closed." if closed else "Spotify was not open."
        speak(r); return r

    # ── 3c. Close entire browser ──
    if any(kw in cmd for kw in BROWSER_CLOSE_KW):
        closed = close_browser_windows()
        r = "Browser closed." if closed else "No browser window found to close."
        speak(r); return r

    # ── 4. Camera open ──
    if any(kw in cmd for kw in CAM_OPEN_KW):
        r = open_camera(); speak(r); return r

    # ── 5. Camera close ──
    if any(kw in cmd for kw in CAM_CLOSE_KW):
        r = close_camera(); speak(r); return r

    # ── 6. Describe / vision ──
    if any(kw in cmd for kw in DESCRIBE_KW):
        speak("Let me take a look.")
        r = describe_with_vision(command)
        speak(r); return r

    # ── 7. Spotify ──
    if any(kw in cmd for kw in SPOTIFY_KW):
        query = cmd
        for kw in sorted(SPOTIFY_KW, key=len, reverse=True):
            query = query.replace(kw,"").strip()
        query = query.replace("on spotify","").replace("for me","").strip()
        r = open_spotify(query); speak(r); return r

    # ── 8. YouTube / play ──
    if any(kw in cmd for kw in YOUTUBE_KW):
        query = cmd
        for kw in sorted(YOUTUBE_KW, key=len, reverse=True):
            query = query.replace(kw,"").strip()
        query = (query.replace("on youtube","")
                      .replace("for me","")
                      .replace("song","").strip())
        if not query: query = "trending music"
        r = open_youtube(query); speak(r); return r

    # ── 9. Time ──
    if any(kw in cmd for kw in TIME_KW) or ("time" in cmd and "what" in cmd):
        r = f"It is {datetime.datetime.now().strftime('%I:%M %p')}."
        speak(r); return r

    # ── 10. Date ──
    if any(kw in cmd for kw in DATE_KW) or ("today" in cmd and "what" in cmd):
        r = f"Today is {datetime.datetime.now().strftime('%A, %B %d, %Y')}."
        speak(r); return r

    # ── 11. Web search — keyword-forced ──
    if needs_web_search(cmd):
        r = search_and_answer(command)
        write_status("SPEAKING")
        speak(r); return r

    # ── 12. Everything else → direct Ollama ──
    return ask_ollama_direct(command)

# ─────────────────────────────────────────────────────────
#  VOICE LOOP — never dies
# ─────────────────────────────────────────────────────────
def voice_loop():
    rec = sr.Recognizer()
    rec.energy_threshold         = 300
    rec.dynamic_energy_threshold = True
    rec.pause_threshold          = 0.6
    rec.non_speaking_duration    = 0.3

    print("[LOOP] Voice loop started ✓")
    time.sleep(1.5)  # give SAPI time to initialize

    write_status("SPEAKING")
    greeting = "Hi, I am Nain. I am listening. Say your command."
    append_transcript("ai", greeting)
    speak(greeting)

    while True:
        write_status("LISTENING")
        command = None

        try:
            with sr.Microphone() as src:
                rec.adjust_for_ambient_noise(src, duration=0.3)
                audio = rec.listen(src, timeout=10, phrase_time_limit=12)
            command = rec.recognize_google(audio).lower().strip()
            print(f"[USER] {command}")
        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            continue
        except Exception as e:
            print(f"[MIC ERROR] {e}")
            write_status("ERROR")
            time.sleep(1)
            continue

        if not command: continue

        if any(w in command for w in STOP_WORDS):
            stop_tts()
            write_status("STOPPED")
            append_transcript("system","── Stopped ──")
            time.sleep(0.8)
            continue

        if any(w in command for w in ["exit","goodbye","bye","quit"]):
            write_status("SPEAKING")
            farewell = "Goodbye. Take care."
            append_transcript("ai", farewell)
            speak(farewell)
            write_status("READY")
            try: os.remove(LOOP_LOCK_F)
            except: pass
            break

        append_transcript("user", command)
        write_status("THINKING")

        try:
            write_status("SPEAKING")
            response = process_command(command)
        except Exception as e:
            print(f"[PROCESS ERROR] {e}")
            response = "I had trouble with that. Please try again."
            speak(response)

        if response == "__STOP__":
            write_status("STOPPED")
            append_transcript("system","── Stopped ──")
            time.sleep(0.8); continue

        if response == "__EXIT__":
            speak("Goodbye. Take care.")
            append_transcript("ai","Goodbye.")
            try: os.remove(LOOP_LOCK_F)
            except: pass
            break

        if response and not response.startswith("__"):
            append_transcript("ai", response)

        write_status("CAMERA" if read_camera_state()=="ON" else "LISTENING")

# ─────────────────────────────────────────────────────────
#  START ONCE
#  _ns.voice_started (in sys.modules) is the true guard —
#  it survives rerenders unlike session_state or lock files.
# ─────────────────────────────────────────────────────────
def start_loop_once():
    if _ns.voice_started:
        return
    _ns.voice_started = True
    threading.Thread(target=voice_loop, daemon=True).start()
    print("[BOOT] Voice thread started ✓")

# ─────────────────────────────────────────────────────────
#  BOOT — session_state ensures the cleanup block runs only
#  once per browser session (not on every rerender).
# ─────────────────────────────────────────────────────────
if "naina_booted" not in st.session_state:
    st.session_state.naina_booted = True

    # 1. Kill leftover PowerShell / SAPI from previous run
    kill_old_speech_processes()
    time.sleep(0.5)

    # 2. Wipe stale state files
    for _f in [TRANSCRIPT_F, STATUS_F, CAMERA_F, LOOP_LOCK_F]:
        try: os.remove(_f)
        except: pass

    # 3. Reset the voice_started flag so a fresh thread starts
    _ns.voice_started         = False
    _ns.vision_thread_running = False

    # 4. Start fresh voice loop
    start_loop_once()

time.sleep(0.3)

# ─────────────────────────────────────────────────────────
#  READ STATE FOR UI
# ─────────────────────────────────────────────────────────
status     = read_status()
transcript = read_transcript()
cam_state  = read_camera_state()

# ─────────────────────────────────────────────────────────
#  RENDER UI
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-title">N<span style="color:#00f090">AI</span>N</div>
  <div class="hdr-sub">voice-first · qwen2.5:0.5b + moondream vision · web search · spotify</div>
</div>
""", unsafe_allow_html=True)

hints = {
    "listening":  "Listening — speak your command…",
    "thinking":   "Thinking…",
    "speaking":   "Speaking — say STOP to interrupt",
    "stopped":    "Stopped — listening again shortly",
    "camera":     "Camera live — say CLOSE CAMERA to stop",
    "searching":  "Searching the web…",
    "boot":       "Starting up…",
    "ready":      "Ready",
    "error":      "Mic error — retrying…",
}
st.markdown(f"""
<div class="ss">
  <div class="dot dot-{status}"></div>
  <span class="slbl {status}">{status.upper()}</span>
  <span class="shint">{hints.get(status,'')}</span>
</div>
""", unsafe_allow_html=True)

if status == "speaking":
    st.markdown('<div class="osc">'+''.join(['<div class="ob"></div>']*20)+'</div>',
                unsafe_allow_html=True)
elif status in ("listening","thinking"):
    st.markdown('<div class="mw">'+''.join(['<div class="mb"></div>']*8)+'</div>',
                unsafe_allow_html=True)
elif status == "searching":
    st.markdown('<div class="sw">'+''.join(['<div class="sb"></div>']*10)+'</div>',
                unsafe_allow_html=True)
elif status == "camera":
    st.markdown('<div class="cw">'+''.join(['<div class="cb"></div>']*6)+'</div>',
                unsafe_allow_html=True)
elif status == "stopped":
    st.markdown("""
    <div class="stop-banner">
      <div class="sq"></div> STOPPED — LISTENING AGAIN SHORTLY <div class="sq"></div>
    </div>""", unsafe_allow_html=True)

if cam_state == "ON":
    st.markdown("""
    <div style="background:#0b1018;border-left:1px solid #182333;
         border-right:1px solid #182333;padding:.8rem 1.5rem .4rem;
         font-size:.6rem;color:#ff8c30;letter-spacing:.2em;text-transform:uppercase;">
      📷 Live Camera + Object Detection — say "close camera" to stop
    </div>""", unsafe_allow_html=True)

    # Grab the latest frame from shared memory
    with _ns.frame_lock:
        frame = _ns.latest_frame.copy() if _ns.latest_frame is not None else None

    if frame is not None:
        # Write to a temp JPEG on disk so Streamlit re-reads it fresh each rerender
        _cam_tmp = "nain_cam_frame.jpg"
        try:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(_cam_tmp, bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            st.image(_cam_tmp, channels="BGR", use_container_width=True)
        except Exception:
            # Fallback: display directly from numpy array
            st.image(frame, channels="RGB", use_container_width=True)
    else:
        st.markdown("""
        <div style="background:#0b1018;border:1px solid #182333;
             height:240px;display:flex;align-items:center;justify-content:center;
             color:#ff8c30;font-size:.65rem;letter-spacing:.2em;">
          📷 CAMERA WARMING UP…
        </div>""", unsafe_allow_html=True)

feed = '<div class="tx" id="txf">'
if not transcript:
    feed += '<div class="sys">── waiting for first voice command ──</div>'
else:
    for e in transcript:
        role,text,ts = e["role"],e["text"],e.get("ts","")
        if role == "user":
            feed += (f'<div class="ur"><div class="lb">YOU &nbsp; {ts}</div>'
                     f'<div class="bb">{text}</div></div>')
        elif role == "ai":
            feed += (f'<div class="ar"><div class="lb">NAIN &nbsp; {ts}</div>'
                     f'<div class="bb">{text}</div></div>')
        else:
            feed += f'<div class="sys">{text}</div>'
feed += '</div>'
feed += ('<script>(function(){var f=document.getElementById("txf");'
         'if(f)f.scrollTop=f.scrollHeight;})();</script>')
st.markdown(feed, unsafe_allow_html=True)

exchanges = len([e for e in transcript if e["role"]=="user"])
st.markdown(f"""
<div class="ft">
  <span>🎙 VOICE ONLY</span>
  <span>{exchanges} exchanges</span>
  <span>qwen2.5:0.5b · moondream · web</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="lg">
  <div class="lg-t">Voice Commands</div>
  <div class="lg-i"><span>"play [song]"</span> — YouTube</div>
  <div class="lg-i"><span>"open spotify"</span> — Spotify</div>
  <div class="lg-i"><span>"close youtube"</span> — closes YouTube tab only</div>
  <div class="lg-i"><span>"close spotify"</span> — closes Spotify only</div>
  <div class="lg-i"><span>"open camera"</span> — live webcam + Moondream</div>
  <div class="lg-i"><span>"what do you see"</span> — on-demand object scan</div>
  <div class="lg-i"><span>"close camera"</span> — stops webcam + vision</div>
  <div class="lg-i"><span>"who is Allu Arjun"</span> — web search</div>
  <div class="lg-i"><span>"today's news"</span> — web search</div>
  <div class="lg-i"><span>"stop / wait"</span> — 🔴 interrupt</div>
</div>
""", unsafe_allow_html=True)

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1000, limit=None, key="pulse")
except ImportError:
    st.markdown("""
    <div style="text-align:center;font-size:.58rem;color:#334455;margin-top:1rem;">
      Run: <code style="color:#29d9ff">pip install streamlit-autorefresh</code>
    </div>""", unsafe_allow_html=True)