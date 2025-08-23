import os
import io
import json
import base64
import uuid
import time
import logging
import traceback
from typing import List, Tuple, Dict, Any, Optional

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image, ImageDraw
import requests
import numpy as np

# Optional: Playwright for screenshots
try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("attention-ai")

def log_exc(msg: str, req_id: str, exc: BaseException):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error("%s | req_id=%s | %s", msg, req_id, tb)

# ---------------- Utils ----------------
def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

def fetch_html(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        return r.text[:250000]
    except Exception:
        return ""

async def screenshot_url(url: str, full_page: bool = True, width: int = 1440, height: int = 900) -> Optional[Image.Image]:
    # Async version to avoid blocking the event loop
    try:
        from playwright.async_api import async_playwright
    except Exception:
        logger.warning("Playwright not available; cannot screenshot url=%s", url)
        return None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=1)
            page = await context.new_page()
            await page.goto(url, timeout=35000, wait_until="networkidle")
            if full_page:
                page_height = await page.evaluate("() => document.body.scrollHeight")
                await page.set_viewport_size({"width": width, "height": int(page_height or height)})
            buf = await page.screenshot(full_page=full_page, type="png")
            await page.close(); await context.close(); await browser.close()
            img = Image.open(io.BytesIO(buf)).convert("RGB")
            logger.info("Screenshot ok | url=%s | size=%s", url, img.size)
            return img
    except Exception:
        logger.exception("Screenshot failed for url=%s", url)
        return None

def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def maybe_downscale(img: Image.Image, max_side: int = 2048) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    if w >= h:
        new_w = max_side
        new_h = int(h * (max_side / w))
    else:
        new_h = max_side
        new_w = int(w * (max_side / h))
    return img.resize((new_w, new_h), Image.LANCZOS)

# ---------------- Structured Outputs schema (strict) ----------------
ATTENTION_JSON_SCHEMA = {
    "name": "attention_report",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "image_size": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            "saliency_grid": {"type": "array", "items": {"type": "array", "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}}},
            "ctas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "bbox_norm": {"type": "array", "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "minItems": 4, "maxItems": 4},
                        "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                        "issues": {"type": "array", "items": {"type": "string"}},
                        "estimates": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "saliency": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "contrast": {"type": "number", "minimum": 0.0, "maximum": 10.0},
                                "thirds_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            },
                            "required": ["saliency", "contrast", "thirds_score"]
                        },
                        "suggested_copy": {"type": "string"}
                    },
                    "required": ["text", "bbox_norm", "priority", "issues", "estimates", "suggested_copy"]
                }
            },
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "bbox_norm": {"type": "array", "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "minItems": 4, "maxItems": 4},
                        "inferred_emotion": {"type": "string"},
                        "gaze_toward_cta": {"type": "boolean"}
                    },
                    "required": ["bbox_norm", "inferred_emotion", "gaze_toward_cta"]
                }
            },
            "global_issues": {"type": "array", "items": {"type": "string"}},
            "prioritized_suggestions": {"type": "array", "items": {"type": "string"}},
            "ab_tests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "hypothesis": {"type": "string"},
                        "variant_A": {"type": "string"},
                        "variant_B": {"type": "string"},
                        "primary_metric": {"type": "string"},
                        "expected_impact": {"type": "string"}
                    },
                    "required": ["hypothesis", "variant_A", "variant_B", "primary_metric", "expected_impact"]
                }
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["image_size", "saliency_grid", "ctas", "people", "global_issues", "prioritized_suggestions", "ab_tests", "confidence"]
    }
}

SYSTEM = (
    "You are a world-class CRO/UX expert. Analyze the landing page image (and optional HTML). "
    "Return JSON strictly matching the schema: saliency_grid (rows x cols), CTAs with bbox_norm 0..1 and estimates "
    "(saliency/contrast/thirds_score), any people & emotions, prioritized suggestions, AB tests, and confidence. "
    "Be precise and actionable. If unsure, produce best-effort estimates."
)

USER_TPL = (
    "Task: Diagnose why key content/CTAs may not capture attention and propose fixes.\n\n"
    "Constraints:\n"
    "- Saliency grid size target: {rows} x {cols}\n"
    "- Bboxes are [x1,y1,x2,y2] in normalized 0..1 coordinates relative to the provided image.\n"
    "- Prioritize one primary CTA.\n"
    "- Use short, justifiable bullets.\n\n"
    "Provide only JSON via structured outputs.\n\n"
    "HTML (truncated):\n{html}\n"
)

def call_llm(image: Image.Image, html: str, rows: int, cols: int, model: str, temperature: float, req_id: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing.")
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed (pip install openai).")

    client = OpenAI(api_key=api_key)
    send_img = maybe_downscale(image, max_side=int(os.getenv("MAX_IMAGE_SIDE", "2048")))
    user_text = USER_TPL.format(rows=rows, cols=cols, html=html[:4000])

    logger.info(
        "LLM call start | req_id=%s | model=%s | temp=%.2f | rows=%d | cols=%d | html_len=%d | img_size=%s -> send_size=%s",
        req_id, model, temperature, rows, cols, len(html), image.size, send_img.size
    )

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(send_img)}}
                ]}
            ],
            response_format={"type": "json_schema", "json_schema": ATTENTION_JSON_SCHEMA}
        )
    except Exception as e:
        log_exc("OpenAI request failed", req_id, e)
        raise

    dt = (time.perf_counter() - t0) * 1000
    logger.info("LLM call end | req_id=%s | status=ok | latency_ms=%.1f", req_id, dt)

    if not resp.choices:
        raise RuntimeError("OpenAI returned no choices.")
    content = resp.choices[0].message.content or ""
    if not content:
        raise RuntimeError("OpenAI returned empty content.")

    if os.getenv("DEBUG_SAVE_IO"):
        with open(f"outputs/{req_id}_raw.txt", "w", encoding="utf-8") as f:
            f.write(content)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed; trying fenced extraction | req_id=%s | err=%s", req_id, str(e))
        import re
        m = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        log_exc("Failed to parse JSON response", req_id, e)
        raise RuntimeError(f"Failed to parse JSON response: {e}")

# ---------------- Saliency helpers ----------------
def _to_np(grid):
    arr = np.array(grid, dtype=float)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-6)

def mean_saliency_in_box(arr: np.ndarray, box_norm: List[float]) -> float:
    R, C = arr.shape
    x1, y1, x2, y2 = box_norm
    r1 = int(np.clip(y1 * R, 0, R - 1)); r2 = int(np.clip(np.ceil(y2 * R), 1, R))
    c1 = int(np.clip(x1 * C, 0, C - 1)); c2 = int(np.clip(np.ceil(x2 * C), 1, C))
    patch = arr[r1:r2, c1:c2]
    return float(patch.mean()) if patch.size else 0.0

def _gaussian_mask_from_box(box_norm, shape, sigma_cells=2.0):
    R, C = shape
    x1,y1,x2,y2 = box_norm
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    rr, cc = np.meshgrid(np.linspace(0,1,R), np.linspace(0,1,C), indexing="ij")
    d2 = (rr - cy)**2 + (cc - cx)**2
    box_w = max(x2-x1, 1e-3); box_h = max(y2-y1, 1e-3)
    s = sigma_cells / max(R, C)
    s = s + 0.5 * (box_w + box_h) * 0.5
    g = np.exp(-d2 / (2*s*s))
    return g / (g.max() + 1e-6)

def _viewport_bias(shape, k=0.9):
    R, C = shape
    rr = np.linspace(0, 1, R)[:, None]
    bias = np.power(k, rr * 10)
    return (bias - bias.min()) / (bias.max() - bias.min() + 1e-6)

def _gaze_bonus(shape, people, toward=1.15, away=0.92):
    R, C = shape
    if not people:
        return np.ones((R, C))
    toward_ratio = np.mean([1.0 if p.get("gaze_toward_cta") else 0.0 for p in people])
    factor = (toward if toward_ratio >= 0.5 else away)
    return np.full((R, C), factor, dtype=float)

def goal_weighted_grid(grid, ctas, people=None,
                       w_base=1.0, w_cta=0.9, w_view=0.6):
    A = _to_np(grid)
    R, C = A.shape

    if ctas:
        bumps = []
        for c in ctas:
            g = _gaussian_mask_from_box(c["bbox_norm"], (R, C))
            pr = float(c.get("priority", 3))
            bumps.append(g * (0.5 + 0.5 * pr / 5.0))
        G = np.clip(np.sum(bumps, axis=0), 0, None)
        G = G / (G.max() + 1e-6)
    else:
        G = np.zeros_like(A)

    V = _viewport_bias((R, C))
    Z = _gaze_bonus((R, C), people or [])

    H = (w_base * A + w_cta * G + w_view * V) * Z
    H = (H - H.min()) / (H.max() - H.min() + 1e-6)
    return H

# ---------------- Rendering ----------------
def upsample_grid_overlay(grid, size):
    from PIL import Image
    rows = len(grid); cols = len(grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        return Image.new("RGBA", size, (0,0,0,0))
    arr = np.array(grid, dtype=float)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    small = Image.fromarray((arr * 255).astype("uint8"), mode="L")
    big = small.resize(size, resample=Image.BILINEAR)
    r = big; g = Image.new("L", size, color=0); b = Image.new("L", size, color=0)
    a = big.point(lambda v: int(120 * (v/255)))
    return Image.merge("RGBA", (r, g, b, a))

def colorize_spectrum(img_gray_rgba):
    from PIL import Image
    gray = img_gray_rgba.split()[0]
    arr = np.array(gray, dtype=np.float32) / 255.0
    stops = np.array([
        [0.0,   0,   0,128],
        [0.33,  0, 128,255],
        [0.66,255,255,  0],
        [1.0, 255,  0,  0]
    ], dtype=np.float32)
    out = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        out[:,:,i] = np.interp(arr, stops[:,0], stops[:,i+1]).astype(np.uint8)
    a = img_gray_rgba.split()[3]
    return Image.merge("RGBA", (Image.fromarray(out[:,:,0]),
                                Image.fromarray(out[:,:,1]),
                                Image.fromarray(out[:,:,2]),
                                a))

def denorm_box(bn: List[float], w: int, h: int) -> Tuple[int,int,int,int]:
    x1 = int(bn[0]*w); y1 = int(bn[1]*h); x2 = int(bn[2]*w); y2 = int(bn[3]*h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    return (x1,y1,x2,y2)

def overlay_boxes(base: Image.Image, boxes: List[Tuple[Tuple[int,int,int,int], str]]) -> Image.Image:
    out = base.convert("RGBA")
    draw = ImageDraw.Draw(out)
    for (x1,y1,x2,y2), label in boxes:
        draw.rectangle([x1,y1,x2,y2], outline=(0,255,0,255), width=3)
        if label:
            pad = 6
            label = label[:28]
            w = 9*len(label)+2*pad; h = 22
            draw.rectangle([x1, max(0,y1-h), x1+w, y1], fill=(0,255,0,180))
            draw.text((x1+pad, y1-h+4), label, fill=(0,0,0,255))
    return out

def compose_with_boxes(base, overlay, report):
    out = Image.alpha_composite(base.convert("RGBA"), overlay)
    W, H = base.size
    cta_boxes = [(denorm_box(c["bbox_norm"], W, H), f"CTA: {c.get('text','')[:18]}") for c in report.get("ctas", [])]
    ppl_boxes = [(denorm_box(p["bbox_norm"], W, H), f"Person {p.get('inferred_emotion','')}") for p in report.get("people", [])]
    out = overlay_boxes(out, cta_boxes)
    out = overlay_boxes(out, ppl_boxes)
    return out

def render_outputs(image: Image.Image, report: Dict[str,Any]) -> Tuple[Dict[str,str], Dict[str,Any]]:
    W, H = image.size

    # 1) Vanilla heatmap
    overlay_vanilla = upsample_grid_overlay(report["saliency_grid"], (W, H))

    # 2) Goal-weighted heatmap
    gw = goal_weighted_grid(
        grid=report["saliency_grid"],
        ctas=report.get("ctas", []),
        people=report.get("people", []),
        w_base=1.0, w_cta=0.9, w_view=0.6
    )
    overlay_goal = upsample_grid_overlay(gw.tolist(), (W, H))
    overlay_goal_spectrum = colorize_spectrum(overlay_goal)

    img_vanilla = compose_with_boxes(image, overlay_vanilla, report)
    img_goal = compose_with_boxes(image, overlay_goal_spectrum, report)

    run_id = uuid.uuid4().hex[:8]
    paths = {
        "vanilla_png": f"outputs/{run_id}_vanilla.png",
        "goal_png": f"outputs/{run_id}_goal.png",
        "json": f"outputs/{run_id}.json",
        "run_id": run_id
    }
    img_vanilla.convert("RGB").save(paths["vanilla_png"], format="PNG")
    img_goal.convert("RGB").save(paths["goal_png"], format="PNG")
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Metrics
    arr_base = _to_np(report["saliency_grid"])
    page_mean = float(arr_base.mean())
    cta_sals = [mean_saliency_in_box(arr_base, c["bbox_norm"]) for c in report.get("ctas", [])] or [0.0]
    avg_cta_sal = float(np.mean(cta_sals))
    primary_idx = int(np.argmax(cta_sals)) if cta_sals else -1
    dominance = round((cta_sals[primary_idx] / (sum(cta_sals) + 1e-6)), 3) if cta_sals else 0.0
    primary_text = report.get("ctas", [{}])[primary_idx].get("text") if primary_idx >= 0 else None

    metrics = {
        "num_ctas": len(report.get("ctas", [])),
        "num_people": len(report.get("people", [])),
        "confidence": report.get("confidence", None),
        "avg_cta_saliency": round(avg_cta_sal, 3),
        "page_mean": round(page_mean, 3),
        "primary_cta_text": primary_text,
        "primary_cta_dominance": dominance
    }
    return paths, metrics

# ---------------- FastAPI App ----------------
ensure_dirs()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
templates = Jinja2Templates(directory="templates")

@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    req_id = uuid.uuid4().hex[:8]
    request.state.req_id = req_id
    t0 = time.perf_counter()
    logger.info("REQ START | id=%s | %s %s", req_id, request.method, request.url.path)
    try:
        resp = await call_next(request)
        return resp
    finally:
        dt = (time.perf_counter() - t0) * 1000
        logger.info("REQ END   | id=%s | duration_ms=%.1f", req_id, dt)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    logger.info("Startup | OpenAI SDK=%s | Playwright=%s | API key present=%s | MODEL_DEFAULT=%s",
                "yes" if OpenAI else "no",
                "yes" if sync_playwright else "no",
                "yes" if key else "no",
                os.getenv("OPENAI_MODEL", "gpt-4o"))
    return templates.TemplateResponse("index.html", {
        "request": request,
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o"),
    })

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request,
                  url: str = Form(default=""),
                  rows: int = Form(default=12),
                  cols: int = Form(default=8),
                  model: str = Form(default=os.getenv("OPENAI_MODEL","gpt-4o")),
                  temperature: float = Form(default=0.2),
                  fullpage: str = Form(default="true"),
                  file: UploadFile = None):

    req_id = getattr(request.state, "req_id", uuid.uuid4().hex[:8])
    logger.info("Analyze start | id=%s | url=%s | file=%s | rows=%s | cols=%s | model=%s | temp=%.2f",
                req_id, url or "-", file.filename if file else "-", rows, cols, model, temperature)

    html = ""
    image = None
    try:
        if file and file.filename:
            content = await file.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
            logger.info("Image uploaded | id=%s | size=%s | bytes=%d", req_id, image.size, len(content))
        elif url:
            image = await screenshot_url(url, full_page=(fullpage == "true"))
            html = fetch_html(url)
            if image is None:
                logger.warning("Screenshot failed | id=%s | url=%s", req_id, url)
                return HTMLResponse('<div class="error">Failed to screenshot URL. Upload an image instead.</div>', status_code=400)
            logger.info("URL processed | id=%s | img_size=%s | html_len=%d", req_id, image.size, len(html))
        else:
            logger.warning("No input provided | id=%s", req_id)
            return HTMLResponse('<div class="error">Provide a URL or upload an image.</div>', status_code=400)
    except Exception as e:
        log_exc("Image acquisition failed", req_id, e)
        return HTMLResponse(f'<div class="error">Invalid image or URL: {str(e)}</div>', status_code=400)

    try:
        report = call_llm(image, html, int(rows), int(cols), model, float(temperature), req_id)
        logger.info("LLM report ok | id=%s | keys=%s", req_id, list(report.keys()))
    except Exception as e:
        log_exc("LLM call failed", req_id, e)
        return HTMLResponse(f'<div class="error">LLM error (req {req_id}). Check server logs.</div>', status_code=500)

    # Ensure presence of keys (for template safety)
    for key in ["ctas","people","global_issues","prioritized_suggestions","ab_tests"]:
        report.setdefault(key, [])

    try:
        paths, metrics = render_outputs(image, report)
        run_id = paths["run_id"]
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "run_id": run_id,
                "metrics": metrics,
                "report": report,
                "vanilla_url": f"/{paths['vanilla_png']}",
                "goal_url": f"/{paths['goal_png']}",
            }
        )
    except Exception as e:
        log_exc("Render/save failed", req_id, e)
        return HTMLResponse(f'<div class="error">Failed to process results (req {req_id}).</div>', status_code=500)

# Optional: file endpoints
@app.get('/download/png/{run_id}')
def download_png(run_id: str):
    path1 = f'outputs/{run_id}_vanilla.png'
    path2 = f'outputs/{run_id}_goal.png'
    if os.path.exists(path1):
        return FileResponse(path1, filename='heatmap_vanilla.png', media_type='image/png')
    if os.path.exists(path2):
        return FileResponse(path2, filename='heatmap_goal.png', media_type='image/png')
    return JSONResponse({'error':'not found'}, status_code=404)

@app.get('/download/json/{run_id}')
def download_json(run_id: str):
    path = f'outputs/{run_id}.json'
    if not os.path.exists(path):
        return JSONResponse({'error':'not found'}, status_code=404)
    return FileResponse(path, filename='attention_report_llm.json', media_type='application/json')
