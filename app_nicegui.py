
import os
import io
import json
import base64
import asyncio
import tempfile
from typing import Dict, Any, List, Tuple, Optional

from nicegui import ui, app
from PIL import Image, ImageDraw, ImageFont
import requests

# Optional Playwright for screenshots
try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

TITLE = 'Attention Optimization AI — LLM Frontend (NiceGUI)'
DESC  = 'Diagnose why CTAs and key content are missed. LLM-only analysis with strict structured outputs.'

# ---------- Helpers ----------

def screenshot_url(url: str, full_page: bool = True, width: int = 1440, height: int = 900) -> Optional[Image.Image]:
    if sync_playwright is None:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={'width': width, 'height': height}, device_scale_factor=1)
            page = context.new_page()
            page.goto(url, timeout=35000, wait_until='networkidle')
            if full_page:
                page_height = page.evaluate('() => document.body.scrollHeight')
                context.set_viewport_size({'width': width, 'height': int(page_height or height)})
            buf = page.screenshot(full_page=full_page, type='png')
            page.close(); context.close(); browser.close()
            return Image.open(io.BytesIO(buf)).convert('RGB')
    except Exception as e:
        return None

def fetch_html(the_url: str) -> str:
    try:
        r = requests.get(the_url, timeout=10)
        return r.text[:250000]
    except Exception:
        return ''

def img_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{b64}'

ATTENTION_SCHEMA = {
    'name': 'attention_report',
    'schema': {
        'type': 'object',
        'properties': {
            'image_size': {'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2},
            'saliency_grid': {
                'type':'array',
                'items': {'type':'array','items': {'type':'number','minimum':0.0,'maximum':1.0}}
            },
            'ctas': {
                'type':'array',
                'items': {
                    'type':'object',
                    'properties': {
                        'text': {'type':'string'},
                        'bbox_norm': {'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4},
                        'priority': {'type':'integer','minimum':1,'maximum':5},
                        'issues': {'type':'array','items':{'type':'string'}},
                        'estimates': {
                            'type':'object',
                            'properties': {
                                'saliency': {'type':'number','minimum':0.0,'maximum':1.0},
                                'contrast': {'type':'number','minimum':0.0,'maximum':10.0},
                                'thirds_score': {'type':'number','minimum':0.0,'maximum':1.0}
                            },
                            'required': ['saliency']
                        },
                        'suggested_copy': {'type':'string'}
                    },
                    'required': ['text','bbox_norm','priority','issues','estimates']
                }
            },
            'people': {
                'type':'array',
                'items': {
                    'type':'object',
                    'properties': {
                        'bbox_norm': {'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4},
                        'inferred_emotion': {'type':'string'},
                        'gaze_toward_cta': {'type':'boolean'}
                    },
                    'required': ['bbox_norm']
                }
            },
            'global_issues': {'type':'array','items':{'type':'string'}},
            'prioritized_suggestions': {'type':'array','items':{'type':'string'}},
            'ab_tests': {
                'type':'array',
                'items': {
                    'type':'object',
                    'properties': {
                        'hypothesis': {'type':'string'},
                        'variant_A': {'type':'string'},
                        'variant_B': {'type':'string'},
                        'primary_metric': {'type':'string'},
                        'expected_impact': {'type':'string'}
                    },
                    'required': ['hypothesis','variant_A','variant_B','primary_metric']
                }
            },
            'confidence': {'type':'number','minimum':0.0,'maximum':1.0}
        },
        'required': ['image_size','saliency_grid','ctas','prioritized_suggestions','confidence']
    },
    'strict': True
}

SYSTEM = (
    'You are a world-class CRO/UX expert. Analyze the landing page image (and optional HTML). '
    'Return JSON strictly matching the schema: saliency_grid (rows x cols), CTAs with bbox_norm 0..1 and estimates '
    '(saliency/contrast/thirds_score), any people & emotions, prioritized suggestions, AB tests, and confidence. '
    'Be precise and actionable. If unsure, produce best-effort estimates.'
)

USER_TPL = (
    'Task: Diagnose why key content/CTAs may not capture attention and propose fixes.\n\n'
    'Constraints:\n'
    '- Saliency grid size target: {rows} x {cols}\n'
    '- Bboxes are [x1,y1,x2,y2] in normalized 0..1 coordinates relative to the provided image.\n'
    '- Prioritize one primary CTA.\n'
    '- Use short, justifiable bullets.\n\n'
    'Provide only JSON via structured outputs.\n\n'
    'HTML (truncated):\n{html}\n'
)

def call_llm(image: Image.Image, html: str, rows: int, cols: int, model: str, temperature: float) -> Dict[str, Any]:
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_APIKEY')
    if not api_key or OpenAI is None:
        raise RuntimeError('OpenAI SDK or API key missing. Set OPENAI_API_KEY.')
    client = OpenAI(api_key=api_key)
    data_url = img_to_data_url(image)
    user_text = USER_TPL.format(rows=rows, cols=cols, html=html[:4000])
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {'role':'system','content':SYSTEM},
            {'role':'user','content':[
                {'type':'text','text': user_text},
                {'type':'image_url','image_url': {'url': data_url}}
            ]}
        ],
        response_format={'type':'json_schema','json_schema': ATTENTION_SCHEMA}
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        if isinstance(content, dict):
            return content
        raise

def upsample_grid_to_image(grid: List[List[float]], size: Tuple[int,int]) -> Image.Image:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        return Image.new('RGBA', size, (0,0,0,0))
    import numpy as np
    arr = np.array(grid, dtype=float)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    small = Image.fromarray((arr*255).astype('uint8'), mode='L')
    big = small.resize(size, resample=Image.BILINEAR)
    r = big
    g = Image.new('L', size, color=0)
    b = Image.new('L', size, color=0)
    a = big.point(lambda v: int(120 * (v/255)))
    red_img = Image.merge('RGBA', (r, g, b, a))
    return red_img

def denorm_box(bn: List[float], w: int, h: int) -> Tuple[int,int,int,int]:
    x1 = int(bn[0]*w); y1 = int(bn[1]*h); x2 = int(bn[2]*w); y2 = int(bn[3]*h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    return (x1,y1,x2,y2)

def overlay_boxes(base: Image.Image, boxes: List[Tuple[Tuple[int,int,int,int], str]]) -> Image.Image:
    out = base.convert('RGBA')
    draw = ImageDraw.Draw(out)
    for (x1,y1,x2,y2), label in boxes:
        draw.rectangle([x1,y1,x2,y2], outline=(0,255,0,255), width=3)
        if label:
            pad = 6
            label = label[:28]
            w = 9*len(label)+2*pad
            h = 22
            draw.rectangle([x1, max(0,y1-h), x1+w, y1], fill=(0,255,0,180))
            draw.text((x1+pad, y1-h+4), label, fill=(0,0,0,255))
    return out

def render_outputs(image: Image.Image, report: Dict[str,Any]) -> Tuple[Image.Image, Dict[str,Any]]:
    W, H = image.size
    heat = upsample_grid_to_image(report.get('saliency_grid', []), (W, H))
    composed = Image.alpha_composite(image.convert('RGBA'), heat)

    cta_boxes = []
    for cta in report.get('ctas', []):
        label = f"CTA: {cta.get('text','')[:18]}"
        box = denorm_box(cta['bbox_norm'], W, H)
        cta_boxes.append((box, label))
    composed = overlay_boxes(composed, cta_boxes)

    ppl_boxes = []
    for p in report.get('people', []):
        label = f"Person {p.get('inferred_emotion','')}"
        box = denorm_box(p['bbox_norm'], W, H)
        ppl_boxes.append((box, label))
    composed = overlay_boxes(composed, ppl_boxes)

    metrics = {
        'num_ctas': len(report.get('ctas', [])),
        'num_people': len(report.get('people', [])),
        'confidence': report.get('confidence', None),
        'avg_cta_saliency': None,
    }
    try:
        sal_vals = [cta.get('estimates',{}).get('saliency',0.0) for cta in report.get('ctas', [])]
        if sal_vals:
            metrics['avg_cta_saliency'] = round(sum(sal_vals)/len(sal_vals), 3)
    except Exception:
        pass
    return composed, metrics

# ---------- UI ----------

app.add_static_files('/static', 'static')

with ui.header().classes('items-center justify-between bg-transparent px-4 py-2'):
    with ui.row().classes('items-center gap-3'):
        ui.icon('travel_explore').classes('text-2xl')
        ui.label(TITLE).classes('text-xl font-semibold')
    with ui.row().classes('items-center gap-2'):
        ui.toggle(['Light', 'Dark'], value='Dark').bind_value_from(ui.dark_mode()).on_value_change(lambda e: ui.dark_mode().set(e.value=='Dark'))

with ui.row().classes('w-full items-stretch px-5 pt-2 gap-4'):
    # INPUT CARD
    with ui.card().classes('w-full max-w-2xl shadow-xl rounded-2xl frosted'):
        ui.label('Input').classes('text-lg font-medium')
        url_input = ui.input('Landing page URL').classes('w-full').props('clearable')
        full_page = ui.checkbox('Full-page screenshot (Playwright)').bind_value(None).set_value(True)
        grid_rows = ui.slider(min=6, max=24, value=12, step=1, on_change=None).props('label-always').classes('w-full')
        ui.label('Saliency grid rows').style('margin-top:-8px')
        grid_cols = ui.slider(min=6, max=24, value=8, step=1, on_change=None).props('label-always').classes('w-full')
        ui.label('Saliency grid cols').style('margin-top:-8px')

        with ui.row().classes('items-center gap-3 mt-2'):
            model_in = ui.input('OpenAI model').classes('w-64').set_value(os.getenv('OPENAI_MODEL','gpt-4o'))
            temp_in  = ui.slider(min=0.0, max=1.0, value=0.2, step=0.1).props('label-always')
            ui.label('Temperature')

        ui.separator()
        ui.label('Or upload an image').classes('text-sm text-gray-500')
        upload = ui.upload(label='Drop or select image', auto_upload=True, multiple=False).classes('w-full')
        ui.label(DESC).classes('text-xs text-gray-500 mt-1')
        analyze_btn = ui.button('Analyze', icon='bolt', color='blue', on_click=lambda: None).classes('self-end mt-2')

    # PREVIEW & RESULTS
    with ui.column().classes('w-full gap-4'):
        with ui.card().classes('w-full shadow-xl rounded-2xl frosted'):
            ui.label('Preview').classes('text-lg font-medium')
            preview = ui.image().classes('w-full rounded-xl border')
        with ui.card().classes('w-full shadow-xl rounded-2xl frosted'):
            ui.label('Heatmap & Annotations').classes('text-lg font-medium')
            result_img = ui.image().classes('w-full rounded-xl border')
            with ui.row().classes('gap-4 mt-2'):
                kpi1 = ui.label().classes('kpi')
                kpi2 = ui.label().classes('kpi')
                kpi3 = ui.label().classes('kpi')
                kpi4 = ui.label().classes('kpi')
            with ui.row().classes('gap-2 mt-2'):
                dl_png = ui.button('Download annotated PNG', icon='download', color='primary')
                dl_json = ui.button('Download JSON report', icon='data_object', color='primary')

        with ui.card().classes('w-full shadow-xl rounded-2xl frosted'):
            ui.label('Prioritized Suggestions').classes('text-lg font-medium')
            suggestions_box = ui.column().classes('gap-2')
            with ui.expansion('Global issues').classes('mt-2'):
                global_issues_box = ui.column().classes('gap-1')
            with ui.expansion('CTAs (parsed)').classes('mt-2'):
                cta_box = ui.column().classes('gap-1')
            with ui.expansion('A/B Tests').classes('mt-2'):
                ab_box = ui.column().classes('gap-1')

# ---------- STYLE ----------
ui.add_head_html('''
<link rel="stylesheet" href="/static/styles.css">
''')

# ---------- Logic ----------
state = {'image': None, 'report': None, 'annotated': None}

def get_uploaded_image() -> Optional[Image.Image]:
    if not upload.value:
        return None
    try:
        f = upload.value[0]
        content = f['content']
        return Image.open(io.BytesIO(content)).convert('RGB')
    except Exception:
        return None

def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='PNG')
    return buf.getvalue()

@ui.refreshable
def run_analysis():
    pass

def format_metric(label: str, value: Any) -> str:
    return f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'

def set_kpis(m: Dict[str,Any]):
    kpi1.set_text('')
    kpi2.set_text('')
    kpi3.set_text('')
    kpi4.set_text('')
    kpi1.set_text(format_metric('# CTAs', m.get('num_ctas', '—')))
    kpi2.set_text(format_metric('# People', m.get('num_people', '—')))
    kpi3.set_text(format_metric('LLM Confidence', m.get('confidence', '—')))
    kpi4.set_text(format_metric('Avg CTA Saliency', m.get('avg_cta_saliency', '—')))

def overlay(image: Image.Image, report: Dict[str,Any]) -> Tuple[Image.Image, Dict[str,Any]]:
    return render_outputs(image, report)

async def on_analyze_click():
    with ui.dialog() as d, ui.card().classes('shadow-xl rounded-2xl'):
        ui.spinner(size='lg')
        ui.label('Analyzing with LLM…').classes('text-sm text-gray-600')
    d.open()

    img = get_uploaded_image()
    html = ''
    url_val = url_input.value.strip() if url_input.value else ''
    if img is None and url_val:
        img = screenshot_url(url_val, full_page.value)
        html = fetch_html(url_val)
        if img is None:
            d.close()
            ui.notify('Failed to screenshot URL. Upload an image instead.', color='negative')
            return
    if img is None and not url_val:
        d.close()
        ui.notify('Provide a URL or upload an image', color='warning')
        return

    state['image'] = img
    preview.set_source(None)
    preview.set_source(to_png_bytes(img))

    try:
        report = call_llm(img, html, int(grid_rows.value), int(grid_cols.value), model_in.value, float(temp_in.value))
    except Exception as e:
        d.close()
        ui.notify(f'LLM call failed: {e}', color='negative')
        return

    annotated, metrics = overlay(img, report)
    state['report'] = report
    state['annotated'] = annotated

    result_img.set_source(to_png_bytes(annotated))
    set_kpis(metrics)

    suggestions_box.clear()
    for s in report.get('prioritized_suggestions', []):
        with suggestions_box:
            ui.label(f'• {s}').classes('text-sm')

    global_issues_box.clear()
    for gi in report.get('global_issues', []):
        with global_issues_box:
            ui.label(f'- {gi}').classes('text-sm')

    cta_box.clear()
    for i, cta in enumerate(report.get('ctas', []), start=1):
        est = cta.get('estimates', {})
        issues = '; '.join(cta.get('issues', [])) if cta.get('issues') else '—'
        text = cta.get('text','(no text)')
        with cta_box:
            ui.html(f'<b>{i}. {text}</b> — priority {cta.get("priority","?")}').classes('text-sm')
            ui.label(f"saliency {est.get('saliency','?')}, contrast {est.get('contrast','?')}, thirds {est.get('thirds_score','?')}").classes('text-xs text-gray-500')
            if issues != '—':
                ui.label(f"Issues: {issues}").classes('text-xs')

    ab_box.clear()
    for t in report.get('ab_tests', []):
        with ab_box:
            ui.html(f"<b>Hypothesis:</b> {t.get('hypothesis','')}").classes('text-sm')
            ui.label(f"A: {t.get('variant_A','')}").classes('text-sm')
            ui.label(f"B: {t.get('variant_B','')}").classes('text-sm')
            ui.label(f"Primary metric: {t.get('primary_metric','')} | Expected: {t.get('expected_impact','')}").classes('text-xs text-gray-500')

    def download_png(_):
        if state.get('annotated') is None:
            ui.notify('No result yet', color='warning'); return
        ui.download(to_png_bytes(state['annotated']), filename='attention_annotations_llm.png')

    def download_json(_):
        if state.get('report') is None:
            ui.notify('No report yet', color='warning'); return
        ui.download(json.dumps(state['report'], indent=2).encode('utf-8'), filename='attention_report_llm.json')

    dl_png.on('click', download_png)
    dl_json.on('click', download_json)

    d.close()
    ui.notify('Done', color='positive')

analyze_btn.on('click', lambda: asyncio.create_task(on_analyze_click()))

# Footer
with ui.footer().classes('justify-between px-4 py-2'):
    ui.label('Tip: Set OPENAI_API_KEY. For URLs, run `python -m playwright install chromium` once.').classes('text-xs text-gray-500')
    ui.link('GitHub', 'https://github.com').props('icon=code')

# Run
ui.run(title=TITLE, favicon='🌶️', reload=False)
