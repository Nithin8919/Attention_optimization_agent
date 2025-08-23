import os
import io
import json
import base64
import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, Request, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont
import requests

# Optional dependencies
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
Path("outputs").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

app = FastAPI(title="Attention Optimization AI")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Templates
templates = Jinja2Templates(directory="templates")

# OpenAI Schema for structured output
ATTENTION_SCHEMA = {
    "name": "attention_analysis",
    "schema": {
        "type": "object",
        "properties": {
            "saliency_grid": {
                "type": "array",
                "items": {
                    "type": "array", 
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "description": "Grid of attention scores"
            },
            "ctas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "bbox": {"type": "array", "items": {"type": "number"}},
                        "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                        "saliency": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "issues": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["text", "bbox", "priority", "saliency"]
                }
            },
            "suggestions": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["saliency_grid", "ctas", "suggestions", "confidence"]
    },
    "strict": True
}

class AttentionAnalyzer:
    def __init__(self):
        self.client = None
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def screenshot_url(self, url: str, full_page: bool = True) -> Optional[Image.Image]:
        """Take screenshot of URL using Playwright"""
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Playwright not available")
            return None
            
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(viewport={"width": 1440, "height": 900})
                page = await context.new_page()
                
                await page.goto(url, timeout=30000, wait_until="networkidle")
                
                if full_page:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(1000)
                
                screenshot = await page.screenshot(full_page=full_page)
                await browser.close()
                
                image = Image.open(io.BytesIO(screenshot)).convert("RGB")
                logger.info(f"Screenshot successful: {image.size}")
                return image
                
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    def encode_image(self, image: Image.Image) -> str:
        """Encode image to base64 for OpenAI"""
        # Resize if too large
        if max(image.size) > 2048:
            ratio = 2048 / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    async def analyze_attention(self, image: Image.Image, html_context: str = "", grid_size: Tuple[int, int] = (12, 8)) -> Dict[str, Any]:
        """Analyze image for attention patterns using OpenAI Vision"""
        if not self.client:
            raise HTTPException(status_code=500, detail="OpenAI not configured")

        rows, cols = grid_size
        encoded_image = self.encode_image(image)
        
        prompt = f"""
        Analyze this landing page for attention optimization. 
        
        Task:
        1. Create a {rows}x{cols} saliency grid where each cell represents attention probability (0.0-1.0)
        2. Identify all Call-to-Action buttons with their position [x1,y1,x2,y2] normalized to 0-1
        3. Rate each CTA's priority (1=highest, 5=lowest) and current saliency score
        4. Provide 5-8 actionable optimization suggestions
        5. Give confidence score for analysis
        
        Focus on: visual hierarchy, contrast, positioning, color psychology, eye-tracking patterns.
        
        HTML context: {html_context[:2000]}
        
        Return only valid JSON matching the schema.
        """

        try:
            response = await self.client.chat.completions.acreate(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                            }
                        ]
                    }
                ],
                response_format={"type": "json_schema", "json_schema": ATTENTION_SCHEMA},
                max_tokens=2000,
                temperature=0.2
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info("OpenAI analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            # Return fallback analysis
            return self._fallback_analysis(image, grid_size)

    def _fallback_analysis(self, image: Image.Image, grid_size: Tuple[int, int]) -> Dict[str, Any]:
        """Fallback analysis when OpenAI fails"""
        rows, cols = grid_size
        
        # Create basic center-weighted saliency grid
        grid = []
        for r in range(rows):
            row = []
            for c in range(cols):
                # Higher attention in center and upper areas
                y_weight = 1.0 - (r / rows) * 0.6  # Top bias
                x_weight = 1.0 - abs((c / cols) - 0.5) * 0.4  # Center bias
                score = (y_weight * x_weight) * 0.7
                row.append(round(score, 3))
            grid.append(row)
        
        return {
            "saliency_grid": grid,
            "ctas": [{
                "text": "Primary CTA (detected)",
                "bbox": [0.3, 0.4, 0.7, 0.5],
                "priority": 1,
                "saliency": 0.6,
                "issues": ["Manual detection - OpenAI unavailable"]
            }],
            "suggestions": [
                "Increase contrast of primary call-to-action",
                "Move important elements to upper-left quadrant",
                "Use larger fonts for key messages",
                "Add visual hierarchy with spacing",
                "Consider color psychology for CTAs"
            ],
            "confidence": 0.3
        }

    def create_heatmap_overlay(self, grid: List[List[float]], image_size: Tuple[int, int]) -> Image.Image:
        """Create heatmap overlay from saliency grid"""
        import numpy as np
        
        rows, cols = len(grid), len(grid[0])
        width, height = image_size
        
        # Convert to numpy array and normalize
        arr = np.array(grid, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        
        # Create small heatmap
        heatmap_small = Image.fromarray((arr * 255).astype(np.uint8), mode='L')
        
        # Resize to image size
        heatmap = heatmap_small.resize((width, height), Image.Resampling.BILINEAR)
        
        # Convert to RGBA with red coloring
        heatmap_rgba = Image.new('RGBA', (width, height))
        pixels = []
        
        for y in range(height):
            for x in range(width):
                intensity = heatmap.getpixel((x, y))
                alpha = int(intensity * 0.6)  # Semi-transparent
                pixels.append((255, 0, 0, alpha))  # Red overlay
        
        heatmap_rgba.putdata(pixels)
        return heatmap_rgba

    def draw_cta_boxes(self, image: Image.Image, ctas: List[Dict]) -> Image.Image:
        """Draw CTA bounding boxes on image"""
        img_with_boxes = image.convert('RGBA')
        draw = ImageDraw.Draw(img_with_boxes)
        width, height = image.size
        
        for i, cta in enumerate(ctas):
            x1, y1, x2, y2 = cta['bbox']
            
            # Convert normalized coords to pixels
            px1, py1 = int(x1 * width), int(y1 * height)
            px2, py2 = int(x2 * width), int(y2 * height)
            
            # Draw box
            color = (0, 255, 0, 200) if cta['priority'] <= 2 else (255, 165, 0, 200)
            draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
            
            # Draw label
            label = f"CTA {i+1}: {cta['text'][:20]}"
            draw.rectangle([px1, py1-25, px1+len(label)*8, py1], fill=color)
            draw.text((px1+5, py1-20), label, fill=(0, 0, 0, 255))
        
        return img_with_boxes

    def generate_final_image(self, original: Image.Image, analysis: Dict[str, Any]) -> Image.Image:
        """Generate final annotated image with heatmap and CTA boxes"""
        # Create heatmap overlay
        heatmap = self.create_heatmap_overlay(analysis['saliency_grid'], original.size)
        
        # Composite with original
        base = original.convert('RGBA')
        combined = Image.alpha_composite(base, heatmap)
        
        # Add CTA boxes
        final = self.draw_cta_boxes(combined, analysis['ctas'])
        
        return final.convert('RGB')


analyzer = AttentionAnalyzer()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "openai_available": OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY")),
        "playwright_available": PLAYWRIGHT_AVAILABLE
    })


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    url: str = Form(default=""),
    rows: int = Form(default=12),
    cols: int = Form(default=8),
    file: UploadFile = None
):
    """Analyze landing page"""
    start_time = time.time()
    run_id = uuid.uuid4().hex[:8]
    
    try:
        # Get image
        image = None
        html_context = ""
        
        if file and file.filename:
            content = await file.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
            logger.info(f"Image uploaded: {image.size}")
            
        elif url.strip():
            image = await analyzer.screenshot_url(url.strip())
            if image:
                # Get HTML context
                try:
                    response = requests.get(url.strip(), timeout=10)
                    html_context = response.text[:5000]  # Limit size
                except:
                    html_context = ""
            else:
                raise HTTPException(status_code=400, detail="Failed to screenshot URL")
        else:
            raise HTTPException(status_code=400, detail="Please provide URL or upload image")

        # Analyze with AI
        analysis = await analyzer.analyze_attention(
            image, 
            html_context, 
            grid_size=(rows, cols)
        )
        
        # Generate visualization
        final_image = analyzer.generate_final_image(image, analysis)
        
        # Save outputs
        image_path = f"outputs/{run_id}.png"
        json_path = f"outputs/{run_id}.json"
        
        final_image.save(image_path)
        
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Calculate metrics
        metrics = {
            "processing_time": round(time.time() - start_time, 2),
            "num_ctas": len(analysis['ctas']),
            "confidence": analysis['confidence'],
            "avg_saliency": round(sum(sum(row) for row in analysis['saliency_grid']) / (rows * cols), 3)
        }
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "run_id": run_id,
            "analysis": analysis,
            "metrics": metrics,
            "image_url": f"/outputs/{run_id}.png"
        })
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return HTMLResponse(f"<div class='error'>Analysis failed: {str(e)}</div>", status_code=500)


@app.get("/download/{run_id}/{file_type}")
async def download_file(run_id: str, file_type: str):
    """Download generated files"""
    if file_type == "png":
        file_path = f"outputs/{run_id}.png"
        if Path(file_path).exists():
            return FileResponse(file_path, filename=f"attention_analysis_{run_id}.png")
    elif file_type == "json":
        file_path = f"outputs/{run_id}.json"
        if Path(file_path).exists():
            return FileResponse(file_path, filename=f"analysis_data_{run_id}.json")
    
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "openai_available": OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY")),
        "playwright_available": PLAYWRIGHT_AVAILABLE,
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)