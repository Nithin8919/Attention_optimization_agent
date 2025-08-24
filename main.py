import os
import io
import json
import base64
import uuid
import time
import logging
import asyncio
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
from PIL import Image, ImageDraw
import requests

# Optional dependencies with better error handling
PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
    print("✅ Playwright available for URL screenshots")
except ImportError:
    print("⚠️  Playwright not available - URL screenshots disabled")
    print("   Install with: pip install playwright && playwright install chromium")

OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI configured and ready")
    else:
        print("⚠️  OpenAI SDK available but API key missing")
        print("   Set OPENAI_API_KEY in your environment")
except ImportError:
    print("⚠️  OpenAI SDK not available")
    print("   Install with: pip install openai")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Create directories
Path("outputs").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

app = FastAPI(title="Attention Optimization AI", debug=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Templates
templates = Jinja2Templates(directory="templates")

# Test if templates exist
template_files = ["index.html", "results.html"]
for template_file in template_files:
    template_path = Path("templates") / template_file
    if not template_path.exists():
        logger.error(f"❌ Missing template: {template_path}")
        print(f"❌ Template missing: {template_path}")
        print("   Please create the template files from the artifacts")

class AttentionAnalyzer:
    def __init__(self):
        self.client = None
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def screenshot_url(self, url: str, full_page: bool = True) -> Optional[Image.Image]:
        """Take screenshot of URL using Playwright"""
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Playwright not available for screenshots")
            return None
            
        try:
            logger.info(f"Taking screenshot of: {url}")
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={"width": 1440, "height": 900},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                )
                page = await context.new_page()
                
                await page.goto(url, timeout=30000, wait_until="networkidle")
                await asyncio.sleep(2)  # Let page fully load
                
                if full_page:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(1)
                
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
        max_size = 2048
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized image to: {new_size}")
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    async def analyze_attention(self, image: Image.Image, html_context: str = "", grid_size: Tuple[int, int] = (12, 8)) -> Dict[str, Any]:
        """Analyze image for attention patterns using OpenAI Vision"""
        if not self.client:
            logger.warning("OpenAI not available, using fallback analysis")
            return self._fallback_analysis(image, grid_size)

        rows, cols = grid_size
        encoded_image = self.encode_image(image)
        
        prompt = f"""Analyze this landing page for attention optimization. 

Create a {rows}x{cols} grid where each cell represents attention probability (0.0 to 1.0).
Identify all Call-to-Action buttons with their positions.
Rate each CTA's priority (1=highest, 5=lowest) and saliency score.
Provide 5-8 actionable optimization suggestions.

Focus on: visual hierarchy, contrast, positioning, color psychology.

HTML context: {html_context[:1000]}

Return ONLY valid JSON with this structure:
{{
  "saliency_grid": [[0.5, 0.3, ...], ...],
  "ctas": [{{
    "text": "Button text",
    "bbox": [x1, y1, x2, y2],
    "priority": 1,
    "saliency": 0.7,
    "issues": ["Low contrast", ...]
  }}],
  "suggestions": ["Increase CTA contrast", ...],
  "confidence": 0.85
}}"""

        try:
            logger.info("Sending request to OpenAI...")
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
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
                max_tokens=2000,
                temperature=0.2
            )
            
            content = response.choices[0].message.content
            if not content:
                logger.error("OpenAI returned empty content")
                return self._fallback_analysis(image, grid_size)
            
            logger.info(f"OpenAI raw response: {content[:200]}...")
            
            # Strip markdown code blocks if present
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.startswith('```'):
                content = content[3:]   # Remove ```
            if content.endswith('```'):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()
            
            result = json.loads(content)
            logger.info("OpenAI analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._fallback_analysis(image, grid_size)

    def _fallback_analysis(self, image: Image.Image, grid_size: Tuple[int, int]) -> Dict[str, Any]:
        """Fallback analysis when OpenAI fails"""
        rows, cols = grid_size
        logger.info("Using fallback analysis")
        
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
                "text": "Primary CTA (auto-detected)",
                "bbox": [0.3, 0.4, 0.7, 0.5],
                "priority": 1,
                "saliency": 0.6,
                "issues": ["Manual detection - AI unavailable"]
            }],
            "suggestions": [
                "Increase contrast of primary call-to-action",
                "Move important elements to upper-left quadrant",
                "Use larger fonts for key messages",
                "Add visual hierarchy with spacing",
                "Consider color psychology for CTAs",
                "Test different button colors",
                "Optimize text readability",
                "Add directional cues"
            ],
            "confidence": 0.3
        }

    def create_heatmap_overlay(self, grid: List[List[float]], image_size: Tuple[int, int]) -> Image.Image:
        """Create heatmap overlay from saliency grid"""
        import numpy as np
        
        rows, cols = len(grid), len(grid[0]) if grid else 0
        if rows == 0 or cols == 0:
            return Image.new('RGBA', image_size, (0, 0, 0, 0))
        
        width, height = image_size
        
        # Convert to numpy array and normalize
        arr = np.array(grid, dtype=np.float32)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        
        # Create small heatmap
        heatmap_small = Image.fromarray((arr * 255).astype(np.uint8), mode='L')
        
        # Resize to image size with smooth interpolation
        heatmap = heatmap_small.resize((width, height), Image.Resampling.BILINEAR)
        
        # Convert to RGBA with red coloring
        heatmap_rgba = Image.new('RGBA', (width, height))
        heatmap_data = list(heatmap.getdata())
        
        pixels = []
        for intensity in heatmap_data:
            alpha = int(intensity * 0.5)  # Semi-transparent
            pixels.append((255, 0, 0, alpha))  # Red overlay
        
        heatmap_rgba.putdata(pixels)
        return heatmap_rgba

    def draw_cta_boxes(self, image: Image.Image, ctas: List[Dict]) -> Image.Image:
        """Draw CTA bounding boxes on image"""
        if not ctas:
            return image
            
        img_with_boxes = image.convert('RGBA')
        draw = ImageDraw.Draw(img_with_boxes)
        width, height = image.size
        
        for i, cta in enumerate(ctas):
            bbox = cta.get('bbox', [0.3, 0.4, 0.7, 0.5])
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = bbox
            
            # Convert normalized coords to pixels
            px1, py1 = int(x1 * width), int(y1 * height)
            px2, py2 = int(x2 * width), int(y2 * height)
            
            # Ensure valid coordinates
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(width, px2), min(height, py2)
            
            # Skip if invalid box
            if px2 <= px1 or py2 <= py1:
                continue
            
            # Draw box
            priority = cta.get('priority', 3)
            color = (0, 255, 0, 200) if priority <= 2 else (255, 165, 0, 200)
            draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
            
            # Draw label
            text = cta.get('text', 'CTA')[:20]
            label = f"CTA {i+1}: {text}"
            
            # Background for label
            label_width = len(label) * 8
            label_height = 20
            draw.rectangle([px1, py1-label_height, px1+label_width, py1], fill=color)
            draw.text((px1+2, py1-label_height+2), label, fill=(0, 0, 0, 255))
        
        return img_with_boxes

    def generate_final_image(self, original: Image.Image, analysis: Dict[str, Any]) -> Image.Image:
        """Generate final annotated image with heatmap and CTA boxes"""
        try:
            # Create heatmap overlay
            heatmap = self.create_heatmap_overlay(
                analysis.get('saliency_grid', []), 
                original.size
            )
            
            # Composite with original
            base = original.convert('RGBA')
            combined = Image.alpha_composite(base, heatmap)
            
            # Add CTA boxes
            final = self.draw_cta_boxes(combined, analysis.get('ctas', []))
            
            return final.convert('RGB')
        except Exception as e:
            logger.error(f"Error generating final image: {e}")
            return original


# Global analyzer instance
analyzer = AttentionAnalyzer()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page"""
    try:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "openai_available": OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY")),
            "playwright_available": PLAYWRIGHT_AVAILABLE
        })
    except Exception as e:
        logger.error(f"Template error: {e}")
        # Return basic HTML if template is missing
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Attention Optimization AI</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: white; }}
                .error {{ background: #d32f2f; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .info {{ background: #1976d2; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                form {{ background: #2d2d2d; padding: 20px; border-radius: 8px; }}
                input {{ width: 100%; padding: 10px; margin: 10px 0; background: #3d3d3d; border: 1px solid #555; color: white; }}
                button {{ background: #1976d2; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }}
            </style>
        </head>
        <body>
            <h1>🎯 Attention Optimization AI</h1>
            
            <div class="error">
                <h2>Template Missing</h2>
                <p>Please create templates/index.html from the provided artifacts.</p>
                <p>Error: {str(e)}</p>
            </div>
            
            <div class="info">
                <h3>Quick Test Form</h3>
                <form action="/analyze" method="post" enctype="multipart/form-data">
                    <input type="url" name="url" placeholder="Enter URL (e.g., https://example.com)">
                    <input type="file" name="file" accept="image/*">
                    <input type="number" name="rows" value="12" min="6" max="24" style="width: 100px;">
                    <input type="number" name="cols" value="8" min="6" max="24" style="width: 100px;">
                    <button type="submit">Analyze</button>
                </form>
            </div>
            
            <div class="info">
                <h3>Status</h3>
                <p>OpenAI Available: {'✅' if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY') else '❌'}</p>
                <p>Playwright Available: {'✅' if PLAYWRIGHT_AVAILABLE else '❌'}</p>
            </div>
        </body>
        </html>
        """)


@app.post("/analyze")
async def analyze(
    request: Request,
    url: str = Form(""),
    rows: int = Form(12),
    cols: int = Form(8),
    file: UploadFile = None
):
    """Analyze landing page"""
    start_time = time.time()
    run_id = uuid.uuid4().hex[:8]
    
    logger.info(f"Analysis request {run_id}: url='{url}', file={file.filename if file else None}")
    
    try:
        # Get image
        image = None
        html_context = ""
        
        if file and file.filename and file.size > 0:
            logger.info(f"Processing uploaded file: {file.filename}")
            content = await file.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
            logger.info(f"Image loaded: {image.size}, {len(content)} bytes")
            
        elif url and url.strip():
            clean_url = url.strip()
            logger.info(f"Processing URL: {clean_url}")
            
            if not clean_url.startswith(('http://', 'https://')):
                clean_url = 'https://' + clean_url
            
            image = await analyzer.screenshot_url(clean_url)
            if image:
                # Get HTML context
                try:
                    response = requests.get(clean_url, timeout=10)
                    html_context = response.text[:5000]
                    logger.info(f"HTML context retrieved: {len(html_context)} chars")
                except Exception as e:
                    logger.warning(f"Failed to get HTML: {e}")
                    html_context = ""
            else:
                raise HTTPException(status_code=400, detail="Failed to screenshot URL")
        else:
            raise HTTPException(status_code=400, detail="Please provide URL or upload image")

        if not image:
            raise HTTPException(status_code=400, detail="No valid image obtained")

        # Analyze with AI
        logger.info("Starting AI analysis...")
        analysis = await analyzer.analyze_attention(
            image, 
            html_context, 
            grid_size=(max(6, min(24, rows)), max(6, min(24, cols)))
        )
        
        # Generate visualization
        logger.info("Generating visualization...")
        final_image = analyzer.generate_final_image(image, analysis)
        
        # Save outputs
        image_path = f"outputs/{run_id}.png"
        json_path = f"outputs/{run_id}.json"
        
        final_image.save(image_path, format="PNG", quality=95)
        logger.info(f"Saved image: {image_path}")
        
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved JSON: {json_path}")
        
        # Calculate metrics
        processing_time = round(time.time() - start_time, 2)
        grid = analysis.get('saliency_grid', [[]])
        grid_size = len(grid) * len(grid[0]) if grid and grid[0] else 0
        
        metrics = {
            "processing_time": processing_time,
            "num_ctas": len(analysis.get('ctas', [])),
            "confidence": analysis.get('confidence', 0.0),
            "avg_saliency": round(sum(sum(row) for row in grid) / max(grid_size, 1), 3) if grid else 0.0
        }
        
        logger.info(f"Analysis complete: {metrics}")
        
        try:
            return templates.TemplateResponse("results.html", {
                "request": request,
                "run_id": run_id,
                "analysis": analysis,
                "metrics": metrics,
                "image_url": f"/outputs/{run_id}.png"
            })
        except Exception as e:
            logger.error(f"Template error: {e}")
            # Return simple HTML if template fails
            return HTMLResponse(f"""
            <div style="max-width: 1000px; margin: 20px auto; background: rgba(255,255,255,0.1); border-radius: 15px; padding: 30px; color: white;">
                <div style="background: linear-gradient(135deg, #28a745, #20c997); border-radius: 10px; padding: 20px; margin-bottom: 30px; text-align: center;">
                    <h2 style="margin: 0 0 10px 0;">✅ Analysis Complete!</h2>
                    <p style="margin: 0;">Processing took {metrics['processing_time']} seconds</p>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px;">
                    <div style="background: rgba(0,123,255,0.2); padding: 20px; border-radius: 10px; text-align: center;">
                        <div style="font-size: 32px; font-weight: bold;">{metrics['num_ctas']}</div>
                        <div>CTAs Found</div>
                    </div>
                    <div style="background: rgba(108,117,125,0.2); padding: 20px; border-radius: 10px; text-align: center;">
                        <div style="font-size: 32px; font-weight: bold;">{int(metrics['confidence'] * 100)}%</div>
                        <div>AI Confidence</div>
                    </div>
                    <div style="background: rgba(255,193,7,0.2); padding: 20px; border-radius: 10px; text-align: center;">
                        <div style="font-size: 32px; font-weight: bold;">{metrics['avg_saliency']}</div>
                        <div>Avg Attention</div>
                    </div>
                    <div style="background: rgba(40,167,69,0.2); padding: 20px; border-radius: 10px; text-align: center;">
                        <div style="font-size: 32px; font-weight: bold;">{metrics['processing_time']}s</div>
                        <div>Process Time</div>
                    </div>
                </div>

                <div style="background: rgba(0,0,0,0.3); border-radius: 15px; padding: 20px; margin-bottom: 30px; text-align: center;">
                    <h3 style="margin-bottom: 20px;">🔥 Attention Heatmap Analysis</h3>
                    <img src="/outputs/{run_id}.png" alt="Analysis Result" style="max-width: 100%; border-radius: 8px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
                    <div style="margin-top: 20px;">
                        <a href="/download/{run_id}/png" style="background: #007bff; color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; margin: 0 10px;">📥 Download PNG</a>
                        <a href="/download/{run_id}/json" style="background: #28a745; color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; margin: 0 10px;">📊 Download Data</a>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 20px;">
                        <h3 style="color: #00ff88;">🎯 CTAs Detected</h3>
                        {"".join([f'<div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid #00ff88;"><strong>{cta.get("text", "CTA")}</strong><br><small>Priority {cta.get("priority", 1)} • Score: {cta.get("saliency", 0):.2f}</small></div>' for cta in analysis.get('ctas', [])])}
                    </div>
                    <div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 20px;">
                        <h3 style="color: #ffeb3b;">💡 AI Recommendations</h3>
                        {"".join([f'<div style="background: rgba(255,235,59,0.1); border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid #ffeb3b;">💡 {suggestion}</div>' for suggestion in analysis.get('suggestions', [])])}
                    </div>
                </div>

                <div style="text-align: center; margin-top: 30px;">
                    <button onclick="window.location.reload()" style="background: #007bff; border: none; color: white; padding: 12px 24px; border-radius: 8px; cursor: pointer; margin: 0 10px;">🔄 Analyze Another</button>
                    <a href="/outputs/{run_id}.png" style="background: #6f42c1; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; margin: 0 10px;">📥 View Full Image</a>
                </div>
            </div>
            """)
        
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


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


@app.get("/debug")
async def debug_page():
    """Debug form page to test form inputs"""
    try:
        return templates.TemplateResponse("debug_form.html", {"request": None})
    except Exception as e:
        # Fallback if template is missing
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Debug Form Inputs</title>
            <style>
                body {{ font-family: Arial; padding: 20px; background: #1a1a1a; color: white; }}
                .form-group {{ margin: 20px 0; }}
                input, button {{ padding: 10px; margin: 5px; }}
                button {{ background: #007bff; color: white; border: none; cursor: pointer; }}
                .test-result {{ background: #333; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>🔧 Form Input Debug Page</h1>
            <p>Template error: {str(e)}</p>
            <p>Please check if debug_form.html exists in templates folder.</p>
            <a href="/" style="color: #007bff;">← Back to Main App</a>
        </body>
        </html>
        """)

@app.get("/simple-test")
async def simple_test_page(request: Request):
    """Simple test page without external dependencies"""
    try:
        return templates.TemplateResponse("simple_test.html", {"request": request})
    except Exception as e:
        # Fallback if template is missing
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simple Form Test</title>
            <style>
                body {{ font-family: Arial; padding: 20px; background: #1a1a1a; color: white; }}
                .form-group {{ margin: 20px 0; padding: 20px; background: #2d2d2d; border-radius: 8px; }}
                input, button {{ padding: 10px; margin: 5px; border-radius: 4px; }}
                button {{ background: #007bff; color: white; border: none; cursor: pointer; }}
            </style>
        </head>
        <body>
            <h1>🧪 Simple Form Test</h1>
            <p>Template error: {str(e)}</p>
            <p>Please check if simple_test.html exists in templates folder.</p>
            <a href="/" style="color: #007bff;">← Back to Main App</a>
        </body>
        </html>
        """)

@app.post("/debug")
async def debug_form_test(test_url: str = Form(default=""), test_file: UploadFile = Form(default=None)):
    """Handle debug form submission"""
    result = {
        "url_received": test_url,
        "file_received": test_file.filename if test_file else "None",
        "file_size": test_file.size if test_file else 0
    }
    return JSONResponse(result)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "openai_available": OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY")),
        "playwright_available": PLAYWRIGHT_AVAILABLE,
        "templates_exist": all(
            Path(f"templates/{t}").exists() 
            for t in ["index.html", "results.html"]
        ),
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 ATTENTION OPTIMIZATION AI")
    print("="*60)
    print(f"OpenAI Available: {'✅' if OPENAI_AVAILABLE else '❌'}")
    print(f"API Key Set: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
    print(f"Playwright Available: {'✅' if PLAYWRIGHT_AVAILABLE else '❌'}")
    print("="*60)
    print("🌐 Starting server at: http://localhost:8080")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)