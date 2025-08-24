#!/usr/bin/env python3
"""
🎯 Attention Optimization AI - Clean Main Application
Professional CRO & UX Analysis powered by OpenAI Vision
"""

import os
import io
import json
import base64
import uuid
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Core dependencies
from fastapi import FastAPI, Request, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import requests
import numpy as np
from scipy.ndimage import gaussian_filter

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

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8080"))
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
    
    # Paths
    BASE_DIR = Path(__file__).parent
    OUTPUTS_DIR = BASE_DIR / "outputs"
    STATIC_DIR = BASE_DIR / "static"
    TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure directories exist
for dir_path in [Config.OUTPUTS_DIR, Config.STATIC_DIR, Config.TEMPLATES_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("attention-ai")

# ============================================================================
# AI ANALYSIS SCHEMA & PROMPTS - ENHANCED FOR BEFORE/AFTER COMPARISON
# ============================================================================

ANALYSIS_SCHEMA = {
    "name": "enhanced_attention_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "current_analysis": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "saliency_grid": {
                        "type": "array",
                        "items": {
                            "type": "array", 
                            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        }
                    },
                    "attention_summary": {"type": "string"},
                    "primary_focus_areas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "area": {"type": "string"},
                                "position": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                                "attention_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            },
                            "required": ["area", "position", "attention_score"]
                        }
                    }
                },
                "required": ["saliency_grid", "attention_summary", "primary_focus_areas"]
            },
            "optimized_analysis": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "saliency_grid": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        }
                    },
                    "attention_summary": {"type": "string"},
                    "optimized_focus_areas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "area": {"type": "string"},
                                "position": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                                "attention_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "improvement_reason": {"type": "string"}
                            },
                            "required": ["area", "position", "attention_score", "improvement_reason"]
                        }
                    }
                },
                "required": ["saliency_grid", "attention_summary", "optimized_focus_areas"]
            },
            "ctas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "current_position": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                        "optimized_position": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                        "current_saliency": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "optimized_saliency": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                        "issues": {"type": "array", "items": {"type": "string"}},
                        "improvements": {"type": "array", "items": {"type": "string"}},
                        "position_change_reason": {"type": "string"}
                    },
                    "required": ["text", "current_position", "optimized_position", "current_saliency", "optimized_saliency", "priority", "issues", "improvements", "position_change_reason"]
                }
            },
            "attention_flow_comparison": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "current_flow": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "primary_entry": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                            "flow_pattern": {"type": "string"},
                            "effectiveness_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["primary_entry", "flow_pattern", "effectiveness_score"]
                    },
                    "optimized_flow": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "primary_entry": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                            "flow_pattern": {"type": "string"},
                            "effectiveness_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["primary_entry", "flow_pattern", "effectiveness_score"]
                    }
                },
                "required": ["current_flow", "optimized_flow"]
            },
            "improvement_suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "category": {"type": "string"},
                        "current_issue": {"type": "string"},
                        "suggested_change": {"type": "string"},
                        "expected_impact": {"type": "string"},
                        "implementation_effort": {"type": "string"}
                    },
                    "required": ["category", "current_issue", "suggested_change", "expected_impact", "implementation_effort"]
                }
            },
            "overall_scores": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "current_score": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                    "potential_score": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                    "improvement_potential": {"type": "number", "minimum": 0.0, "maximum": 100.0}
                },
                "required": ["current_score", "potential_score", "improvement_potential"]
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["current_analysis", "optimized_analysis", "ctas", "attention_flow_comparison", "improvement_suggestions", "overall_scores", "confidence"]
    }
}

ENHANCED_SYSTEM_PROMPT = """You are a world-class CRO expert analyzing landing pages for attention optimization.

Your task: Provide a detailed before/after comparison showing current attention patterns vs. optimized attention patterns.

For CURRENT ANALYSIS:
1. Analyze the actual visual attention patterns in the image
2. Create a 16x12 saliency grid showing where attention currently goes
3. Identify what's actually drawing attention (both good and bad elements)
4. Note the current attention flow and primary focus areas

For OPTIMIZED ANALYSIS:
1. Design an improved attention pattern that maximizes conversion
2. Create a 16x12 optimized saliency grid showing where attention SHOULD go
3. Prioritize key elements (primary CTA, value proposition, social proof)
4. Design optimal attention flow patterns (F-pattern, Z-pattern, etc.)

For CTA COMPARISON:
- Show current vs. optimized positions for each CTA
- Explain WHY each position change would improve performance
- Calculate current vs. potential saliency scores

For SUGGESTIONS:
- Categorize improvements (Layout, Typography, Colors, Positioning, etc.)
- Explain current issues and specific solutions
- Estimate implementation effort and expected impact

Focus on creating a clear before/after visualization that shows the transformation from current to optimized attention patterns."""

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 data URL"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64_string = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{b64_string}"

def downscale_image(image: Image.Image, max_size: int = 2048) -> Image.Image:
    """Downscale image if too large"""
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    
    if width >= height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.LANCZOS)

# ============================================================================
# OPENAI INTEGRATION
# ============================================================================

async def analyze_with_openai(image: Image.Image, html: str = "") -> Dict[str, Any]:
    """Analyze image with OpenAI Vision API"""
    if not Config.OPENAI_API_KEY or not OPENAI_AVAILABLE:
        raise HTTPException(500, "OpenAI not configured. Set OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    # Prepare image
    processed_image = downscale_image(image, Config.MAX_IMAGE_SIZE)
    image_data_url = image_to_base64(processed_image)
    
    # Create enhanced prompt
    user_prompt = f"""Analyze this landing page for attention optimization and conversion potential.

Focus on:
- Visual hierarchy and attention flow
- CTA visibility and effectiveness  
- Color psychology and contrast
- Typography and readability
- Layout and whitespace usage
- Mobile responsiveness indicators

HTML context: {html[:3000]}

Provide detailed, actionable insights for improving conversions."""
    
    logger.info(f"Calling OpenAI API with enhanced analysis prompt")
    
    try:
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": ENHANCED_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]}
            ],
            response_format={"type": "json_schema", "json_schema": ANALYSIS_SCHEMA}
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
        
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(500, f"AI analysis failed: {str(e)}")

# ============================================================================
# IMAGE PROCESSING & VISUALIZATION
# ============================================================================

def create_professional_heatmap(saliency_grid: List[List[float]], image_size: Tuple[int, int]) -> Image.Image:
    """Create a professional-looking heatmap with smooth gradients"""
    rows, cols = len(saliency_grid), len(saliency_grid[0]) if saliency_grid else 0
    if rows == 0 or cols == 0:
        return Image.new("RGBA", image_size, (0, 0, 0, 0))
    
    # Convert to numpy array and normalize
    arr = np.array(saliency_grid, dtype=float)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    
    # Apply Gaussian smoothing for better visual appeal
    smoothed = gaussian_filter(arr, sigma=0.8)
    
    # Create colormap (cool to hot)
    def apply_heatmap_colormap(value):
        """Apply a beautiful blue-to-red colormap"""
        if value < 0.2:
            # Cool blue for low attention
            return (int(50 + value * 400), int(100 + value * 300), 255, int(120 * value))
        elif value < 0.5:
            # Green transition
            t = (value - 0.2) / 0.3
            return (int(150 + t * 105), 200, int(255 - t * 155), int(120 + t * 60))
        elif value < 0.8:
            # Yellow to orange
            t = (value - 0.5) / 0.3
            return (255, int(200 - t * 100), int(100 - t * 100), int(180 + t * 40))
        else:
            # Hot red for high attention
            t = (value - 0.8) / 0.2
            return (255, int(100 - t * 50), int(50 - t * 50), int(220 + t * 35))
    
    # Apply colormap
    colored_array = np.zeros((rows, cols, 4), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            r, g, b, a = apply_heatmap_colormap(smoothed[i, j])
            colored_array[i, j] = [r, g, b, a]
    
    # Create image and resize with high-quality resampling
    heatmap_small = Image.fromarray(colored_array, 'RGBA')
    heatmap = heatmap_small.resize(image_size, Image.LANCZOS)
    
    # Apply additional blur for smoothness
    return heatmap.filter(ImageFilter.GaussianBlur(radius=1))

def draw_enhanced_cta_boxes(image: Image.Image, ctas: List[Dict]) -> Image.Image:
    """Draw enhanced CTA bounding boxes with better styling"""
    img_with_boxes = image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    width, height = image.size
    
    # Priority color scheme
    priority_colors = {
        1: (255, 59, 48),   # Red - Critical
        2: (255, 149, 0),   # Orange - High  
        3: (52, 199, 89),   # Green - Medium
        4: (0, 122, 255),   # Blue - Low
        5: (142, 142, 147)  # Gray - Minimal
    }
    
    for i, cta in enumerate(ctas):
        bbox = cta['bbox']  # [x1, y1, x2, y2] normalized
        
        # Convert to pixel coordinates
        x1, y1, x2, y2 = [
            int(bbox[0] * width),
            int(bbox[1] * height), 
            int(bbox[2] * width),
            int(bbox[3] * height)
        ]
        
        priority = cta.get('priority', 3)
        base_color = priority_colors.get(priority, priority_colors[3])
        
        # Draw enhanced box with gradient effect
        # Outer glow
        for thickness in range(5, 0, -1):
            alpha = int(60 / thickness)
            glow_color = base_color + (alpha,)
            draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness], 
                         outline=glow_color, width=1)
        
        # Main border
        draw.rectangle([x1, y1, x2, y2], outline=base_color + (255,), width=3)
        
        # Corner indicators
        corner_size = 8
        for corner in [(x1, y1), (x2-corner_size, y1), (x1, y2-corner_size), (x2-corner_size, y2-corner_size)]:
            draw.rectangle([corner[0], corner[1], corner[0]+corner_size, corner[1]+corner_size],
                         fill=base_color + (200,))
        
        # Enhanced label
        text = cta['text'][:25] + ('...' if len(cta['text']) > 25 else '')
        label = f"CTA #{i+1}: {text}"
        
        # Calculate label dimensions
        label_width = len(label) * 8 + 16
        label_height = 28
        
        # Position label (avoid overlaps)
        label_y = y1 - label_height - 5 if y1 > label_height + 10 else y2 + 5
        label_x = min(x1, width - label_width)
        
        # Draw label background
        label_bg_color = base_color + (220,)
        draw.rectangle([label_x, label_y, label_x + label_width, label_y + label_height],
                      fill=label_bg_color)
        
        # Priority indicator
        priority_circle_x = label_x + 8
        priority_circle_y = label_y + 8
        draw.ellipse([priority_circle_x, priority_circle_y, priority_circle_x + 12, priority_circle_y + 12],
                    fill=(255, 255, 255, 255))
        
        # Priority text
        draw.text((priority_circle_x + 3, priority_circle_y + 1), str(priority), 
                 fill=(0, 0, 0, 255))
        
        # Label text
        draw.text((label_x + 24, label_y + 6), text, fill=(255, 255, 255, 255))
    
    # Composite the overlay
    return Image.alpha_composite(img_with_boxes, overlay)

def add_attention_flow_indicators(image: Image.Image, attention_flow: Dict) -> Image.Image:
    """Add visual indicators for attention flow"""
    img_with_flow = image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    width, height = image.size
    
    # Primary focus point
    primary = attention_flow['primary_focus']
    px, py = int(primary[0] * width), int(primary[1] * height)
    
    # Draw primary attention indicator (pulsing circle)
    for radius in range(30, 10, -5):
        alpha = int(150 * (30 - radius) / 20)
        draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                    outline=(255, 215, 0, alpha), width=2)
    
    # Primary label
    draw.text((px - 30, py - 50), "PRIMARY FOCUS", fill=(255, 215, 0, 255))
    
    # Secondary focus point
    secondary = attention_flow['secondary_focus']
    sx, sy = int(secondary[0] * width), int(secondary[1] * height)
    
    # Draw secondary attention indicator
    for radius in range(20, 5, -3):
        alpha = int(120 * (20 - radius) / 15)
        draw.ellipse([sx - radius, sy - radius, sx + radius, sy + radius],
                    outline=(100, 200, 255, alpha), width=2)
    
    # Secondary label
    draw.text((sx - 40, sy - 40), "SECONDARY FOCUS", fill=(100, 200, 255, 255))
    
    # Flow connection line
    draw.line([px, py, sx, sy], fill=(200, 200, 200, 150), width=3)
    
    return Image.alpha_composite(img_with_flow, overlay)

def create_comparison_visualization(image: Image.Image, analysis: Dict[str, Any]) -> Tuple[Image.Image, Image.Image, Dict]:
    """Create side-by-side comparison of current vs optimized attention patterns"""
    
    # Extract current and optimized data
    current = analysis.get('current_analysis', {})
    optimized = analysis.get('optimized_analysis', {})
    
    # Create current attention heatmap
    current_heatmap = create_professional_heatmap(
        current.get('saliency_grid', []), image.size
    )
    
    # Create optimized attention heatmap  
    optimized_heatmap = create_professional_heatmap(
        optimized.get('saliency_grid', []), image.size
    )
    
    # Composite current image
    base_image = image.convert('RGBA')
    current_image = Image.alpha_composite(base_image, current_heatmap)
    
    # Composite optimized image
    optimized_image = Image.alpha_composite(base_image.copy(), optimized_heatmap)
    
    # Add CTA annotations for current positions
    current_image = draw_cta_comparison_boxes(current_image, analysis.get('ctas', []), 'current')
    
    # Add CTA annotations for optimized positions  
    optimized_image = draw_cta_comparison_boxes(optimized_image, analysis.get('ctas', []), 'optimized')
    
    # Add attention flow indicators
    current_image = add_flow_indicators(current_image, analysis.get('attention_flow_comparison', {}).get('current_flow', {}))
    optimized_image = add_flow_indicators(optimized_image, analysis.get('attention_flow_comparison', {}).get('optimized_flow', {}))
    
    # Add labels
    current_image = add_comparison_label(current_image, "CURRENT ATTENTION", (255, 100, 100))
    optimized_image = add_comparison_label(optimized_image, "OPTIMIZED ATTENTION", (100, 255, 100))
    
    # Calculate enhanced metrics
    ctas = analysis.get('ctas', [])
    current_scores = analysis.get('overall_scores', {})
    
    metrics = {
        'num_ctas': len(ctas),
        'current_score': current_scores.get('current_score', 0),
        'potential_score': current_scores.get('potential_score', 0),
        'improvement_potential': current_scores.get('improvement_potential', 0),
        'confidence': analysis.get('confidence', 0.0),
        'current_flow_score': analysis.get('attention_flow_comparison', {}).get('current_flow', {}).get('effectiveness_score', 0),
        'optimized_flow_score': analysis.get('attention_flow_comparison', {}).get('optimized_flow', {}).get('effectiveness_score', 0),
        'avg_current_saliency': np.mean([cta['current_saliency'] for cta in ctas]) if ctas else 0.0,
        'avg_optimized_saliency': np.mean([cta['optimized_saliency'] for cta in ctas]) if ctas else 0.0,
    }
    
    return current_image, optimized_image, metrics

def draw_cta_comparison_boxes(image: Image.Image, ctas: List[Dict], version: str) -> Image.Image:
    """Draw CTA boxes for current or optimized positions"""
    img_with_boxes = image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    width, height = image.size
    
    # Different colors for current vs optimized
    colors = {
        'current': {
            1: (255, 59, 48, 200),   # Red
            2: (255, 149, 0, 200),   # Orange  
            3: (255, 204, 0, 200),   # Yellow
            4: (0, 122, 255, 200),   # Blue
            5: (142, 142, 147, 200)  # Gray
        },
        'optimized': {
            1: (52, 199, 89, 200),   # Green
            2: (48, 209, 88, 200),   # Light Green
            3: (0, 245, 128, 200),   # Bright Green
            4: (102, 217, 173, 200), # Mint
            5: (162, 162, 167, 200)  # Light Gray
        }
    }
    
    for i, cta in enumerate(ctas):
        # Choose position based on version
        if version == 'current':
            bbox = cta.get('current_position', [0, 0, 0.1, 0.1])
            saliency = cta.get('current_saliency', 0)
        else:
            bbox = cta.get('optimized_position', [0, 0, 0.1, 0.1])
            saliency = cta.get('optimized_saliency', 0)
        
        # Convert to pixel coordinates
        x1, y1, x2, y2 = [
            int(bbox[0] * width),
            int(bbox[1] * height), 
            int(bbox[2] * width),
            int(bbox[3] * height)
        ]
        
        priority = cta.get('priority', 3)
        color_set = colors[version]
        base_color = color_set.get(priority, color_set[3])
        
        # Draw enhanced box with different styling for optimized
        if version == 'optimized':
            # Optimized boxes have a glow effect
            for thickness in range(8, 0, -1):
                alpha = int(40 / thickness)
                glow_color = base_color[:3] + (alpha,)
                draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness], 
                             outline=glow_color, width=1)
        
        # Main border (thicker for optimized)
        border_width = 4 if version == 'optimized' else 2
        draw.rectangle([x1, y1, x2, y2], outline=base_color, width=border_width)
        
        # Label with saliency score
        text = cta['text'][:20] + ('...' if len(cta['text']) > 20 else '')
        label = f"#{i+1}: {text} ({saliency:.0%})"
        
        # Calculate label dimensions
        label_width = len(label) * 7 + 16
        label_height = 24
        
        # Position label
        label_y = y1 - label_height - 5 if y1 > label_height + 10 else y2 + 5
        label_x = min(x1, width - label_width)
        
        # Draw label background
        draw.rectangle([label_x, label_y, label_x + label_width, label_y + label_height],
                      fill=base_color)
        
        # Label text (white for better contrast)
        draw.text((label_x + 8, label_y + 4), label, fill=(255, 255, 255, 255))
    
    return Image.alpha_composite(img_with_boxes, overlay)

def add_flow_indicators(image: Image.Image, flow_data: Dict) -> Image.Image:
    """Add attention flow indicators"""
    if not flow_data:
        return image
        
    img_with_flow = image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    width, height = image.size
    
    # Primary entry point
    primary_entry = flow_data.get('primary_entry', [0.5, 0.3])
    px, py = int(primary_entry[0] * width), int(primary_entry[1] * height)
    
    # Draw entry point indicator
    for radius in range(25, 5, -4):
        alpha = int(180 * (25 - radius) / 20)
        draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                    outline=(255, 215, 0, alpha), width=3)
    
    # Add entry point label
    draw.text((px - 25, py - 40), "ENTRY POINT", fill=(255, 215, 0, 255))
    
    return Image.alpha_composite(img_with_flow, overlay)

def add_comparison_label(image: Image.Image, label: str, color: Tuple[int, int, int]) -> Image.Image:
    """Add comparison label to image"""
    img_with_label = image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Label dimensions
    label_width = len(label) * 12 + 40
    label_height = 40
    
    # Position at top center
    x = (image.size[0] - label_width) // 2
    y = 20
    
    # Draw label background with rounded corners effect
    draw.rectangle([x, y, x + label_width, y + label_height],
                  fill=color + (220,))
    
    # Draw border
    draw.rectangle([x, y, x + label_width, y + label_height],
                  outline=color + (255,), width=2)
    
    # Draw text (centered)
    text_x = x + 20
    text_y = y + 10
    draw.text((text_x, text_y), label, fill=(255, 255, 255, 255))
    
    return Image.alpha_composite(img_with_label, overlay)

# ============================================================================
# WEB SCRAPING & SCREENSHOTS
# ============================================================================

async def screenshot_url(url: str, full_page: bool = True) -> Optional[Image.Image]:
    """Take screenshot of URL using Playwright"""
    if not PLAYWRIGHT_AVAILABLE:
        logger.warning("Playwright not available for screenshots")
        return None
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport={'width': 1440, 'height': 900})
            
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            if full_page:
                height = await page.evaluate('document.body.scrollHeight')
                await page.set_viewport_size({'width': 1440, 'height': height})
            
            screenshot_bytes = await page.screenshot(full_page=full_page)
            await browser.close()
            
            return Image.open(io.BytesIO(screenshot_bytes)).convert('RGB')
            
    except Exception as e:
        logger.error(f"Screenshot failed for {url}: {e}")
        return None

def fetch_page_html(url: str) -> str:
    """Fetch HTML content from URL"""
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return response.text[:15000]
    except Exception as e:
        logger.error(f"Failed to fetch HTML for {url}: {e}")
        return ""

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="🎯 Attention Optimization AI", version="2.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory=Config.STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=Config.OUTPUTS_DIR), name="outputs")

# Templates
templates = Jinja2Templates(directory=Config.TEMPLATES_DIR)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests with timing"""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    logger.info(f"🚀 Request {request_id}: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"✅ Request {request_id} completed in {process_time:.2f}s")
    
    return response

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Enhanced main page"""
    status = {
        'openai_available': OPENAI_AVAILABLE and bool(Config.OPENAI_API_KEY),
        'playwright_available': PLAYWRIGHT_AVAILABLE,
        'model': Config.OPENAI_MODEL
    }
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "status": status
    })

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    url: str = Form(""),
    file: UploadFile = Form(None)
):
    """Enhanced analysis endpoint"""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"🔍 Analysis request {request_id}: url='{url}', file='{file.filename if file and file.filename else None}'")
    
    try:
        # Get image and HTML
        image = None
        html = ""
        
        # Check if file was uploaded and has content
        if file and file.filename and file.size > 0:
            try:
                content = await file.read()
                logger.info(f"📁 File read: {len(content)} bytes")
                
                if len(content) == 0:
                    return HTMLResponse(
                        '<div class="error-message">⚠️ Uploaded file is empty. Please try again.</div>',
                        status_code=400
                    )
                
                image = Image.open(io.BytesIO(content)).convert('RGB')
                logger.info(f"📁 Image uploaded successfully: {image.size}")
                
            except Exception as e:
                logger.error(f"Error processing uploaded file: {e}")
                return HTMLResponse(
                    '<div class="error-message">❌ Invalid image file. Please upload a valid PNG, JPEG, or WebP image.</div>',
                    status_code=400
                )
            
        elif url and url.strip():
            clean_url = url.strip()
            if not clean_url.startswith(('http://', 'https://')):
                clean_url = f'https://{clean_url}'
            
            logger.info(f"🌐 Taking screenshot of: {clean_url}")
            image = await screenshot_url(clean_url, True)
            html = fetch_page_html(clean_url)
            
            if not image:
                return HTMLResponse(
                    '<div class="error-message">📸 Failed to capture screenshot. Please check the URL and try again, or upload an image instead.</div>',
                    status_code=400
                )
            
            logger.info(f"🌐 Screenshot captured: {image.size}")
        else:
            logger.warning("No input provided - neither URL nor file")
            return HTMLResponse(
                '<div class="error-message">⚠️ Please provide either a URL or upload an image file.</div>',
                status_code=400
            )
        
        # Run enhanced AI analysis
        logger.info(f"🤖 Running AI analysis...")
        analysis = await analyze_with_openai(image, html)
        logger.info(f"✨ Analysis completed - Current: {analysis.get('overall_scores', {}).get('current_score', 0)}/100, Potential: {analysis.get('overall_scores', {}).get('potential_score', 0)}/100")
        
        # Create comparison visualization (current vs optimized)
        current_image, optimized_image, metrics = create_comparison_visualization(image, analysis)
        
        # Save results
        run_id = str(uuid.uuid4())[:8]
        current_path = Config.OUTPUTS_DIR / f"{run_id}_current.png"
        optimized_path = Config.OUTPUTS_DIR / f"{run_id}_optimized.png"
        json_path = Config.OUTPUTS_DIR / f"{run_id}.json"
        
        current_image.convert('RGB').save(current_path, 'PNG', quality=95, optimize=True)
        optimized_image.convert('RGB').save(optimized_path, 'PNG', quality=95, optimize=True)
        
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Template context
        context = {
            "request": request,
            "run_id": run_id,
            "analysis": analysis,
            "metrics": metrics,
            "current_image_url": f"/outputs/{run_id}_current.png",
            "optimized_image_url": f"/outputs/{run_id}_optimized.png",
            "success": True
        }
        
        return templates.TemplateResponse("results.html", context)
        
    except Exception as e:
        logger.error(f"❌ Analysis failed for request {request_id}: {e}")
        error_msg = str(e) if Config.DEBUG else "Analysis failed. Please try again."
        return HTMLResponse(
            f'<div class="error-message">❌ {error_msg}</div>',
            status_code=500
        )

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check system status"""
    return {
        "status": "running",
        "openai_available": OPENAI_AVAILABLE,
        "playwright_available": PLAYWRIGHT_AVAILABLE,
        "openai_key_set": bool(Config.OPENAI_API_KEY),
        "directories": {
            "outputs": str(Config.OUTPUTS_DIR),
            "static": str(Config.STATIC_DIR),  
            "templates": str(Config.TEMPLATES_DIR)
        },
        "directories_exist": {
            "outputs": Config.OUTPUTS_DIR.exists(),
            "static": Config.STATIC_DIR.exists(),
            "templates": Config.TEMPLATES_DIR.exists()
        }
    }

@app.get("/test-upload")
async def test_upload():
    """Test upload form"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head><title>Test Upload</title></head>
    <body>
        <h2>Test File Upload</h2>
        <form action="/analyze" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required><br><br>
            <button type="submit">Upload</button>
        </form>
    </body>
    </html>
    """)

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "🟢 healthy",
        "openai_available": OPENAI_AVAILABLE and bool(Config.OPENAI_API_KEY),
        "playwright_available": PLAYWRIGHT_AVAILABLE,
        "version": "2.0.0"
    }

@app.get("/download/{run_id}")
async def download_results(run_id: str, format: str = "png"):
    """Download results - supports current, optimized, and json formats"""
    if format == "current":
        file_path = Config.OUTPUTS_DIR / f"{run_id}_current.png"
        if file_path.exists():
            return FileResponse(file_path, filename=f"current-attention-{run_id}.png")
    elif format == "optimized":
        file_path = Config.OUTPUTS_DIR / f"{run_id}_optimized.png"
        if file_path.exists():
            return FileResponse(file_path, filename=f"optimized-attention-{run_id}.png")
    elif format == "json":
        file_path = Config.OUTPUTS_DIR / f"{run_id}.json"
        if file_path.exists():
            return FileResponse(file_path, filename=f"attention-analysis-{run_id}.json")
    elif format == "png":  # Backwards compatibility
        # Try optimized first, then current
        for suffix in ["_optimized", "_current"]:
            file_path = Config.OUTPUTS_DIR / f"{run_id}{suffix}.png"
            if file_path.exists():
                return FileResponse(file_path, filename=f"attention-analysis-{run_id}.png")
    
    raise HTTPException(404, f"File not found for format: {format}")

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🎯 ATTENTION OPTIMIZATION AI - ENHANCED VERSION")
    print("="*80)
    print(f"✅ OpenAI Available: {'Yes' if OPENAI_AVAILABLE and Config.OPENAI_API_KEY else 'No'}")
    print(f"🔑 API Key: {'Configured' if Config.OPENAI_API_KEY else 'Missing'}")
    print(f"📷 Screenshots: {'Enabled' if PLAYWRIGHT_AVAILABLE else 'Disabled'}")
    print(f"🤖 AI Model: {Config.OPENAI_MODEL}")
    print(f"🌐 Server: http://{Config.HOST}:{Config.PORT}")
    print("="*80)
    print("🚀 Features:")
    print("  • Professional heatmap visualization")
    print("  • Enhanced CTA analysis with priorities")
    print("  • Beautiful two-column results layout")
    print("  • Quick wins vs strategic improvements")
    print("  • Ready-to-implement A/B test ideas")
    print("  • Mobile-responsive design")
    print("="*80)
    
    if not Config.OPENAI_API_KEY:
        print("⚠️  Set OPENAI_API_KEY environment variable to enable analysis")
        print("   Example: export OPENAI_API_KEY='sk-your-key-here'")
        print("="*80)
    
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG
    )