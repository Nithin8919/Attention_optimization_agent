#!/usr/bin/env python3
"""
Quick test script to debug file upload issues
"""

import requests
from pathlib import Path

def test_upload():
    """Test file upload functionality"""
    
    # Create a simple test image
    from PIL import Image
    import io
    
    # Create a test image
    img = Image.new('RGB', (300, 200), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Test the upload
    files = {'file': ('test.png', img_bytes, 'image/png')}
    
    try:
        print("🧪 Testing file upload...")
        response = requests.post('http://localhost:8080/analyze', files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500]}...")
        
    except Exception as e:
        print(f"Error: {e}")

def test_debug():
    """Test debug endpoint"""
    try:
        print("🔍 Testing debug endpoint...")
        response = requests.get('http://localhost:8080/debug')
        print(f"Debug info: {response.json()}")
        
    except Exception as e:
        print(f"Error: {e}")

def test_health():
    """Test health endpoint"""
    try:
        print("❤️ Testing health endpoint...")
        response = requests.get('http://localhost:8080/health')
        print(f"Health: {response.json()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("🎯 Attention Optimization AI - Test Script")
    print("="*50)
    
    test_health()
    print()
    test_debug()
    print()
    test_upload()