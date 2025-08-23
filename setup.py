#!/usr/bin/env python3
"""
Attention Optimization AI - Setup Script
Automates the installation and configuration process.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🔧 {description}")
    print(f"Running: {command}")
    print('='*50)
    
    try:
        # Split command into list to avoid shell escaping issues with spaces
        if isinstance(command, str):
            command = command.split()
        
        result = subprocess.run(command, shell=False, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"📍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['outputs', 'static', 'templates']
    
    print("\n📁 Creating directories...")
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✅ {dir_name}/")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      "Installing requirements"):
        return False
    
    return True

def install_playwright():
    """Install Playwright browsers"""
    print("\n🎭 Installing Playwright browsers...")
    
    # Install playwright
    if not run_command([sys.executable, "-m", "pip", "install", "playwright"], 
                      "Installing Playwright"):
        return False
    
    # Install browsers
    if not run_command([sys.executable, "-m", "playwright", "install", "chromium"], 
                      "Installing Chromium browser"):
        print("⚠️  Playwright browser installation failed - URL screenshots will be disabled")
        return False
    
    return True

def create_env_file():
    """Create .env file template"""
    env_file = Path('.env')
    
    if env_file.exists():
        print("\n✅ .env file already exists")
        return True
    
    print("\n📝 Creating .env template...")
    
    env_content = """# Attention Optimization AI Configuration

# OpenAI API Key (Required for AI analysis)
# Get your key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# OpenAI Model (Optional)
OPENAI_MODEL=gpt-4o

# Debug settings (Optional)
DEBUG_MODE=false
LOG_LEVEL=INFO

# Image processing (Optional)
MAX_IMAGE_SIDE=2048
"""
    
    env_file.write_text(env_content)
    print("✅ Created .env template")
    print("📝 Please edit .env and add your OpenAI API key")
    
    return True

def validate_setup():
    """Validate the setup"""
    print("\n🔍 Validating setup...")
    
    # Check if main.py exists
    if not Path('main.py').exists():
        print("❌ main.py not found")
        return False
    
    # Check templates
    templates_dir = Path('templates')
    required_templates = ['index.html', 'results.html']
    
    for template in required_templates:
        if not (templates_dir / template).exists():
            print(f"❌ Template missing: {template}")
            return False
    
    print("✅ All required files present")
    
    # Test imports
    try:
        import fastapi
        import uvicorn
        import PIL
        import requests
        print("✅ Core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test optional imports
    try:
        import openai
        print("✅ OpenAI SDK available")
    except ImportError:
        print("⚠️  OpenAI SDK not available - AI analysis will be disabled")
    
    try:
        import playwright
        print("✅ Playwright available")
    except ImportError:
        print("⚠️  Playwright not available - URL screenshots will be disabled")
    
    return True

def show_next_steps():
    """Show what to do next"""
    print(f"\n{'='*60}")
    print("🎉 SETUP COMPLETE!")
    print('='*60)
    
    print("\n📋 Next Steps:")
    print("   1. Edit .env file and add your OpenAI API key")
    print("   2. Get API key from: https://platform.openai.com/api-keys")
    print("   3. Run the application:")
    print(f"      python main.py")
    print("   4. Open browser to: http://localhost:8080")
    
    print("\n🔧 Optional Setup:")
    print("   • For URL screenshots: playwright install")
    print("   • For development: pip install uvicorn[standard]")
    
    print("\n📚 Documentation:")
    print("   • README.md - Full documentation")
    print("   • Check GitHub for examples and updates")
    
    print(f"\n{'='*60}")

def main():
    """Main setup function"""
    print("🚀 Attention Optimization AI - Setup Script")
    print("This will install dependencies and configure the application.\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Install Playwright (optional)
    install_playwright()
    
    # Create .env file
    if not create_env_file():
        print("❌ Failed to create .env file")
        sys.exit(1)
    
    # Validate setup
    if not validate_setup():
        print("❌ Setup validation failed")
        sys.exit(1)
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()