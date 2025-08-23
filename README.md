# 🎯 Attention Optimization AI

**Professional CRO & UX Analysis powered by OpenAI Vision**

Transform your landing pages with AI-powered attention optimization insights, saliency maps, and data-driven optimization recommendations.

![Attention Optimization AI](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **🔍 AI-Powered Analysis**: OpenAI Vision API for intelligent landing page analysis
- **📸 URL Screenshots**: Automatic webpage capture using Playwright
- **🎨 Saliency Maps**: Visual attention heatmaps with customizable grid resolution
- **📊 CTA Analysis**: Detailed call-to-action performance metrics
- **💡 Optimization Tips**: AI-generated improvement suggestions
- **🧪 A/B Test Ideas**: Data-driven hypotheses for experimentation
- **📱 Modern UI**: Beautiful, responsive interface with real-time updates
- **⚡ Fast Performance**: HTMX-powered dynamic updates without page reloads

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Modern web browser

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd "Attention optmization bot"

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install
```

### 2. Configure OpenAI

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"

# Optional: Set custom model
export OPENAI_MODEL="gpt-4o"  # or gpt-4o-mini, gpt-4-turbo
```

### 3. Launch Application

```bash
# Option 1: Use the startup script
./start.sh

# Option 2: Manual launch
source .venv/bin/activate
uvicorn main:app --reload --port 8080
```

### 4. Open Browser

Navigate to [http://localhost:8080](http://localhost:8080)

## 🎨 Usage

### Analyze a URL
1. Enter a landing page URL in the form
2. Choose screenshot options (full-page recommended)
3. Adjust analysis settings (grid resolution, AI model, creativity level)
4. Click "Analyze with AI"

### Upload an Image
1. Click the file upload area
2. Select a PNG, JPEG, or WebP image
3. Configure analysis parameters
4. Submit for AI analysis

### Understanding Results

- **Annotated Image**: Visual overlay with CTA highlights and saliency map
- **Key Metrics**: Performance indicators and confidence scores
- **Optimization Suggestions**: AI-recommended improvements
- **CTA Analysis**: Detailed breakdown of each call-to-action
- **A/B Test Ideas**: Data-driven experimentation hypotheses

## 🏗️ Architecture

- **Backend**: FastAPI with async support
- **Frontend**: Modern HTML5 + HTMX for dynamic updates
- **AI**: OpenAI Vision API with structured JSON outputs
- **Screenshots**: Playwright for reliable webpage capture
- **Styling**: Custom CSS with modern design system
- **Icons**: Font Awesome for professional iconography

## 📁 Project Structure

```
Attention optmization bot/
├── main.py                 # FastAPI application
├── templates/              # Jinja2 templates
│   ├── base.html          # Base template with navigation
│   └── index.html         # Main form interface
├── partials/              # HTMX partial templates
│   └── result.html        # Results display template
├── static/                # Static assets
│   └── styles.css         # Modern CSS design system
├── outputs/               # Generated analysis files
├── requirements.txt       # Python dependencies
├── start.sh              # Startup script
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | AI model to use | `gpt-4o` |

### Analysis Parameters

- **Grid Resolution**: 6x6 to 24x24 (default: 12x8)
- **AI Model**: Any OpenAI Vision-compatible model
- **Temperature**: 0.0 (focused) to 1.0 (creative)
- **Screenshot**: Full-page or viewport-only

## 📊 Output Formats

### PNG Image
- High-resolution annotated screenshot
- Saliency heatmap overlay
- CTA bounding boxes with labels
- Professional presentation ready

### JSON Data
- Raw AI analysis results
- Structured data for further processing
- Compatible with analytics tools
- API integration ready

## 🚀 Deployment

### Local Development
```bash
./start.sh
```

### Production Deployment
```bash
# Install production dependencies
pip install -r requirements.txt

# Set production environment
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o"

# Run with production server
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Docker (Coming Soon)
```bash
# Build and run with Docker
docker build -t attention-ai .
docker run -p 8080:8080 -e OPENAI_API_KEY=your-key attention-ai
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the Vision API
- FastAPI team for the excellent framework
- Playwright for reliable web automation
- HTMX for seamless dynamic updates

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

---

**Made with ❤️ for better user experiences**
