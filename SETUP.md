# Setup Guide for Attention Optimization Bot

## Environment Variables

The main issue causing the 500 Internal Server Error is missing environment variables. You need to set up the following:

### 1. OpenAI API Key (Required)

You need to set your OpenAI API key as an environment variable. You can do this in several ways:

**Option A: Export in your shell (temporary)**
```bash
export OPENAI_API_KEY="your_actual_api_key_here"
```

**Option B: Create a .env file (recommended)**
```bash
# Create a .env file in the project root
echo "OPENAI_API_KEY=your_actual_api_key_here" > .env
```

**Option C: Set in your shell profile (permanent)**
```bash
# Add to ~/.zshrc or ~/.bash_profile
echo 'export OPENAI_API_KEY="your_actual_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

### 2. Get Your OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key and use it in the environment variable above

### 3. Optional Environment Variables

```bash
# Specify OpenAI model (defaults to gpt-4o)
export OPENAI_MODEL="gpt-4o"

# Set temperature for AI responses (0.0 to 2.0, defaults to 0.2)
export TEMPERATURE=0.2
```

## Running the Application

1. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Set your API key:**
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

4. **Or use uvicorn directly:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Troubleshooting

### Common Issues:

1. **500 Internal Server Error**: Usually means missing or invalid OpenAI API key
2. **"OpenAI API key missing"**: Set the OPENAI_API_KEY environment variable
3. **"Invalid OpenAI API key"**: Check your API key is correct and has credits
4. **"API quota exceeded"**: Your OpenAI account needs more credits

### Testing Your Setup:

```bash
# Test if environment variable is set
echo $OPENAI_API_KEY

# Test if the app can import all dependencies
python -c "import fastapi, uvicorn, jinja2, PIL, requests, openai; print('All good!')"

# Test if the app starts without errors
python main.py
```

## Security Notes

- Never commit your API key to version control
- The .env file is already in .gitignore
- Consider using a service like [direnv](https://direnv.net/) for local development
