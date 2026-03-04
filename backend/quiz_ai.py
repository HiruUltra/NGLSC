import os
import json
import random
import time
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configurations (Use environment variables for security)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

SYSTEM_PROMPT = """
You are a specialized MCQ generator. Create high-quality multiple choice questions.
For each question, provide:
1. The question text.
2. An array of 4 distinct options.
3. The exact text of the correct answer from the options.

Format the output strictly as a valid JSON array of objects with keys:
"question", "options", "correctAnswer"
"""

USER_PROMPT_TEMPLATE = "Generate {count} unique and challenging multiple choice questions about '{topic}'."

def generate_with_claude(topic, count):
    """Generates MCQs using Anthropic's Claude API"""
    if not ANTHROPIC_API_KEY:
        raise ValueError("Anthropic API key not configured")

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 2048,
        "temperature": 0.9,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(topic=topic, count=count)}
        ]
    }
    
    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    
    # Extract JSON string from Claude's response
    content = result["content"][0]["text"]
    return _extract_json(content)

def generate_with_gemini(topic, count):
    """Generates MCQs using Google's Gemini API"""
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key not configured")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    
    data = {
        "contents": [{
            "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(topic=topic, count=count)}"}]
        }],
        "generationConfig": {
            "temperature": 1.0,
            "response_mime_type": "application/json"
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    
    content = result["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(content)

def generate_with_deepseek(topic, count):
    """Generates MCQs using DeepSeek API (OpenAI-compatible)"""
    if not DEEPSEEK_API_KEY:
        raise ValueError("DeepSeek API key not configured")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(topic=topic, count=count)}
        ],
        "temperature": 0.8,
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    
    content = result["choices"][0]["message"]["content"]
    # Handle cases where DeepSeek might return an object with a key like 'questions'
    data = json.loads(content)
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], list):
                return data[key]
    return data

def generate_quiz(topic, number_of_questions):
    """Main entry point: Randomly rotates between APIs and handles seed"""
    # Set seed based on current timestamp for high randomness
    random.seed(time.time())
    
    generators = [generate_with_claude, generate_with_gemini, generate_with_deepseek]
    random.shuffle(generators)
    
    last_error = None
    for generator in generators:
        try:
            logger.info(f"Attempting quiz generation with {generator.__name__}")
            questions = generator(topic, number_of_questions)
            if questions and isinstance(questions, list):
                return questions
        except Exception as e:
            logger.error(f"Error with {generator.__name__}: {str(e)}")
            last_error = e
            continue
            
    raise Exception(f"All AI providers failed. Last error: {str(last_error)}")

def _extract_json(text):
    """Helper to extract JSON array from potentially messy LLM output"""
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
        return json.loads(text)
    except Exception:
        raise ValueError("Failed to parse AI response as JSON")
