import os
import chainlit as cl
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import httpx
from dotenv import load_dotenv
import requests
import json
import logging
import tracemalloc
import asyncio
import subprocess
import re

# Enable tracemalloc
tracemalloc.start()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# AI Provider and Model settings
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()
AI_MODEL = os.getenv("AI_MODEL", "gpt-3.5-turbo")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Initialize AI clients
if AI_PROVIDER == "openai":
    ai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    cl.instrument_openai()
elif AI_PROVIDER == "anthropic":
    ai_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
elif AI_PROVIDER == "ollama":
    ai_client = httpx.AsyncClient(base_url=OLLAMA_BASE_URL)
else:
    raise ValueError(f"Unsupported AI provider: {AI_PROVIDER}")

# Mempool API base URL
MEMPOOL_API_BASE = "https://mempool.space/api"

# AI settings
AI_SETTINGS = {
    "temperature": 0.7,
    "max_tokens": 1500,
    "stream": True
}


async def check_ollama_model():
    """Check if the specified Ollama model is available using the command-line."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if AI_MODEL in result.stdout:
            logger.info(f"Ollama model {AI_MODEL} is available.")
            return True
        
        logger.warning(f"Ollama model {AI_MODEL} is not available.")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama model: {str(e)}")
        return False
    
async def query_mempool_api(endpoint, params=None):
    """Query the Mempool API and return the response."""
    url = f"{MEMPOOL_API_BASE}{endpoint}"
    try:
        async with asyncio.timeout(15):  # Set a timeout of 15 seconds
            response = await asyncio.to_thread(requests.get, url, params=params)
        response.raise_for_status()
        logger.info(f"API request successful: {url}")
        return response.json()
    except asyncio.TimeoutError:
        logger.error(f"API request timed out: {url}")
        return None
    except requests.RequestException as e:
        logger.error(f"API request failed: {url}. Error: {str(e)}")
        return None

async def generate_ai_response(messages):
    """Generate AI response using the configured provider."""
    try:
        if AI_PROVIDER == "openai":
            response = await ai_client.chat.completions.create(
                model=AI_MODEL,
                messages=messages,
                **AI_SETTINGS
            )
            return response.choices[0].message.content
        elif AI_PROVIDER == "anthropic":
            system_message = next((msg['content'] for msg in messages if msg['role'] == 'system'), '')
            user_message = next((msg['content'] for msg in messages if msg['role'] == 'user'), '')
            response = await ai_client.messages.create(
                model=AI_MODEL,
                system=system_message,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                **AI_SETTINGS
            )
            return response.content[0].text
        elif AI_PROVIDER == "ollama":
            model_available = await check_ollama_model()
            if not model_available:
                return f"Error: The specified Ollama model '{AI_MODEL}' is not available. Please check your configuration or pull the model manually."
            
            response = await ai_client.post("/api/chat", json={
                "model": AI_MODEL,
                "messages": messages,
                **AI_SETTINGS
            })
            response.raise_for_status()
            response_text = response.text
            logger.debug(f"Ollama raw response: {response_text}")
            try:
                response_json = json.loads(response_text)
                return response_json.get("message", {}).get("content", "")
            except json.JSONDecodeError:
                # Handle streaming response
                response_lines = response_text.strip().split('\n')
                full_response = ""
                for line in response_lines:
                    try:
                        chunk = json.loads(line)
                        full_response += chunk.get("message", {}).get("content", "")
                    except json.JSONDecodeError:
                        continue
                return full_response
    except Exception as e:
        logger.error(f"AI response generation failed. Error: {str(e)}")
        return f"I'm sorry, but I encountered an error while processing your request: {str(e)}"
    

async def process_user_query(query):
    """Process user query and determine which API endpoint to use."""
    query = query.lower()
    
    # Define patterns for various intents
    patterns = {
        'transaction_details': r'(?:transaction|tx).*?([a-fA-F0-9]{64})',
        'address_info': r'(?:address|wallet).*?([13][a-km-zA-HJ-NP-Z1-9]{25,34})',
        'block_info': r'block.*?([a-fA-F0-9]{64})',
        'block_height': r'block.*?height.*?(\d+)',
        'bitcoin_price': r'(?:bitcoin|btc).*?(?:price|value|worth)',
        'mempool_info': r'mempool',
        'hashrate': r'(?:hash ?rate|mining power)',
        'difficulty': r'difficulty',
        'fees': r'(?:transaction )?fees?',
    }
    
    # Check for matches
    for intent, pattern in patterns.items():
        match = re.search(pattern, query)
        if match:
            if intent == 'transaction_details':
                txid = match.group(1)
                data = await query_mempool_api(f"/tx/{txid}")
                return f"Transaction details for {txid}:\n{json.dumps(data, indent=2)}" if data else "No data found or error occurred."
            
            elif intent == 'address_info':
                address = match.group(1)
                txs_data = await query_mempool_api(f"/address/{address}/txs")
                utxo_data = await query_mempool_api(f"/address/{address}/utxo")
                return f"Address information for {address}:\nTransactions: {json.dumps(txs_data, indent=2)}\nUTXOs: {json.dumps(utxo_data, indent=2)}"
            
            elif intent == 'block_info':
                block_hash = match.group(1)
                data = await query_mempool_api(f"/block/{block_hash}")
                return f"Block information for {block_hash}:\n{json.dumps(data, indent=2)}" if data else "No data found or error occurred."
            
            elif intent == 'block_height':
                height = match.group(1)
                data = await query_mempool_api(f"/block-height/{height}")
                return f"Block at height {height}: {data}" if data else "No data found or error occurred."
            
            elif intent == 'bitcoin_price':
                data = await query_mempool_api("/v1/prices")
                return f"Current Bitcoin prices:\n{json.dumps(data, indent=2)}" if data else "No data found or error occurred."
            
            elif intent == 'mempool_info':
                data = await query_mempool_api("/mempool")
                return f"Current mempool status:\n{json.dumps(data, indent=2)}" if data else "No data found or error occurred."
            
            elif intent == 'hashrate':
                data = await query_mempool_api("/v1/mining/hashrate/3m")
                return f"Bitcoin network hashrate (3 months):\n{json.dumps(data, indent=2)}" if data else "No data found or error occurred."
            
            elif intent == 'difficulty':
                data = await query_mempool_api("/v1/mining/difficulty-adjustments")
                return f"Bitcoin mining difficulty adjustments:\n{json.dumps(data, indent=2)}" if data else "No data found or error occurred."
            
            elif intent == 'fees':
                data = await query_mempool_api("/v1/fees/recommended")
                return f"Recommended transaction fees:\n{json.dumps(data, indent=2)}" if data else "No data found or error occurred."
    
    # If no intent is matched, return None to allow for general conversation
    return None




# Update the main function to use the new process_user_query
@cl.on_message
async def main(message: cl.Message):
    query = message.content
    logger.info(f"Received user query: {query}")

    # Process the query to get relevant API data
    api_data = await process_user_query(query)

    # Prepare the context for the AI
    if api_data:
        context = f"User query: {query}\n\nRelevant Bitcoin data: {api_data}\n\nYou are a Bitcoin expert in explaining current bitcoin data extracted from the api request, Please provide a helpful response based on this information, explaining the data if necessary. Provide a chart, graph or summery if it applies. If the data is too long or complex, summarize the key points."
    else:
        context = f"User query: {query}\n\nNo specific Bitcoin data was retrieved for this query. Please provide a friendly and helpful response based on your general knowledge about Bitcoin and blockchain technology. If the query is not related to Bitcoin, you can engage in general conversation."

    # Generate AI response
    ai_response = await generate_ai_response([
        {"role": "system", "content": "You are a friendly and knowledgeable AI assistant specializing in Bitcoin and blockchain technology, but also capable of general conversation. When asked about Bitcoin, provide accurate and helpful information, explaining technical concepts in an easy-to-understand manner. For other topics, engage in a natural, conversational manner."},
        {"role": "user", "content": context}
    ])

    # Send the response back to the user
    await cl.Message(content=ai_response).send()

@cl.on_chat_start
async def start():
    if AI_PROVIDER == "ollama":
        model_available = await check_ollama_model()
        if not model_available:
            await cl.Message(content=f"Warning: The specified Ollama model '{AI_MODEL}' is not available. Please check your configuration or pull the model manually.").send()
        else:
            await cl.Message(content=f"Ollama model '{AI_MODEL}' is available and ready to use.").send()

    welcome_message = f"""
# 🪙 Welcome to the ₿itcoin Mempool AI Assistant!

Hello! I'm your friendly AI assistant specializing in Bitcoin and blockchain technology. I'm here to chat about anything, but I have particular expertise in:

1. 📈 ₿itcoin prices and market trends
2. 🔗 ₿lockchain technology and how it works
3. 💼 ₿itcoin transactions and the mempool
4. ⛏️ Mining and network statistics
5. 🌐 General cryptocurrency topics

Feel free to ask me about these topics or anything else you'd like to discuss!

---

**Current Configuration:**
- AI Provider: `{AI_PROVIDER.capitalize()}`
- AI Model: `{AI_MODEL}`

---

💡 **Tip**: Try asking about a specific ₿itcoin transaction, address, or the current market price!
"""

    await cl.Message(content=welcome_message).send()
    