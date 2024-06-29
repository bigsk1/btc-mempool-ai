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
import plotly.graph_objects as go
import pandas as pd
import httpx
import aiohttp
from datetime import datetime, timedelta


# Enable tracemalloc
tracemalloc.start()

# Add these variables at the top of your file
OLLAMA_MAX_RETRIES = 3
OLLAMA_RETRY_DELAY = 2  # seconds

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# AI Provider and Model settings
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o")
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
    "max_tokens": 1500
}


async def query_historical_price(timestamp, currency='USD'):
    """Query the Mempool API for historical price data."""
    params = {
        'timestamp': int(timestamp),
        'currency': currency
    }
    data = await query_mempool_api("/v1/historical-price", params)
    return data

async def query_current_price():
    """Query the Mempool API for current price data."""
    return await query_mempool_api("/v1/prices")

async def create_bitcoin_price_graph(start_price, end_price, currency, time_period):
    """Create a graph of Bitcoin price change."""
    start_time = datetime.now() - timedelta(days=30)  # Assuming 1 month for this example
    end_time = datetime.now()
    
    df = pd.DataFrame([
        {'timestamp': start_time, 'price': start_price},
        {'timestamp': end_time, 'price': end_price}
    ])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines+markers', name=f'Bitcoin Price in {currency}'))
    
    fig.update_layout(
        title=f'Bitcoin Price Change in {currency} over the last {time_period}',
        xaxis_title='Date',
        yaxis_title=f'Price ({currency})',
        height=600,
        width=800
    )
    
    img_bytes = fig.to_image(format="png")
    return img_bytes

def parse_time_period(query):
    """Parse the time period from the user query."""
    time_periods = {
        'day': 1,
        'week': 7,
        'month': 30,
        'year': 365
    }
    
    for period, days in time_periods.items():
        if period in query or f"{days} days" in query:
            return timedelta(days=days), period
    
    return timedelta(days=30), 'month'  # Default to 1 month if no period is specified

async def check_ollama_model():
    """Check if the specified Ollama model is available using the command-line."""
    if hasattr(check_ollama_model, "result"):
        return check_ollama_model.result

    for attempt in range(OLLAMA_MAX_RETRIES):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if AI_MODEL in result.stdout:
                logger.info(f"Ollama model {AI_MODEL} is available.")
                check_ollama_model.result = True
                return True
            else:
                logger.warning(f"Ollama model {AI_MODEL} is not available.")
                check_ollama_model.result = False
                return False
        except Exception as e:
            logger.error(f"Error checking Ollama model (attempt {attempt + 1}): {str(e)}")
            if attempt < OLLAMA_MAX_RETRIES - 1:
                await asyncio.sleep(OLLAMA_RETRY_DELAY)
            else:
                check_ollama_model.result = False
                return False
    
async def query_mempool_api(endpoint, params=None):
    """Query the Mempool API and return the response."""
    url = f"{MEMPOOL_API_BASE}{endpoint}"
    logger.info(f"Attempting to query Mempool API: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=15) as response:
                logger.info(f"API request status: {response.status}")
                response.raise_for_status()
                data = await response.json()
                logger.info(f"API request successful: {url}")
                logger.debug(f"API response data: {data}")
                return data
    except aiohttp.ClientError as e:
        logger.error(f"API request failed: {url}. Error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in API request: {url}. Error: {str(e)}")
        return None


async def generate_ai_response(messages):
    """Generate AI response using the configured provider."""
    try:
        if AI_PROVIDER == "openai":
            stream = await ai_client.chat.completions.create(
                model=AI_MODEL,
                messages=messages,
                **AI_SETTINGS
            )
            full_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
            return full_response
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
            
            for attempt in range(OLLAMA_MAX_RETRIES):
                try:
                    async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=30.0) as client:
                        response = await client.post("/api/chat", json={
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
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    logger.error(f"Ollama request failed (attempt {attempt + 1}): {str(e)}")
                    if attempt < OLLAMA_MAX_RETRIES - 1:
                        await asyncio.sleep(OLLAMA_RETRY_DELAY)
                    else:
                        raise
    except Exception as e:
        logger.error(f"AI response generation failed. Error: {str(e)}", exc_info=True)
        return f"I'm sorry, but I encountered an error while processing your request: {str(e)}"
    

async def process_user_query(query):
    """Process user query and determine which API endpoint to use."""
    query = query.lower()
    
    # Pattern for historical price queries
    historical_price_pattern = r'(bitcoin|btc) price.*(over|for|in|during).*?(last|past|previous)'
    match = re.search(historical_price_pattern, query)
    if match:
        time_delta, time_period = parse_time_period(query)
        end_time = datetime.now()
        start_time = end_time - time_delta
        
        currency = 'USD'  # Default to USD, but you could extract this from the query too
        if 'eur' in query:
            currency = 'EUR'
        elif 'gbp' in query:
            currency = 'GBP'
        
        try:
            historical_data = await query_historical_price(int(start_time.timestamp()), currency)
            current_data = await query_current_price()
            
            start_price = historical_data['prices'][0][currency]
            end_price = current_data[currency]
            
            return "price_change", (start_price, end_price), (currency, time_period)
        except Exception as e:
            logger.error(f"Error fetching price data: {str(e)}")
            return "error", str(e), None

    # Pattern for mempool fee queries
    mempool_fee_pattern = r'(mempool|transaction) fee.*(over|for|in|during).*?(last|past|previous)'
    match = re.search(mempool_fee_pattern, query)
    if match:
        time_delta, time_period = parse_time_period(query)
        try:
            mempool_data = await query_mempool_api("/mempool")
            return "mempool_fees", mempool_data, time_period  # Return full mempool data
        except Exception as e:
            logger.error(f"Error fetching mempool data: {str(e)}")
            return "error", str(e), None
    
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
            try:
                if intent == 'transaction_details':
                    txid = match.group(1)
                    data = await query_mempool_api(f"/tx/{txid}")
                    return intent, data, txid
                
                elif intent == 'address_info':
                    address = match.group(1)
                    txs_data = await query_mempool_api(f"/address/{address}/txs")
                    utxo_data = await query_mempool_api(f"/address/{address}/utxo")
                    return intent, {"transactions": txs_data, "utxos": utxo_data}, address
                
                elif intent == 'block_info':
                    block_hash = match.group(1)
                    data = await query_mempool_api(f"/block/{block_hash}")
                    return intent, data, block_hash
                
                elif intent == 'block_height':
                    height = match.group(1)
                    data = await query_mempool_api(f"/block-height/{height}")
                    return intent, data, height
                
                elif intent == 'bitcoin_price':
                    data = await query_mempool_api("/v1/prices")
                    return intent, data, None
                
                elif intent == 'mempool_info':
                    data = await query_mempool_api("/mempool")
                    return intent, data, None
                
                elif intent == 'hashrate':
                    data = await query_mempool_api("/v1/mining/hashrate/3m")
                    return intent, data, None
                
                elif intent == 'difficulty':
                    data = await query_mempool_api("/v1/mining/difficulty-adjustments")
                    return intent, data, None
                
                elif intent == 'fees':
                    data = await query_mempool_api("/v1/fees/recommended")
                    return intent, data, None
            
            except Exception as e:
                logger.error(f"Error processing {intent} query: {str(e)}")
                return "error", str(e), None
    
    # If no specific intent is matched, return a general intent
    return "general", None, None

# Update the main function to use the new process_user_query
async def create_mempool_graph(mempool_data):
    fee_histogram = mempool_data.get('fee_histogram', [])
    
    # Extract fee rates and counts
    fee_rates = [entry[0] for entry in fee_histogram]
    counts = [entry[1] for entry in fee_histogram]
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({'fee_rate': fee_rates, 'count': counts})
    
    # Sort by fee rate and calculate cumulative sum
    df = df.sort_values('fee_rate')
    df['cumulative'] = df['count'].cumsum()
    
    # Create the plot
    fig = go.Figure()
    
    # Add bar chart for transaction count
    fig.add_trace(go.Bar(
        x=df['fee_rate'],
        y=df['count'],
        name='Transaction Count',
        marker_color='blue',
        opacity=0.7
    ))
    
    # Add line chart for cumulative distribution
    fig.add_trace(go.Scatter(
        x=df['fee_rate'],
        y=df['cumulative'],
        name='Cumulative Distribution',
        yaxis='y2',
        line=dict(color='red', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title='Mempool Fee Distribution',
        xaxis_title='Fee Rate (sat/vB)',
        yaxis_title='Transaction Count',
        yaxis2=dict(
            title='Cumulative Count',
            overlaying='y',
            side='right'
        ),
        barmode='overlay',
        legend=dict(x=0.1, y=1.1, orientation='h'),
        height=600,
        width=800
    )
    
    # Save the figure as a PNG image
    img_bytes = fig.to_image(format="png")
    return img_bytes

@cl.on_message
async def main(message: cl.Message):
    query = message.content
    logger.info(f"Received user query: {query}")

    # Direct API call test
    logger.info("Attempting direct API call to Mempool")
    direct_api_result = await query_mempool_api("/v1/prices")
    if direct_api_result:
        logger.info(f"Direct API call successful. Result: {direct_api_result}")
    else:
        logger.error("Direct API call failed")

    # Process the query to get relevant API data
    intent, api_data, extra_info = await process_user_query(query)
    logger.info(f"Processed query. Intent: {intent}, Extra info: {extra_info}")

    if intent == "price_change":
        start_price, end_price = api_data
        currency, time_period = extra_info
        try:
            img_bytes = await create_bitcoin_price_graph(start_price, end_price, currency, time_period)
            await cl.Message(content=f"Here's a graph of the Bitcoin price change in {currency} over the last {time_period}:").send()
            await cl.Message(content="", elements=[
                cl.Image(name="bitcoin_price_graph", content=img_bytes, mime="image/png")
            ]).send()
        except Exception as e:
            logger.error(f"Error creating Bitcoin price graph: {str(e)}")
            await cl.Message(content=f"I'm sorry, I couldn't create the graph due to an error: {str(e)}").send()

    elif intent == "mempool_fees":
        mempool_data = api_data
        time_period = extra_info
        try:
            img_bytes = await create_mempool_graph(mempool_data)
            await cl.Message(content=f"Here's a graph of the mempool fee distribution:").send()
            await cl.Message(content="", elements=[
                cl.Image(name="mempool_fee_graph", content=img_bytes, mime="image/png")
            ]).send()
        except Exception as e:
            logger.error(f"Error creating mempool fee graph: {str(e)}")
            await cl.Message(content=f"I'm sorry, I couldn't create the graph due to an error: {str(e)}").send()

    elif intent == "error":
        await cl.Message(content=f"I'm sorry, I encountered an error while processing your request: {api_data}").send()


    # Prepare the context for the AI
    if intent == "price_change":
        context = f"User query: {query}\n\nIntent: {intent}\n\nRelevant Bitcoin data: Start price: {api_data[0]}, End price: {api_data[1]}, Currency: {extra_info[0]}, Time period: {extra_info[1]}\n\nYou are a Bitcoin expert in explaining Bitcoin price changes. Please provide a helpful response based on this information, explaining the price change if necessary. A graph has been generated showing the price change, so please refer to it in your explanation."
    elif intent == "mempool_fees":
        context = f"User query: {query}\n\nIntent: {intent}\n\nRelevant Bitcoin data: Mempool fee distribution data\n\nYou are a Bitcoin expert in explaining mempool fees. Please provide a helpful response based on this information, explaining the fee distribution if necessary. A graph has been generated showing the fee distribution, so please refer to it in your explanation."
    else:
        context = f"User query: {query}\n\nIntent: {intent}\n\nRelevant Bitcoin data: {json.dumps(api_data, indent=2)}\n\nAdditional info: {extra_info}\n\nYou are a Bitcoin expert. Please provide a helpful response based on this information, explaining the data if necessary."

    # Create a new message for streaming
    msg = cl.Message(content="")
    await msg.send()

    # Generate AI response
    system_message = "You are a friendly and knowledgeable AI assistant specializing in Bitcoin and blockchain technology, but also capable of general conversation. You have direct access to mempool api that returns bitcoin data, When asked about Bitcoin, provide accurate and helpful information, explaining technical concepts in an easy-to-understand manner. For other topics, engage in a natural, conversational manner."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": context}
    ]

    try:
        if AI_PROVIDER == "openai":
            stream = await ai_client.chat.completions.create(
                model=AI_MODEL,
                messages=messages,
                stream=True,
                **AI_SETTINGS
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    await msg.stream_token(chunk.choices[0].delta.content)
        elif AI_PROVIDER == "anthropic":
            response = await generate_ai_response(messages)
            await msg.stream_token(response)
        elif AI_PROVIDER == "ollama":
            response = await generate_ai_response(messages)
            await msg.stream_token(response)
        else:
            raise ValueError(f"Unsupported AI provider: {AI_PROVIDER}")

        await msg.send()  # Finalize the message

    except asyncio.TimeoutError:
        logger.error("Request to AI provider timed out")
        await msg.update()  # Clear the existing message content
        await cl.Message(content="I'm sorry, but the request timed out. Please try again later.").send()
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        await msg.update()  # Clear the existing message content
        await cl.Message(content=f"An error occurred while generating the response: {str(e)}").send()

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
    