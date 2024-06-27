# ₿itcoin Mempool AI Assistant

## Overview

The Bitcoin Mempool AI Assistant is an advanced chatbot designed to provide real-time insights and information about Bitcoin transactions, blockchain data, and market trends. Powered by state-of-the-art language models and integrating with live Bitcoin network data, this assistant offers a unique blend of AI capabilities and blockchain expertise.


![bitcoin-mempool-ai](https://imagedelivery.net/WfhVb8dSNAAvdXUdMfBuPQ/578ff544-27ed-456a-82bc-9c0b22a3d800/public)


## Features

- 📊 Real-time Bitcoin transaction analysis
- 💼 Mempool insights and fee estimation
- 🔍 Blockchain exploration and data retrieval
- 📈 Market trends and price information
- 🧠 General Bitcoin and blockchain knowledge

## Technologies Used

- Python
- Chainlit
- OpenAI GPT models
- Anthropic Claude models
- Ollama (for local model deployment)
- Mempool Space API (no api key needed)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/bigsk1/btc-mempool-ai.git
   cd btc-mempool-ai
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   Create a `.env` file in the project root and add your API keys:
   ```
   # You have the option to use any of the AI Providers below
   
   OPENAI_API_KEY=api_key

   ANTHROPIC_API_KEY=api_key

   AI_PROVIDER=openai  # or anthropic or ollama

   AI_MODEL=gpt-4o     # or your preferred model

   OLLAMA_BASE_URL=http://localhost:11434  #  http://host.docker.internal:11434    
   ```

5. If using Ollama, ensure it's installed and the desired model is pulled before running:
   ```
   ollama pull llama3    # or your preferred model
   ```

## Usage

To start the Bitcoin Mempool AI Assistant:

```
chainlit run app.py -w
```

Navigate to `http://localhost:8000` in your web browser to interact with the assistant.

### Example Queries

- "What's the current Bitcoin price?"
- "Show me the mempool fee distribution for the last 24 hours."
- "Explain the latest difficulty adjustment in Bitcoin mining."
- "What's the transaction history for this Bitcoin address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
- "Graph the Bitcoin price change over the last month."


## Configuration

You can customize the assistant's behavior by modifying the .env file. Change the AI_PROVIDER and AI_MODEL variables to switch between different AI providers and models.



You can customize the assistant's behavior and appearance by modifying the `.chainlit/config.toml` file. Refer to the Chainlit documentation for more details on configuration options.

---


![mem1](https://imagedelivery.net/WfhVb8dSNAAvdXUdMfBuPQ/82844360-af76-42d2-9163-e0ba7fe44e00/public)


---

![mem2](https://imagedelivery.net/WfhVb8dSNAAvdXUdMfBuPQ/51cb8776-46b0-409f-1979-f1b8ce552000/public)

## Contributing

Contributions to the Bitcoin Mempool AI Assistant are welcome! Please feel free to submit pull requests, create issues or spread the word.

## License

This project is licensed under the MIT License

## Acknowledgments

- Bitcoin Mempool Space for their comprehensive API
- The Chainlit team for their excellent chat interface framework
- OpenAI, Anthropic, and the Ollama project for their powerful language models

## Disclaimer

This assistant is for informational purposes only and should not be considered financial advice. Always do your own research and consult with a qualified financial advisor before making any investment decisions.