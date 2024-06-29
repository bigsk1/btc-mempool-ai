# ₿itcoin Mempool AI Assistant

## Overview

The Bitcoin Mempool AI Assistant is an advanced chatbot designed to provide real-time insights and information about Bitcoin transactions, blockchain data, and market trends. Powered by state-of-the-art language models and integrating with live Bitcoin network data, this assistant offers a unique blend of AI capabilities and blockchain expertise.


![bitcoin-mempool-ai](https://imagedelivery.net/WfhVb8dSNAAvdXUdMfBuPQ/2bd334cd-3e65-4d93-98ec-2d68f8f90a00/public)


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


## GPT from OpenAI

I also made a custom GPT from openai's gpt store, which has similar functionalities and api calls to mempool, it can be found here  [https://chatgpt.com/g/g-fhvMWr0qz-btc-mempool-ai](https://chatgpt.com/g/g-fhvMWr0qz-btc-mempool-ai)



## Docker Setup and Usage

This project uses Docker to ensure consistent environments and easy deployment. Follow these steps to get started with Docker:

### Prerequisites

- Docker installed on your system
- Docker Compose installed on your system

### Building the Docker Image

To build the Docker image for the Bitcoin Mempool AI Assistant:

```bash
docker-compose build
```

### Running the Container

To start the container:

```bash
docker-compose up -d
```

The `-d` flag runs the container in detached mode.

### Viewing Logs

To view the logs of the running container:

```bash
docker-compose logs -f
```

### Stopping the Container

To stop the running container:

```bash
docker-compose down
```

### Changing AI Provider

You can easily switch between different AI providers (OpenAI, Anthropic, Ollama) without rebuilding the container. Use the provided script:

1. Make the script executable (first time only):
   ```bash
   chmod +x switch_provider.sh
   ```

2. Run the script to switch providers:
   ```bash
   ./switch_provider.sh [openai|anthropic|ollama] [model_name]
   ```

   Example:
   ```bash
   ./switch_provider.sh openai gpt-4
   ```

   Or use the default model for a provider:
   ```bash
   ./switch_provider.sh anthropic
   ```

### Environment Variables

The following environment variables can be set in your `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `OLLAMA_BASE_URL`: URL for Ollama (default: http://localhost:11434)
- `AI_PROVIDER`: The AI provider to use (openai, anthropic, or ollama)
- `AI_MODEL`: The specific model to use with the chosen provider

### Accessing the Application

Once the container is running, you can access the Bitcoin Mempool AI Assistant at:

```
http://localhost:8000
```

### Troubleshooting

If you encounter any issues:

1. Ensure all required environment variables are set in your `.env` file.
2. Try rebuilding the image without cache:
   ```bash
   docker-compose build --no-cache
   ```
3. Check the logs for any error messages:
   ```bash
   docker-compose logs -f
   ```

For more detailed information about the project structure and API usage, refer to the sections above in this README.



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