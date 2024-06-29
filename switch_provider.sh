#!/bin/bash


# Run it like this: ./switch_provider.sh openai gpt-4 or ./switch_provider.sh anthropic

# This script updates the .env file and restarts the container, making it easy to switch providers and models quickly.


# Check if provider is supplied
if [ -z "$1" ]; then
    echo "Usage: ./switch_ai_provider.sh [openai|anthropic|ollama] [model_name]"
    exit 1
fi

# Set the provider
provider=$1

# Set the model (use default if not provided)
model=${2:-default_model}

# Define default models for each provider
declare -A default_models
default_models[openai]="gpt-4o"
default_models[anthropic]="claude-3-sonnet-20240229"
default_models[ollama]="llama3"

# Use default model if 'default_model' is still set
if [ "$model" = "default_model" ]; then
    model=${default_models[$provider]}
fi

# Update the .env file
sed -i "s/AI_PROVIDER=.*/AI_PROVIDER=$provider/" .env
sed -i "s/AI_MODEL=.*/AI_MODEL=$model/" .env

# Restart the container
docker-compose down
docker-compose up -d

echo "Switched to $provider using model $model"