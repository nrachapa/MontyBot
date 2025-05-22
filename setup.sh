#!/bin/zsh

# Check if .env directory exists
if [ -d ".env" ]; then
    echo "Environment directory '.env' already exists."
    exit 0
fi

echo "Creating Python virtual environment in '.env'..."
python3 -m venv .env

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
fi

echo "Virtual environment created successfully."
source .env/bin/activate

if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    python3 -m pip install --upgrade -r requirements.txt
    echo "Dependencies installed successfully."
else
    echo "requirements.txt not found. Skipping dependency installation."
fi