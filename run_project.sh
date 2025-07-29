#!/bin/bash

VENV_DIR=".venv"

# Step 1: Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Step 2: Activate virtual environment
source "$VENV_DIR/bin/activate"

# Step 3: Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Run the main Python script
echo "Running main script..."
python scripts/main.py
