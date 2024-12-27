# Create a virtual environment (optional)
python3 -m venv donut_env
source donut_env/bin/activate  # On Windows: donut_env\Scripts\activate

# Install necessary packages
pip install transformers datasets torch torchvision huggingface_hub sentencepiece protobuf

