# Use a specific Python version
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy requirements file into the container
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your local repository into the container
COPY test.py /app

# Install PyTorch Geometric (customized to your version)
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.2+cpu.html

# Replace <TORCH_VERSION> and <CUDA_VERSION> with the correct values
