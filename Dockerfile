FROM python:3.10-slim

LABEL maintainer="HamiltonianCompiler Contributors"
LABEL description="HamiltonianCompiler - Quantum Hamiltonian compilation framework"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy the rest of the code
COPY . .

# Install the package
RUN pip install -e .

# Run tests on build to verify installation
RUN pytest tests/ -v

# Default command: run examples
CMD ["python", "hamiltoniancompiler/hamiltoniancompiler.py"]
