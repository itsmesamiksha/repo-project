# ===================
# Dockerfile
# ===================
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Debug: Ensure the file is actually inside /app
RUN echo "=== LISTING FILES IN /app ===" && ls -l /app



# Copy requirements first for better caching
COPY requirements.txt .
COPY winequality-red.csv .


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy application code
COPY . .

# Train the model during build (optional - can be done at runtime)
RUN python train.py

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "app.py"]

