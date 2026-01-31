# =============================================================================
# Dockerfile for HuggingFace Spaces
# =============================================================================
# Big Data Midterm Project - Superstore Sales Dashboard
# Authors: Lương Minh Tiến (K214162157), Lê Thành Tuân (K214161343)
# =============================================================================

# Use lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create non-root user for security (HuggingFace requirement)
RUN useradd -m -u 1000 user
USER user

# Set HOME for the user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Set working directory for user
WORKDIR $HOME/app

# Copy files to user directory
COPY --chown=user . $HOME/app

# Expose port 7860 (HuggingFace Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false", \
    "--browser.gatherUsageStats=false"]
