# Use official Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model file and prediction script
COPY california_housing_lr_model.pkl .
COPY predict.py .

# Run the prediction script
CMD ["python", "predict.py"]
