# Use Python 3.11
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/app /app

# Copy additional data and model files
COPY data/processed/df_final.csv /app/data/df_final.csv
COPY models/pipe.pkl /app/models/pipe.pkl

# Expose the port that Streamlit will run on
EXPOSE 5000

# Run Streamlit
CMD ["python", "./app.py"]
