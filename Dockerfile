FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the environment variable for the service account key path
# The GitHub Actions workflow will place the key at this location.
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/gcp-key.json"

# Make port 8080 available to the world outside this container
# Cloud Run expects the container to listen on this port
ENV PORT=8080

# It tells Gunicorn to find the 'app' object inside the 'run.py' file.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "run:app"]
