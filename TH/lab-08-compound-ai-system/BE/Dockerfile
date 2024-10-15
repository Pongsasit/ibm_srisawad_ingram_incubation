# Use the official Python image from the Docker Hub
FROM python:3.11.5

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements_venv.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements_venv.txt

# Copy the rest of the application code into the container
COPY . .

# Set the environment variable to ensure the output is sent straight to the terminal (standard out)
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

RUN mkdir -p /app/temp && chown -R root:root /app/temp

# Expose the port that the Flask app runs on
EXPOSE 8080

# Specify the command to run the Flask app
CMD ["python", "app.py"]
