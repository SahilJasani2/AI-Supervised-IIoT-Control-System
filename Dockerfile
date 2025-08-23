# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY src/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model and scaler into the container
COPY src/model.pth .
COPY src/scaler.gz .

# Copy ALL scripts into the container
COPY src/publisher.py .
COPY src/subscriber.py .
COPY src/controller.py .

# Run publisher.py by default when the container launches
CMD ["python", "-u", "publisher.py"]
