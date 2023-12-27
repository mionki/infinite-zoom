# Use the official Python image as the base image
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the current directory into the container at /app
COPY . /app/

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# Expose the port that your Flask app will run on
EXPOSE 5000

# Define the command to run your Flask application
CMD ["python", "app.py"]
