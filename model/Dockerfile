
FROM ubuntu:20.04 

FROM python:3.8

MAINTAINER Amazon AI <[sage-learner@amazon.com](mailto:sage-learner@amazon.com)>


RUN apt-get update


RUN pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the current directory into the container at /app
COPY . /app/

ENV PATH="/app:${PATH}"

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION


# Expose the port that your Flask app will run on
EXPOSE 8080

ENTRYPOINT ["./serve"]

