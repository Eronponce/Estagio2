# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container to /code/WDYP
WORKDIR /code/WDYP

# Copy the current directory contents into the container at /code/WDYP
ADD . /code/WDYP

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run manage.py when the container launches
CMD ["sh", "-c", "python manage.py makemigrations && python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]
