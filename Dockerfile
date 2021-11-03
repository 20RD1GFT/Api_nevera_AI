FROM python:3.7

#set envionment variables
ENV PYTHONUNBUFFERED 1

# run this before copying requirements for cache efficiency
RUN pip install --upgrade pip

#set work directory early so remaining paths can be relative
WORKDIR /code

# Adding requirements file to current directory
# just this file first to cache the pip install step when code changes
COPY requirements.txt .

#install dependencies
RUN pip install -r requirements.txt

# copy code itself from context to image
COPY /src .

RUN mkdir imagenes

# run from working directory, and separate args in the json syntax
CMD ["python", "./server.py"]

#we open the ports
EXPOSE 5000/tcp