FROM python:3.6

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
RUN mv object_detection /usr/local/lib/python3.6/site-packages

#RUN mkdir imagenes

# run from working directory, and separate args in the json syntax
CMD ["python", "./main.py"]

#we open the ports
EXPOSE 5000/tcp

# pip uninstall/install opencv-python 
# borrar __init__.py de object detection, 
# añadir carpeta object_detection en site-packages