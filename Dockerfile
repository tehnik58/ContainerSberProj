FROM nginx/unit:1.23.0-python3.9

# Our Debian with Python and Nginx for python apps.
# See https://hub.docker.com/r/nginx/unit/

COPY ./config/config.json /docker-entrypoint.d/config.json

RUN mkdir build

# We create folder named build for our app.

COPY . ./build

# We copy our app folder to the /build

RUN apt update && apt install -y python3-pip                                  
#RUN pip3 install -r /build/requirements.txt 
RUN pip3 install fastapi
RUN pip3 install 'uvicorn[standard]'
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118                            
RUN apt remove -y python3-pip                                              
RUN apt autoremove --purge -y                                              
RUN rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/*.list

EXPOSE 80
