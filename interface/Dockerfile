FROM python:3-slim

#set working directory for image
WORKDIR /usr/src/app

# set local non-root user inside docker
ARG local_user=docknonroot
RUN useradd -m $local_user

#install required python packages
# torch is installed separately from requirements.txt for the slimmer cpu-only install
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt
RUN pip install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

#copy code files to image owned by local user
RUN mkdir code
COPY ./interface_code ./code
RUN chown -hR $local_user ./code

# change to local user and run entrypoint
USER $local_user
RUN chmod 755 ./code
ENTRYPOINT ["python", "./code/interface_main.py"]
