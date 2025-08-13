FROM tensorflow/tensorflow:2.16.1-gpu

ARG USER_ID
ARG GROUP_ID

RUN groupadd -r -g $GROUP_ID myuser && useradd -r -u $USER_ID -g myuser -m -d /home/myuser myuser
ENV SHELL /bin/bash

RUN mkdir -p /home/myuser/code && chown -R myuser:myuser /home/myuser/code

WORKDIR /home/myuser/code

RUN apt update
RUN apt install -y jq

RUN pip install aeon==1.0.0
RUN pip install keras==3.6.0
RUN pip install hydra-core==1.3.2
RUN pip install omegaconf==2.3.0
RUN pip install pandas==2.2.0
RUN pip install matplotlib==3.10.0
RUN pip install numba==0.60.0
RUN pip install black
RUN pip install flake8
RUN pip install mypy
RUN pip install pytest
RUN pip install pytest-cov
RUN pip install pytest-xdist
RUN pip install pytest-timeout
RUN pip install pytest-rerunfailures
RUN pip install pre-commit
