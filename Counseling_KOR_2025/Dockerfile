FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ADD . /work
WORKDIR /work

RUN apt-get update
RUN pip install --no-cache-dir -r requirements.txt

CMD sh docker_run_script.sh