CUDA_VISIBLE_DEVICES=0 uvicorn app:app\
 --host 0.0.0.0 --port 3457 \
 --loop uvloop \
 --http h11 \
 --ssl-certfile=/home/sjyang114/ssl/ssl.crt \
 --ssl-keyfile=/home/sjyang114/ssl/ssl.key.pem