VLLM_DISABLE_COMPILE=1 PYTHONNOUSERSITE=1 python -m uvicorn Counseling_KOR_2025.main:app \
  --host 0.0.0.0 --port 8000 \
  --ssl-certfile=/home/sjyang114/ssl/ssl.crt \
  --ssl-keyfile=/home/sjyang114/ssl/ssl.key.pem
# 실행: /home/sjyang114/Counseling_KOR_2025/main.sh