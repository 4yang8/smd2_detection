/home/sjyang114/miniconda3/envs/NLP/bin/vllm serve /home/sjyang114/Counseling_KOR_2025/models/exaone_110_maxturn10/merged \
  --host 0.0.0.0 --port 9000 \
  --served-model-name counseling \
  --dtype auto \
  --trust-remote-code \

