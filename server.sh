python -u server.py \
    --port '9000' \
    --max-workers 1 \
    --config 'checkpoint/config.json' \
    --model 'checkpoint/G.pth' \
    --api-prefix '/tts'