import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from huggingface_hub import snapshot_download

target = r'C:\Users\DELL\Downloads\bearing_detect\datasets\neu_det_raw'
path = snapshot_download(
    repo_id='KeenForgeAI/NEU-DET-corrected',
    repo_type='dataset',
    local_dir=target,
    max_workers=8,
)
print('DONE', path)
