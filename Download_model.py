import os
import json
from transformers import AutoConfig

# 1. configs 폴더 생성 (없을 경우)
#os.makedirs('configs', exist_ok=True)

# 2. GPT-2 설정 로드 (기본 gpt2는 약 124M, gpt2-medium은 약 355M 파라미터)
# 사용자의 스크립트가 350m 모델 기준이므로 gpt2-medium이 적절할 수 있습니다.
model_name = "gpt2" 
config = AutoConfig.from_pretrained(model_name)

# 3. JSON 파일로 저장
config_path = os.path.join('configs', 'gpt2_124m.json')
with open(config_path, 'w') as f:
    json.dump(config.to_dict(), f, indent=4)

print(f"설정 파일이 저장되었습니다: {config_path}")