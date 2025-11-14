import os
import re
import pandas as pd
from tqdm import tqdm

# 입력 / 출력 폴더 경로
INPUT_DIR = "/root/ros2_ws/src/physical_ai_tools/docker/huggingface/lerobot/PA/ffw_sg2_rev1_pick_n_place/data/chunk-000"
OUTPUT_DIR = "/root/ros2_ws/src/physical_ai_tools/docker/huggingface/lerobot/PA/ffw_sg2_rev1_pick_n_place/data/chunk-001"

# 출력 폴더가 없으면 자동 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# episode_000000.parquet 형태의 파일만 매칭
pattern = re.compile(r"episode_(\d+)\.parquet$")

for filename in tqdm(os.listdir(INPUT_DIR)):
    match = pattern.match(filename)
    if not match:
        continue  # 형식이 다르면 건너뛰기

    # 파일명에서 숫자 추출 후 int로 변환 (앞의 0 제거)
    episode_num = int(match.group(1))
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        # parquet 읽기
        df = pd.read_parquet(input_path)

        # episode_index 업데이트 (없으면 생성)
        df["episode_index"] = episode_num

        # 출력 폴더에 동일 이름으로 저장
        df.to_parquet(output_path, index=False)

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")

print(f"✅ 모든 parquet 파일이 '{OUTPUT_DIR}'에 저장되었습니다.")
