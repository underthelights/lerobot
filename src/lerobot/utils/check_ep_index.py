import os
import re
import pandas as pd
from tqdm import tqdm

# 새로 저장된 parquet 파일들이 있는 폴더
CHECK_DIR = "/root/ros2_ws/src/physical_ai_tools/docker/huggingface/lerobot/PA/ffw_sg2_rev1_pick_n_place/data/chunk-001"

pattern = re.compile(r"episode_(\d+)\.parquet$")

mismatch_files = []

for filename in tqdm(os.listdir(CHECK_DIR)):
    match = pattern.match(filename)
    if not match:
        continue

    episode_num = int(match.group(1))
    file_path = os.path.join(CHECK_DIR, filename)

    try:
        df = pd.read_parquet(file_path)
        if "episode_index" not in df.columns:
            mismatch_files.append((filename, "❌ no 'episode_index' column"))
            continue

        unique_vals = df["episode_index"].unique()
        if len(unique_vals) != 1 or unique_vals[0] != episode_num:
            mismatch_files.append((filename, f"❌ mismatch: found {unique_vals.tolist()} expected {episode_num}"))

    except Exception as e:
        mismatch_files.append((filename, f"⚠️ error reading file: {e}"))

# 결과 출력
if not mismatch_files:
    print("✅ 모든 파일의 episode_index가 파일명과 일치합니다!")
else:
    print("🚨 불일치 파일 발견:")
    for fname, msg in mismatch_files:
        print(f"  {fname}: {msg}")
