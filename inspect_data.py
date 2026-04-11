import os

# Inspect the annotation TSV
tsv_path = r"C:\Users\Temiloluwa\Downloads\archive (1)\tvsum_dataset\ydata-tvsum50-data\data\ydata-tvsum50-anno.tsv"
print("=== TVSum annotation TSV (first 5 lines) ===")
with open(tsv_path, 'r') as f:
    for i, line in enumerate(f):
        if i < 5:
            parts = line.strip().split('\t')
            print(f"  Line {i}: video_id={parts[0]}, category={parts[1]}, scores_len={len(parts[2].split(','))}")
        else:
            break

# Count unique videos
print("\n=== Counting unique videos ===")
video_ids = set()
with open(tsv_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        video_ids.add(parts[0])
print(f"  Unique video IDs: {len(video_ids)}")
print(f"  Sample IDs: {list(video_ids)[:5]}")

# Check corresponding video files
video_dir = r"C:\Users\Temiloluwa\Downloads\archive (1)\tvsum_dataset\ydata-tvsum50-video\video"
video_files = [f.replace('.mp4', '') for f in os.listdir(video_dir) if f.endswith('.mp4')]
print(f"\n  Video files found: {len(video_files)}")
print(f"  Sample: {video_files[:5]}")

# Check overlap
overlap = video_ids.intersection(set(video_files))
print(f"  Matching video_id <-> video_file: {len(overlap)}")
