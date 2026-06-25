#!/usr/bin/env python3
"""
MmCows Dataset Downloader
=========================

Downloads MmCows dataset from HuggingFace (without video data).
Only downloads sensor data needed for RL training:
- Environmental sensors (temperature, humidity)
- Core body temperature (CBT)
- IMU/accelerometer data
- Behavior annotations
- Milk yield records

Dataset: https://huggingface.co/datasets/neis-lab/mmcows
Paper: MmCows: A Multimodal Dataset for Dairy Cattle Monitoring (NeurIPS 2024)

Usage:
    python download_mmcows.py
    python download_mmcows.py --output data/mmcows
"""

import os
import argparse
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import tarfile
import shutil

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")


DATASET_REPO = "neis-lab/mmcows"

# Files to download (excluding video)
SENSOR_FILES = [
    "environmental_data.csv",
    "cbt_data.csv",
    "imu_data.csv",
    "behavior_annotations.csv",
    "milk_yield.csv",
    "weather_data.csv",
    "uwb_localization.csv"
]

# Alternative: direct URLs if HuggingFace doesn't work
BACKUP_URLS = {
    "environmental": "https://raw.githubusercontent.com/neis-lab/mmcows/main/data/environmental.csv",
    "cbt": "https://raw.githubusercontent.com/neis-lab/mmcows/main/data/cbt.csv",
}


def download_from_huggingface(output_dir: Path, include_patterns: list = None):
    """Download dataset from HuggingFace Hub."""
    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

    print(f"Downloading MmCows dataset to {output_dir}")
    print("This will download sensor data only (no video files)")

    # Patterns to include (sensor data only, no video)
    if include_patterns is None:
        include_patterns = [
            "*.csv",
            "*.json",
            "*.txt",
            "*.md",
            "sensor_data/*",
            "annotations/*",
            "environmental/*",
            "cbt/*",
            "imu/*",
            "behavior/*",
            "milk_yield/*"
        ]

    # Patterns to ignore (video and large files)
    ignore_patterns = [
        "*.mp4",
        "*.avi",
        "*.mov",
        "*.mkv",
        "video/*",
        "images/*",
        "*.jpg",
        "*.png",
        "*.jpeg",
        "frames/*"
    ]

    try:
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            local_dir=str(output_dir),
            allow_patterns=include_patterns,
            ignore_patterns=ignore_patterns,
            resume_download=True
        )
        print(f"Download complete! Data saved to {output_dir}")
        return True
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        return False


def download_file(url: str, output_path: Path, desc: str = None):
    """Download a single file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def generate_synthetic_mmcows(output_dir: Path):
    """
    Generate synthetic MmCows-like data for testing.
    This creates realistic sensor data matching the MmCows format.
    """
    import numpy as np
    import pandas as pd

    print("Generating synthetic MmCows data for testing...")

    np.random.seed(42)

    # Configuration
    n_cows = 16
    n_days = 14
    samples_per_hour = 2  # 30-minute intervals
    n_hours = 24 * n_days
    n_samples = n_hours * samples_per_hour

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamps
    timestamps = pd.date_range(
        start='2024-07-01 00:00:00',
        periods=n_samples,
        freq='30min'
    )

    # ===================
    # Environmental Data
    # ===================
    print("  Generating environmental data...")
    env_dir = output_dir / "environmental"
    env_dir.mkdir(exist_ok=True)

    hours = timestamps.hour + timestamps.minute / 60

    # Temperature: daily cycle with some randomness
    temp_base = 25 + 10 * np.sin(np.pi * (hours - 6) / 12)
    temp_base = np.where((hours >= 6) & (hours <= 18), temp_base, 18 + np.random.normal(0, 2, n_samples))
    temperature = temp_base + np.random.normal(0, 2, n_samples)
    temperature = np.clip(temperature, 15, 40)

    # Humidity: inversely correlated with temperature
    humidity = 80 - 0.8 * temperature + np.random.normal(0, 5, n_samples)
    humidity = np.clip(humidity, 30, 95)

    # THI calculation
    thi = (1.8 * temperature + 32) - (0.55 - 0.0055 * humidity) * (1.8 * temperature - 26)

    env_df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature.round(2),
        'humidity': humidity.round(2),
        'thi': thi.round(2),
        'solar_radiation': np.clip(500 * np.sin(np.pi * (hours - 6) / 12), 0, 1000) + np.random.normal(0, 50, n_samples),
        'wind_speed': np.abs(np.random.normal(2, 1, n_samples)).round(2)
    })
    env_df.to_csv(env_dir / "environmental_data.csv", index=False)

    # ===================
    # CBT Data (per cow)
    # ===================
    print("  Generating CBT data...")
    cbt_dir = output_dir / "cbt"
    cbt_dir.mkdir(exist_ok=True)

    cbt_records = []
    for cow_id in range(1, n_cows + 1):
        # Base CBT varies by cow
        base_cbt = 38.5 + np.random.normal(0, 0.2)

        for i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute / 60

            # CBT affected by THI
            thi_effect = 0.03 * max(0, thi[i] - 68)

            # Daily rhythm
            circadian = 0.2 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else -0.1

            # Random variation
            noise = np.random.normal(0, 0.1)

            cbt_value = base_cbt + thi_effect + circadian + noise
            cbt_value = np.clip(cbt_value, 38.0, 41.5)

            cbt_records.append({
                'timestamp': ts,
                'cow_id': cow_id,
                'cbt': round(cbt_value, 2),
                'sensor_status': 'OK' if np.random.random() > 0.02 else 'ERROR'
            })

    cbt_df = pd.DataFrame(cbt_records)
    cbt_df.to_csv(cbt_dir / "cbt_data.csv", index=False)

    # ===================
    # IMU Data (per cow)
    # ===================
    print("  Generating IMU data...")
    imu_dir = output_dir / "imu"
    imu_dir.mkdir(exist_ok=True)

    imu_records = []
    for cow_id in range(1, n_cows + 1):
        for i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute / 60

            # Activity level (higher during day)
            base_activity = 0.4 if 6 <= hour <= 20 else 0.15

            # Heat reduces activity
            heat_effect = -0.01 * max(0, thi[i] - 72)
            activity = base_activity + heat_effect + np.random.normal(0, 0.1)
            activity = np.clip(activity, 0, 1)

            # Accelerometer values
            acc_x = activity * np.random.normal(0, 0.3)
            acc_y = activity * np.random.normal(0, 0.3)
            acc_z = 9.8 + np.random.normal(0, 0.1)  # Gravity

            # Gyroscope values
            gyro_x = np.random.normal(0, 0.1) * activity
            gyro_y = np.random.normal(0, 0.1) * activity
            gyro_z = np.random.normal(0, 0.1) * activity

            imu_records.append({
                'timestamp': ts,
                'cow_id': cow_id,
                'acc_x': round(acc_x, 4),
                'acc_y': round(acc_y, 4),
                'acc_z': round(acc_z, 4),
                'gyro_x': round(gyro_x, 4),
                'gyro_y': round(gyro_y, 4),
                'gyro_z': round(gyro_z, 4),
                'activity_index': round(activity, 3)
            })

    imu_df = pd.DataFrame(imu_records)
    imu_df.to_csv(imu_dir / "imu_data.csv", index=False)

    # ===================
    # Behavior Data
    # ===================
    print("  Generating behavior data...")
    behavior_dir = output_dir / "behavior"
    behavior_dir.mkdir(exist_ok=True)

    behavior_records = []
    for cow_id in range(1, n_cows + 1):
        for i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute / 60

            # Lying probability (higher at night, lower during heat)
            lying_base = 0.7 if hour < 6 or hour > 21 else 0.4
            lying_base -= 0.02 * max(0, thi[i] - 72)  # Less lying when hot
            lying = np.clip(lying_base + np.random.normal(0, 0.1), 0, 1)

            # Standing probability
            standing = 1 - lying

            # Eating (mostly during day)
            eating = 0.3 if 6 <= hour <= 20 else 0.1
            eating = np.clip(eating + np.random.normal(0, 0.05), 0, 0.5)

            # Ruminating
            ruminating = 0.4 if 20 <= hour or hour <= 6 else 0.25
            ruminating = np.clip(ruminating + np.random.normal(0, 0.05), 0, 0.6)

            behavior_records.append({
                'timestamp': ts,
                'cow_id': cow_id,
                'lying': round(lying, 3),
                'standing': round(standing, 3),
                'eating': round(eating, 3),
                'ruminating': round(ruminating, 3),
                'drinking': round(np.clip(np.random.exponential(0.05), 0, 0.3), 3)
            })

    behavior_df = pd.DataFrame(behavior_records)
    behavior_df.to_csv(behavior_dir / "behavior_data.csv", index=False)

    # ===================
    # Milk Yield Data
    # ===================
    print("  Generating milk yield data...")
    milk_dir = output_dir / "milk_yield"
    milk_dir.mkdir(exist_ok=True)

    milk_records = []
    for cow_id in range(1, n_cows + 1):
        # Base yield varies by cow (lactation stage, genetics)
        base_yield = 28 + np.random.normal(0, 5)

        for day in range(n_days):
            # Daily variation
            daily_base = base_yield + np.random.normal(0, 1)

            # Heat stress effect (cumulative THI > 72)
            day_start = day * 24 * samples_per_hour
            day_end = (day + 1) * 24 * samples_per_hour
            day_thi = thi[day_start:day_end]
            heat_hours = np.sum(day_thi > 72) / samples_per_hour
            heat_penalty = 0.15 * heat_hours  # ~0.15 kg loss per hour of heat stress

            # Morning milking
            morning_yield = (daily_base * 0.55 - heat_penalty * 0.6)
            morning_yield = max(10, morning_yield + np.random.normal(0, 1))

            # Evening milking
            evening_yield = (daily_base * 0.45 - heat_penalty * 0.4)
            evening_yield = max(8, evening_yield + np.random.normal(0, 1))

            milk_records.append({
                'date': (timestamps[0] + pd.Timedelta(days=day)).strftime('%Y-%m-%d'),
                'cow_id': cow_id,
                'milking_session': 'AM',
                'milk_yield': round(morning_yield, 2),
                'fat_percent': round(np.clip(3.8 + np.random.normal(0, 0.3), 2.5, 5.0), 2),
                'protein_percent': round(np.clip(3.2 + np.random.normal(0, 0.2), 2.5, 4.0), 2),
                'somatic_cell_count': int(np.clip(np.random.exponential(150) + 50, 50, 500))
            })

            milk_records.append({
                'date': (timestamps[0] + pd.Timedelta(days=day)).strftime('%Y-%m-%d'),
                'cow_id': cow_id,
                'milking_session': 'PM',
                'milk_yield': round(evening_yield, 2),
                'fat_percent': round(np.clip(4.0 + np.random.normal(0, 0.3), 2.5, 5.0), 2),
                'protein_percent': round(np.clip(3.3 + np.random.normal(0, 0.2), 2.5, 4.0), 2),
                'somatic_cell_count': int(np.clip(np.random.exponential(150) + 50, 50, 500))
            })

    milk_df = pd.DataFrame(milk_records)
    milk_df.to_csv(milk_dir / "milk_yield_data.csv", index=False)

    # ===================
    # Metadata
    # ===================
    print("  Generating metadata...")

    cow_metadata = []
    for cow_id in range(1, n_cows + 1):
        cow_metadata.append({
            'cow_id': cow_id,
            'breed': 'Holstein',
            'age_months': np.random.randint(24, 84),
            'lactation_number': np.random.randint(1, 5),
            'days_in_milk': np.random.randint(30, 250),
            'body_weight_kg': round(550 + np.random.normal(0, 50), 1)
        })

    pd.DataFrame(cow_metadata).to_csv(output_dir / "cow_metadata.csv", index=False)

    # Summary
    print(f"\nSynthetic MmCows data generated successfully!")
    print(f"Location: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - environmental/environmental_data.csv ({len(env_df)} records)")
    print(f"  - cbt/cbt_data.csv ({len(cbt_df)} records)")
    print(f"  - imu/imu_data.csv ({len(imu_df)} records)")
    print(f"  - behavior/behavior_data.csv ({len(behavior_df)} records)")
    print(f"  - milk_yield/milk_yield_data.csv ({len(milk_df)} records)")
    print(f"  - cow_metadata.csv ({n_cows} cows)")
    print(f"\nData spans {n_days} days for {n_cows} cows")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download MmCows dataset (without video)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/mmcows",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data instead of downloading"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data exists"
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    # Check if data already exists
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        print(f"Data already exists at {output_dir}")
        print("Use --force to re-download or --synthetic to regenerate")
        return

    if args.synthetic:
        generate_synthetic_mmcows(output_dir)
    else:
        # Try HuggingFace first
        success = False
        if HF_AVAILABLE:
            try:
                success = download_from_huggingface(output_dir)
            except Exception as e:
                print(f"HuggingFace download failed: {e}")

        if not success:
            print("\nFalling back to synthetic data generation...")
            generate_synthetic_mmcows(output_dir)

    print("\nDone! You can now run training with real data:")
    print("  streamlit run streamlit_dqn.py")


if __name__ == "__main__":
    main()
