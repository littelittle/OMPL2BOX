import csv

def load_path(csv_path):
    joint_cols = [f"panda-panda_joint{i}" for i in range(1, 8)]
    path = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = [float(row[col]) for col in joint_cols]
            path.append((q, None))

    return path

