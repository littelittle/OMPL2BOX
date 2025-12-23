def pts2obj(pts, filename="pointcloud.obj"):
    with open(filename, "w") as f:
        for p in pts:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")