import tarfile

tar_path = "data/swim_ir_v1.tar.gz"
extract_path = "data/swim_ir_v1"

with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=extract_path)