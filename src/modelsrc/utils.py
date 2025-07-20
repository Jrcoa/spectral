import os

def get_latest_checkpoint_epoch(checkpoint_dir: str, prefix="model_epoch_") -> int:
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith(prefix) and f.endswith(".pt")
    ]
    if not checkpoints:
        return -1 

    def extract_epoch(fname):
        try:
            return int(fname.replace(prefix, "").replace(".pt", ""))
        except ValueError:
            return -1

    epochs = [extract_epoch(f) for f in checkpoints]
    return max(epochs)