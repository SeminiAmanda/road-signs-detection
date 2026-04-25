from pathlib import Path

DATA_FOLDER = "data/archive/Train"   
classes = sorted([f for f in Path(DATA_FOLDER).iterdir() if f.is_dir()])
total   = sum(len(list(c.glob("*.*"))) for c in classes)

print(f"Classes found : {len(classes)}")
print(f"Total images  : {total}")
print("")
print("First 5 classes:")
for c in classes[:5]:
    count = len(list(c.glob("*.*")))
    print(f"  Class {c.name:>3} — {count} images")