import os
import hashlib
from pathlib import Path
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHANGE THIS LINE to match your folder structure
# Common options:
#   "data/Train"
#   "data/archive/Train"
#   "data/GTSRB/Train"
DATA_FOLDER = "data/archive/Train"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ImageFile.LOAD_TRUNCATED_IMAGES = False

# ─────────────────────────────────────────────────
# HELPER — count images in a folder
# ─────────────────────────────────────────────────
def count_images(folder):
    total = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                total += 1
    return total


print()
print("=" * 55)
print("   ROAD SIGN DATA CLEANING — Sri Lanka Project")
print("=" * 55)

before_total = count_images(DATA_FOLDER)
print(f"\n   Images before cleaning : {before_total}")
print()


# ─────────────────────────────────────────────────
# STEP 1 — Remove corrupt / unreadable images
# ─────────────────────────────────────────────────
print("━" * 55)
print("[1/5]  Removing corrupt images...")
print("━" * 55)
corrupt_count = 0

for root, dirs, files in os.walk(DATA_FOLDER):
    for filename in files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(root, filename)
            try:
                img = Image.open(filepath)
                img.verify()            # checks file is not broken
            except Exception as e:
                print(f"   Corrupt → removing: {filepath}")
                os.remove(filepath)
                corrupt_count += 1

print(f"   Removed : {corrupt_count} corrupt images\n")


# ─────────────────────────────────────────────────
# STEP 2 — Remove exact duplicate images
# ─────────────────────────────────────────────────
print("━" * 55)
print("[2/5]  Removing duplicate images...")
print("━" * 55)
seen_hashes  = {}
duplicate_count = 0

for root, dirs, files in os.walk(DATA_FOLDER):
    for filename in files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash in seen_hashes:
                    os.remove(filepath)
                    duplicate_count += 1
                else:
                    seen_hashes[file_hash] = filepath
            except Exception:
                pass

print(f"   Removed : {duplicate_count} duplicate images\n")


# ─────────────────────────────────────────────────
# STEP 3 — Convert grayscale images to RGB
# ─────────────────────────────────────────────────
print("━" * 55)
print("[3/5]  Converting grayscale images to RGB...")
print("━" * 55)
converted_count = 0

for root, dirs, files in os.walk(DATA_FOLDER):
    for filename in files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(root, filename)
            try:
                img = Image.open(filepath)
                if img.mode != "RGB":
                    print(f"   Converting: {filepath}  ({img.mode} → RGB)")
                    img = img.convert("RGB")
                    img.save(filepath)
                    converted_count += 1
            except Exception:
                pass

print(f"   Converted : {converted_count} images to RGB\n")


# ─────────────────────────────────────────────────
# STEP 4 — Remove images that are too small
#          (smaller than 20x20 pixels — unusable)
# ─────────────────────────────────────────────────
print("━" * 55)
print("[4/5]  Removing images that are too small (<20x20)...")
print("━" * 55)
small_count = 0

for root, dirs, files in os.walk(DATA_FOLDER):
    for filename in files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(root, filename)
            try:
                img  = Image.open(filepath)
                w, h = img.size
                if w < 20 or h < 20:
                    print(f"   Too small ({w}x{h}) → removing: {filepath}")
                    os.remove(filepath)
                    small_count += 1
            except Exception:
                pass

print(f"   Removed : {small_count} too-small images\n")


# ─────────────────────────────────────────────────
# STEP 5 — Count images per class + check balance
# ─────────────────────────────────────────────────
print("━" * 55)
print("[5/5]  Analysing class distribution...")
print("━" * 55)

class_counts = Counter()

for class_folder in Path(DATA_FOLDER).iterdir():
    if class_folder.is_dir():
        count = len([
            f for f in class_folder.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])
        class_counts[class_folder.name] = count

# Sort numerically
sorted_classes = sorted(class_counts.keys(),
                        key=lambda x: int(x) if x.isdigit() else 0)
sorted_counts  = [class_counts[k] for k in sorted_classes]

after_total = sum(class_counts.values())
largest     = max(class_counts, key=class_counts.get)
smallest    = min(class_counts, key=class_counts.get)
avg         = after_total // len(class_counts)

# Print per-class table
print(f"\n   {'Class':>6}  {'Images':>7}  {'Bar'}")
print(f"   {'─'*6}  {'─'*7}  {'─'*20}")
for cls in sorted_classes:
    bar = "█" * (class_counts[cls] // 100)
    print(f"   {cls:>6}  {class_counts[cls]:>7}  {bar}")


# ─────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────
print()
print("=" * 55)
print("   CLEANING COMPLETE — SUMMARY")
print("=" * 55)
print(f"   Images before cleaning  : {before_total}")
print(f"   Images after  cleaning  : {after_total}")
print(f"   Images removed total    : {before_total - after_total}")
print(f"   ─────────────────────────────────────")
print(f"   Total classes           : {len(class_counts)}")
print(f"   Largest class  → {largest:>3}   : {class_counts[largest]} images")
print(f"   Smallest class → {smallest:>3}   : {class_counts[smallest]} images")
print(f"   Average per class       : {avg} images")
print(f"   Imbalance ratio         : {class_counts[largest]/class_counts[smallest]:.1f}x")
print("=" * 55)

if class_counts[largest] / class_counts[smallest] > 5:
    print()
    print("   ⚠  HIGH IMBALANCE DETECTED (>5x)")
    print("   ✓  Will be handled in training using")
    print("      WeightedRandomSampler — noted for report")

print()


# ─────────────────────────────────────────────────
# SAVE BAR CHART — include in your report
# ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))

colors = ["#e74c3c" if c == largest
          else "#2ecc71" if c == smallest
          else "steelblue"
          for c in sorted_classes]

bars = ax.bar(sorted_classes, sorted_counts,
              color=colors, edgecolor="white", linewidth=0.4)

ax.axhline(y=avg, color="orange", linestyle="--",
           linewidth=1.2, label=f"Average ({avg})")

ax.set_xlabel("Class (sign type)", fontsize=11)
ax.set_ylabel("Number of images",  fontsize=11)
ax.set_title("Class distribution — GTSRB dataset (after cleaning)\n"
             "Sri Lanka Road Sign Detection Project", fontsize=13)
ax.set_xticks(range(len(sorted_classes)))
ax.set_xticklabels(sorted_classes, rotation=90, fontsize=7)

red_patch   = mpatches.Patch(color="#e74c3c", label=f"Largest class ({largest})")
green_patch = mpatches.Patch(color="#2ecc71", label=f"Smallest class ({smallest})")
blue_patch  = mpatches.Patch(color="steelblue", label="Other classes")
ax.legend(handles=[red_patch, green_patch, blue_patch,
                   plt.Line2D([0],[0], color="orange",
                              linestyle="--", label=f"Average ({avg})")])

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

print("   Chart saved → class_distribution.png")
print("   Screenshot this and add it to your report!\n")