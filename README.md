# ship-detection

### Step 1. Clone Repo
``` 
git clone https://github.com/Chaitanyabalivada/ship-detection.git
cd ship-detection

```

### Step 2. Download Dataset
```
mkdir -p data/raw
kaggle datasets download -d siddharthkumarsah/ships-in-aerial-images -p ./data/raw --unzip
```

##### This creates:
```
data/raw/ships-in-aerial-images/train/images/...
data/raw/ships-in-aerial-images/valid/images/...
data/raw/ships-in-aerial-images/test/images/...
```

### Step 4. Train
```
sbatch train.sbatch
```

#### Results are saved in:

```
runs/detect/predict/
```