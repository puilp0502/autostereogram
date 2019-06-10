# Autostereogram encoding & decoding
## Cloning
1. (If not installed) Install Git-LFS  

On Debian/Ubuntu:
```bash
sudo apt install git-lfs
```
On macOS:
```bash
brew install git-lfs
```

2. Clone
```bash
git clone https://github.com/puilp0502/autostereogram.git
```

3. Fetch LFS objects
```
cd autostereogram && git lfs fetch --all
```

## Getting started

0. (Optional, recommended) Create virtualenv
```bash
python -m virtualenv -p python3 venv
source venv/bin/activate
```

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run Real-time video decoding example:
```bash
python mevid.py
```
