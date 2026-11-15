<img width="960" height="717" alt="ramfish_studio" src="https://github.com/user-attachments/assets/599110d4-f84e-4f6f-b1f8-a4fe5045105e" />


# 🔬 RAMFISH Studio

**RAMFISH Studio** brings the five tools of the RAMFISH Software Suite together in one app that opens in your web browser:

```text
1. Import & normalize → 2. Rough alignment → 3. Fine alignment → 4. Spot counter → 5. Multiplex merge
```

Everything runs on your own computer. Your images are never uploaded anywhere.

---

## 🌱 Installation (one time only)

### Step 1: Install Miniconda

Download and install Miniconda for your operating system from the official documentation:

- **Windows:** Install the Python 3.12 64-bit Miniconda installer. During installation, select **Just Me**. Adding Miniconda to `PATH` is optional; using the Miniconda Prompt is recommended.
- **Linux (x86_64):** Run:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -f ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

- **macOS Apple Silicon (M1/M2/M3/M4):** Install the `MacOSX-arm64` Miniconda package.
- **macOS Intel:** Install the `MacOSX-x86_64` Miniconda package.

Official Miniconda documentation:

https://docs.conda.io/en/latest/

---

### Step 2: Create the RAMFISH Environment

Open **Anaconda Prompt / Miniconda Prompt** on Windows, or a terminal on macOS/Linux.

Run the following commands one at a time:

#### 1. Create the environment

```bash
conda create -n ramfish python=3.12 -y
```

#### 2. Activate the environment

```bash
conda activate ramfish
```

#### 3. Install Java and Bio-Formats support

```bash
conda install -c conda-forge bioformats_jar openjdk maven -y
```

#### 4. Install the app dependencies

```bash
pip install streamlit pandas numpy scipy scikit-image tifffile imagecodecs imageio opencv-python SimpleITK pillow "aicsimageio>=4.14,<5" "numcodecs<0.16.0" scyjava
```

#### 5. Optional: Install NVIDIA GPU acceleration

For NVIDIA GPUs supporting CUDA 12.x:

```bash
pip install "cupy-cuda12x[ctk]"
```

For systems using CUDA 11.x, use:

```bash
pip install "cupy-cuda11x[ctk]"
```

---

### Step 3: Download RAMFISH Studio

On this GitHub page, click the green **Code** button → **Download ZIP**, unzip it into your **Documents** folder and rename the unzipped folder to `RAMFISH_Studio`.

---

## 🚀 Start RAMFISH Studio (every time)

Open **Anaconda Prompt / Miniconda Prompt** on Windows, or a terminal on macOS/Linux, and run:

#### 1. Activate the environment

```bash
conda activate ramfish
```

#### 2. Go to the RAMFISH Studio folder

```bash
cd Documents/RAMFISH_Studio
```

#### 3. Start the app

```bash
streamlit run app.py
```

The app opens in your web browser. Keep the prompt/terminal window open while you work; close it when you are done.

---

## 🧬 Using the app

Work through the five steps in the left menu. Each page tells you exactly what to do.

1. **Import & normalize:** type a project name and load your images, one folder per round (`Round_1`, `Round_2`, …). Name the nuclear stain `DAPI` (or the tissue outline `mask`) and every other image after its gene.
2. **Rough alignment:** drag the sliders until each round overlaps Round 1, then click **Apply Rough Alignment to Round**.
3. **Fine alignment:** click **Run Batch Alignment**.
4. **Spot counter:** click **Run spot counter** to count the mRNA spots of every gene.
5. **Multiplex merge:** pick colors and click **Generate Composite**.

All results are saved in the `Output` folder inside the RAMFISH Studio folder:

```text
Output/<project name>/
├── 1_normalized/
├── 2_rough_aligned/
├── 3_fine_aligned/
├── 4_spots/
└── 5_merge/
```

For a quick test you can run the online version with the two round example datasets provided along with this repository: https://ramfishstudio.streamlit.app/

---

# 💻 Hardware Requirements

| Component | Recommended | High Performance |
|---|---|---|
| CPU | AMD Ryzen 5 5600X / Intel Core i5-12400F | AMD Ryzen Threadripper 9965WX |
| GPU | NVIDIA RTX 3060 | NVIDIA RTX 5080 / RTX 5090 |
| RAM | 32 GB DDR4 | 128 GB DDR5 |
| Storage | SSD recommended | High-speed NVMe SSD |

Tested on Windows 11 and Ubuntu 24.04.

---

## License

MIT License, © 2026 Tirtha Das Banerjee
