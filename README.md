

https://github.com/user-attachments/assets/77bb9daf-1a8e-4c43-90a9-7923419bbdc8

Video created with the help of Google's Gemini. 


# 🔬 RAMFISH Studio

**RAMFISH Studio** brings the five tools of the RAMFISH Software Suite together in one app that opens in your web browser:

```text
1. Import & normalize → 2. Rough alignment → 3. Fine alignment → 4. Spot counter → 5. Multiplex merge
```

Everything runs on your own computer. Your images are never uploaded anywhere.

Read the detailed manual for installation and file structure used by the Studio. File: RAMFISH_Studio_manual.pdf

---
##  📽️Video tutorial


https://github.com/user-attachments/assets/6e29fea9-a44c-4f2a-ab2f-219cba0840b0

Video created with the help of Anthropic's Claude.


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
pip install streamlit pandas numpy scipy scikit-image tifffile imagecodecs imageio opencv-python SimpleITK pillow "aicsimageio>=4.14,<5" "numcodecs<0.16.0" scyjava plotly
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

1. **Import & normalize:** type a project name and load your microscope files. A file with several channels is named after its round and its genes, each gene followed by its dye, for example `round1_dpp_alexa546__optix_alexa647__omb_alexa488__hh_alexa514.oir`; the app matches every gene to the right channel by its dye. Single-channel images are named after their gene, the nuclear stain `DAPI`, or the tissue outline `mask`. Without DAPI, the app makes a tissue mask for every round. For thick 3D samples, choose *Same depth in every round* and the app finds the depth of Round 1 in every other round.
2. **Rough alignment:** drag the sliders until each round overlaps Round 1, then click **Apply Rough Alignment to Round**.
   - **3D stacks** (thick samples such as a zebrafish brain): step 2 becomes a 3D aligner. Each round's volume is placed onto Round 1's (up/down, left/right, front/back, turn, tilt, scale) in a 3D view of the tissue outlines; then every depth that all rounds cover becomes a small flat project that goes through steps 3 to 5, each of which gets a Z depth slider. Thin samples such as wings keep the flat aligner.
3. **Fine alignment:** click **Run Batch Alignment**.
4. **Spot counter:** click **Run spot counter** to count the mRNA spots of every gene.
5. **Multiplex merge:** pick colors and click **Generate Composite**.

All results are saved in the `Output` folder inside the RAMFISH Studio folder:

```text
Output/<project name>/
├── 1_import_and_normalize/
│   ├── raw/          every channel as read from the microscope (all Z planes, 16-bit)
│   ├── normalized/   the 2D images used by the next steps
│   └── exports/      copies in the chosen format (Z planes, projection, slab, mask)
├── sections/         3D samples only: one small project per z-plane (S1_6p38um/, S2_9p57um/, ...)
├── 2_rough_aligned/
├── 3_fine_aligned/
├── 4_spots/
└── 5_merge/
```

For a quick test, choose **Example data** under *1. Locate Data* on the first page: two rounds of a butterfly wing (`Rawfile_Example/rawdata.zip`, 87 MB) come with the app and load with one click, so every step can be tried without your own files. The same example also runs in the online version: https://ramfishstudio.streamlit.app/

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
