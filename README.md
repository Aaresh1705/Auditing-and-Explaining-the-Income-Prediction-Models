# Auditing-and-Explaining-the-Income-Prediction-Models
Project 1 and 2 of the XAI & UX special course...

## Participants:
```
Albert F.K. Hansen - (AlbertFKHansen)
Kaitlyn Wu Brooks - (KaitlynWuBrooks)
Marcus Olssen - (Aaresh1705)
```

## Version and env control
The repository depends on several PyPi packages, some of which requires older versions of python.
To ensure a smooth experience, local version control and a virtual environment is recommended.

### Linux
Use the automatic installer
```bash
curl -fsSL https://pyenv.run | bash
```
Then make a local version of python, make a virtual environment, and install packages.
```bash
pyenv local 3.11.9
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### MacOS
You can install pyenv via brew
```bash
brew update
brew install pyenv
```
Then make a local version of python, make a virtual environment, and install packages.
```bash
pyenv local 3.11.9
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows
##### Step 1 - Back up your files
Save anything important — documents, photos, that one folder you definitely didn't name `misc_final_FINAL_v3_USE THIS`.

Your Windows activation key does not need to be backed up.

##### Step 2 — Download Linux (Ubuntu is recommended for beginners)
Head to ubuntu.com and download the latest LTS release. It's free. It has always been free. You've been paying for an OS this whole time.

##### Step 3 — Create a bootable USB
Use Balena Etcher or Rufus to flash the ISO to a USB drive.

##### Step 4 — Boot from USB and install
Restart your machine, spam `F12` (or `F2`, or `Del`), select the USB drive, and follow the installer. When it asks what to do with your disk, click Erase and install.

##### Step 5 — Install pyenv
Open a terminal. Notice how fast it opens and how clean everything looks. Follow the Linux guide.

## Repository Structure

```
.
├── adult/                          # Adult income dataset
│   ├── adult.data
│   ├── adult.test
│   ├── adult.names
│   ├── old.adult.names
│   └── Index
├── assignment1/                    # Assignment 1: Stakeholder explanations
│   ├── Project1.ipynb
│   ├── stakeholder_plots.py
│   ├── stakeholder_plots_variants.py
│   ├── plot_applicant_nn.png
│   ├── plot_applicant_nn_no_sensitive_no_country.png
│   ├── plot_applicant_xgb.png
│   ├── plot_applicant_xgb_no_sensitive_no_country.png
│   ├── plot_data_scientist.png
│   └── plot_director.png
├── assignment2_outputs/            # Assignment 2: Model audit outputs
│   ├── xgb_audit_metrics.csv
│   ├── xgb_concept_probe_results.csv
│   ├── xgb_leaf_embeddings.npy
│   ├── xgb_tsne_coordinates.csv
│   ├── xgb_tsne_age_group.png
│   ├── xgb_tsne_income.png
│   ├── xgb_tsne_race.png
│   └── xgb_tsne_sex.png
├── models/                         # Trained models
│   ├── nn_model.keras
│   ├── nn_no_sensitive.keras
│   ├── nn_no_sensitive_no_country.keras
│   ├── xgb_model.json
│   ├── xgb_no_sensitive.json
│   └── xgb_no_sensitive_no_country.json
├── assignment2_xgb_audit.py        # XGBoost audit script
├── dataloader.py                   # Data loading utilities
├── requirements.txt
└── README.md
```
