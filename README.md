## SDS Workshops 25/26

Welcome! This public repository hosts all materials for SDS workshops in AY25/26. You’ll find code, data samples, and resources used during live sessions. Participants are encouraged to watch or star this repo to stay updated as new workshop materials will continue to be added throughout the year.

### What you’ll find
- **All workshop folders** organized by session
- **Runnable notebooks and scripts** used in each workshop
- **Datasets** or links to data required to follow along
- **Per-workshop `README.md`** with specific instructions

## How to navigate
Each workshop has its own directory with a consistent structure:

```text
(SEM X WK X) WORKSHOP_NAME
    ├── DATA_FOLDER
    │   └── DATA_SHEETS
    ├── MAIN_CODE_FILE
    ├── OTHER_SCRIPTS
    ├── requirements.txt
    └── README.md
```

### Example
```text
(SEM 1 WK 5) EDA Workshop
    ├── Data
    │   └── train_data.xlsx
    ├── workshop_code.ipynb
    ├── extra_code1.py
    ├── extra_code2.py
    ├── requirements.txt
    └── README.md
```

## Getting started
1. Clone the repository
   ```bash
   git clone https://github.com/NUS-SDS-Workshops/AY-25-26-Public.git
   cd AY-25-26-Public
   ```
2. Pick a workshop folder and review its `README.md` for any specifics
3. Create and activate a virtual environment (recommended)
   - Using `venv`:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
4. Install dependencies for that workshop
   ```bash
   pip install -r requirements.txt
   ```
5. Open the main notebook or script (e.g., `workshop_code.ipynb`) and run through the cells/steps

### Data access
- Small sample files may be included under each workshop’s `Data/` folder
- If a dataset is too large or restricted, the workshop `README.md` will provide a download link or instructions

## Attribution
- Code and materials in this repository are created by our workshop members
- If you use these materials, please provide appropriate credit to the workshop and the original contributors

## Stay updated
- Star and watch this repository to be notified when new workshop materials land


