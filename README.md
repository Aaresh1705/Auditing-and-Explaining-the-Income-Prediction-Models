# Auditing-and-Explaining-the-Income-Prediction-Models
Project 1 and 2 of the XAI & UX special course...

### Participants:
```
Albert F.K. Hansen - (AlbertFKHansen)
Kaitlyn Wu Brooks - (KaitlynWuBrooks)
Marcus Olssen - (Aaresh1705)
```


### Repository Structure

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
