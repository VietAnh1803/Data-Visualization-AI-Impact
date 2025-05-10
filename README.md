# Data Visualization: AI Impact Analysis

## Overview

This project analyzes the Global AI Content Impact Dataset to understand the impact and trends of AI-generated content. The analysis includes statistical analysis, data visualization, and exploratory data analysis (EDA).

## Features

- 📊 Statistical analysis
- 📈 Data visualization
- 🔍 Exploratory Data Analysis (EDA)
- 📉 Distribution analysis
- 🔗 Correlation analysis
- 📊 PCA analysis
- 📈 Outlier detection

## Requirements

```
pandas>=1.5.0
matplotlib>=3.5.0
seaborn>=0.12.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/VietAnh1803/Data-Visualization-AI-Impact.git
cd Data-Visualization-AI-Impact
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the analysis script:

```bash
python analyze_data.py
```

The script will generate:

1. Analysis report in the terminal
2. Visualization files:
   - `enhanced_distributions.png`: Distribution analysis
   - `correlation_matrix.png`: Correlation heatmap
   - `box_plots.png`: Box plot analysis
   - `pca_analysis.png`: PCA analysis
   - `[column_name]_distribution.png`: Categorical distributions

## Project Structure

```
Data-Visualization-AI-Impact/
├── analyze_data.py          # Main analysis script
├── requirements.txt         # Project dependencies
├── README.md               # Project documentation
└── data/                   # Data directory
    └── Global_AI_Content_Impact_Dataset.csv
```

## Author

- **Viet Anh** - [GitHub](https://github.com/VietAnh1803)

## License

This project is licensed under the MIT License.
