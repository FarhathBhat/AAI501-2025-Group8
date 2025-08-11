# Diabetes Onset Prediction Project

## Overview
This repository contains the code and resources for a machine learning project aimed at predicting diabetes onset and identifying patient risk profiles using the PIMA Indians Diabetes Dataset. The dataset, originally with 768 samples, has been augmented to 1100 samples with synthetic data to enhance robustness. The project implements classification (decision trees, k-NN, XGBoost) and clustering (k-means) algorithms, aligning with Intro to AI course objectives. The work was collaboratively developed by a team of three, with results designed to be interpretable for medical applications.

## Project Purpose and Goals
- Predict diabetes onset using supervised learning techniques.
- Identify high- and low-risk patient groups through unsupervised clustering.
- Provide a prediction pipeline for real-time use with new patient data.
- Demonstrate practical AI applications in healthcare, focusing on interpretability.

## Files
- `diabetes_prediction.ipynb`: The main Jupyter Notebook containing the project code, including data preprocessing, model training, evaluation, and prediction pipeline.
- `pima_diabetes_combined.csv`: The merged dataset (1100 samples) with original and synthetic data.
- `scaler.pkl`, `xgb_model.pkl`, `kmeans_model.pkl`: Saved model and scaler files for the prediction pipeline.

## Installation
To run the project, ensure you have the following dependencies installed:
- Python 3.8+
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `joblib`

Install dependencies using pip:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

Clone the repository and navigate to the directory:
```bash
git clone https://github.com/[your-username]/diabetes-prediction.git
cd diabetes-prediction
```

## Usage
1. Open `diabetes_prediction.ipynb` in Jupyter Notebook or JupyterLab.
2. Run all cells to preprocess the data, train models, evaluate performance, and generate visualizations.
3. Use the prediction pipeline section to test new patient data (update the `new_patient` DataFrame with relevant values).
4. View saved plots in the `plots/` directory or within the notebook.

Example to predict for a new patient (modify as needed):
- Adjust `new_patient` in the notebook with features like Glucose, BMI, etc.
- Run the prediction cells to get the outcome and cluster assignment.

## Contributing
Contributions are welcome! To contribute:
- Fork the repository.
- Create a new branch (`git checkout -b feature-branch`).
- Make changes and commit (`git commit -m "Description of changes"`).
- Push to the branch (`git push origin feature-branch`).
- Submit a pull request.

Please adhere to the team’s coding style and include comments for clarity.

## Team Contributions
- **Anbu (Data Preparation)**: Merged datasets, conducted exploratory data analysis, and preprocessed data.
- **Farhath (Modeling)**: Implemented classification and clustering algorithms, developed the prediction pipeline.
- **Bhavleen (Evaluation/Interpretation)**: Evaluated models, analyzed results, and tested the prediction pipeline.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset source: UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/diabetes).
- Inspired by machine learning techniques from Hastie et al. (2009), Chen & Guestrin (2016), and Lloyd (1982).

## Contact
For questions or feedback, open an issue in the repository.