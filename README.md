# Weather Models for Rain Prediction in Australia

## Project Overview
This project focuses on predicting rainfall in Australia using historical weather data. The goal is to build and evaluate machine learning models that can accurately predict whether it will rain tomorrow (`RainTomorrow`) based on meteorological features.

The project is implemented in a Jupyter Notebook: `Rain_Prediction_Australia.ipynb`.

## Dataset
The dataset contains detailed meteorological observations collected from multiple weather stations across Australia.

### Key Features
- Temperature (MinTemp, MaxTemp)
- Rainfall
- Evaporation
- Sunshine
- Wind direction and speed
- Humidity
- Pressure
- Cloud cover
- Target variable: **RainTomorrow** (Yes / No)

## Project Workflow
1. **Data Loading**
   - Importing and inspecting the Australian weather dataset

2. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables (e.g. wind direction, location)
   - Feature scaling

3. **Exploratory Data Analysis (EDA)**
   - Distribution analysis of weather features
   - Correlation analysis
   - RainTomorrow class balance inspection

4. **Model Building**
   The following classification models are implemented and compared:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier

5. **Model Evaluation**
   - Accuracy score
   - Confusion matrix
   - Classification report (Precision, Recall, F1-score)

## Technologies Used
- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://https://github.com/ahmedbenbellaa/rain-prediction-australia.git
   ```
2. Navigate to the project directory:
   ```bash
   cd rain-prediction-australia
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Rain_Prediction_Australia.ipynb
   ```

## Results
The trained models were evaluated on a test set to compare their predictive performance. Metrics such as accuracy, precision, recall, and F1-score are reported in the notebook, along with confusion matrices for clearer interpretation.

Random Forest achieved the strongest overall performance compared to simpler baseline models.

## Future Improvements
- Hyperparameter tuning
- Feature selection
- Trying advanced models (XGBoost, Random Forest, Neural Networks)
- Handling class imbalance more effectively



## Author
Ahmed Ben Bella

## License
This project is intended for educational and academic purposes.
