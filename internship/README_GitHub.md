# House Price Prediction

A machine learning project that predicts house prices using Linear Regression. This project demonstrates a complete ML pipeline: data cleaning, preprocessing, model training, and predictions.

## 📋 Overview

This end-to-end ML project shows how to build a predictive model from raw data. It covers practical concepts like data cleaning, feature encoding, model training, and evaluation. Perfect for learning machine learning fundamentals.

## 🚀 Quick Start

### Installation

```bash
pip install pandas numpy scikit-learn
```

### Run the Project

```bash
python house_project.py
```

### Example Usage

```
Enter House Age (years): 25
Enter Number of Rooms: 4
Enter House Size (square feet): 2500
Select Location (1=Urban, 2=Suburban, 3=Rural): 1
Select Income Level (1=Low, 2=Medium, 3=High): 2

>>> PREDICTED HOUSE PRICE: ₹ 685,432.50
```

## 📊 Dataset

- **158 house records** with 6 features
- **Features:** Age, Rooms, Size, Location, Income Level, Price
- **Real-world issues:** Missing values, duplicates, outliers (all handled in code)

## 🛠️ Technologies

- **Python 3.x** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine learning

## 📁 Project Structure

```
house-price-prediction/
├── house_project.py          # Main executable code
├── house_data_fresh.csv      # Dataset
├── project_docs.md           # Detailed documentation
└── README.md                 # This file
```

## 🔄 How It Works

**8-Step ML Pipeline:**

1. **Load Dataset** - Read CSV file
2. **Clean Data** - Remove nulls, duplicates, outliers
3. **Encode Features** - Convert text to numbers
4. **Prepare Data** - Organize features and target
5. **Split Data** - 80% train, 20% test
6. **Train Model** - Linear Regression algorithm
7. **Evaluate** - Calculate R² Score
8. **Predict** - Get house price for user input

## 💡 Key Features

✅ Complete data preprocessing pipeline
✅ Handles real-world data issues
✅ Interactive user input system
✅ Model performance evaluation
✅ Well-commented code
✅ Production-ready

## 📈 Model Performance

- **Algorithm:** Linear Regression
- **R² Score:** Measures model accuracy (0-1 scale)
- **Train-Test Split:** 80-20 for unbiased evaluation

## 🎓 What You'll Learn

- Data loading and cleaning
- Handling missing values and outliers
- Feature encoding (categorical to numerical)
- Train-test split concept
- Linear Regression algorithm
- Model evaluation metrics
- Complete ML workflow

## 📚 Topics Covered

✓ Python basics and libraries (Pandas, NumPy, Scikit-learn)
✓ Data Preprocessing and Cleaning
✓ Machine Learning Fundamentals
✓ Linear Regression
✓ Model Training and Evaluation
✓ Real-world AI/ML applications

## 🎯 Use Cases

- Real estate property valuation
- House price estimation
- Investment analysis
- Market research
- Learning machine learning

## 📝 Code Example

```python
# Load and clean data
data = pd.read_csv('house_data_fresh.csv')
data = data.dropna()
data = data.drop_duplicates()

# Encode categorical features
data['Location_Code'] = data['Location'].map({'Urban': 1, 'Suburban': 2, 'Rural': 3})
data['Income_Code'] = data['Income'].map({'Low': 1, 'Medium': 2, 'High': 3})

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make prediction
prediction = model.predict([[age, rooms, size, location, income]])
```

## 🔍 For Viva/Interview

Be ready to explain:
- How data cleaning improves accuracy
- Why we use train-test split
- How Linear Regression works
- What R² Score means
- Real-world ML applications
- Complete pipeline steps

## 📖 For More Details

See `project_docs.md` for:
- Complete theory and concepts
- Detailed code explanations
- Viva Q&A section
- Future enhancements
- Advanced topics

## 🚀 Future Enhancements

- Use advanced algorithms (Random Forest, Gradient Boosting)
- Add more features and visualizations
- Create web interface (Flask/Django)
- Deploy on cloud (AWS, Heroku)
- Build REST API for predictions

## 📌 Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn

## 📄 License

MIT License - Free to use and modify

## 👨‍💻 Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python house_project.py`
4. Enter house details and get predictions!

---

**Happy Learning! Start your ML journey with this project.** 🎉
