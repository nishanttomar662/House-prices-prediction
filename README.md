House Price Prediction
A machine learning project that predicts house prices using Linear Regression. The project demonstrates the full ML workflow: data cleaning, preprocessing, model building, and prediction.

Features
Cleans and preprocesses real-world housing data

Handles missing values, duplicates, and outliers

Uses Linear Regression for accurate prediction

Interactive user input for price estimation

Dataset
158 house records

Features: Age, Rooms, Size, Location, Income Level, Price

Technologies
Python 3.x

Pandas

NumPy

Scikit-learn

How to Run
bash
pip install pandas numpy scikit-learn
python house_project.py
Follow the prompts to enter house details and get your predicted price.

Example Usage
text
Enter House Age (years): 30
Enter Number of Rooms: 4
Enter House Size (square feet): 2500
Select Location (1=Urban, 2=Suburban, 3=Rural): 1
Select Income Level (1=Low, 2=Medium, 3=High): 2

>>> PREDICTED HOUSE PRICE: ₹ 685,432.50
Project Structure
text
house-price-prediction/
├── house_project.py         # Main ML pipeline
├── house_data_fresh.csv     # Dataset
├── project_docs.md          # Documentation
└── README.md                # This file
Learning Points
End-to-end ML pipeline in Python

Real-world data cleaning and preprocessing

Linear Regression basics

Model evaluation (R² Score)

User interaction for predictions
