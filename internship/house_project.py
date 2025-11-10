import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ============================================================================
# HOUSE PRICE PREDICTION PROJECT - SIMPLE & CLEAN
# ============================================================================

print("\n" + "="*70)
print("HOUSE PRICE PREDICTION SYSTEM")
print("="*70)

# STEP 1: LOAD DATASET
print("\n[STEP 1] Loading dataset...")
data = pd.read_csv('house_data_fresh.csv')
print(f"✓ Dataset loaded. Shape: {data.shape}")
print(f"  Columns: {list(data.columns)}")

# STEP 2: DATA CLEANING
print("\n[STEP 2] Cleaning dataset...")

# Remove null values
initial_shape = data.shape[0]
data = data.dropna()
print(f"✓ Removed null values. Removed {initial_shape - data.shape[0]} rows")

# Remove duplicates
initial_shape = data.shape[0]
data = data.drop_duplicates()
print(f"✓ Removed duplicates. Removed {initial_shape - data.shape[0]} rows")

# Remove outliers (using IQR method)
Q1 = data['Price'].quantile(0.25)
Q3 = data['Price'].quantile(0.75)
IQR = Q3 - Q1
initial_shape = data.shape[0]
data = data[(data['Price'] >= Q1 - 1.5 * IQR) & (data['Price'] <= Q3 + 1.5 * IQR)]
print(f"✓ Removed outliers. Removed {initial_shape - data.shape[0]} rows")

print(f"\n✓ Final cleaned dataset shape: {data.shape}")

# STEP 3: ENCODE CATEGORICAL FEATURES
print("\n[STEP 3] Encoding categorical features...")
data['Location_Code'] = data['Location'].map({'Urban': 1, 'Suburban': 2, 'Rural': 3})
data['Income_Code'] = data['Income'].map({'Low': 1, 'Medium': 2, 'High': 3})
print("✓ Location encoded: Urban=1, Suburban=2, Rural=3")
print("✓ Income encoded: Low=1, Medium=2, High=3")

# STEP 4: PREPARE FEATURES AND TARGET
print("\n[STEP 4] Preparing features for training...")
X = data[['Age', 'Rooms', 'Size', 'Location_Code', 'Income_Code']]
y = data['Price']
print(f"✓ Features (X) shape: {X.shape}")
print(f"✓ Target (y) shape: {y.shape}")

# STEP 5: TRAIN-TEST SPLIT
print("\n[STEP 5] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Training set size: {X_train.shape[0]}")
print(f"✓ Testing set size: {X_test.shape[0]}")

# STEP 6: TRAIN LINEAR REGRESSION MODEL
print("\n[STEP 6] Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("✓ Model training completed!")

# STEP 7: MODEL EVALUATION
print("\n[STEP 7] Evaluating model performance...")
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"✓ Training Accuracy (R² Score): {train_score:.4f}")
print(f"✓ Testing Accuracy (R² Score): {test_score:.4f}")

# ============================================================================
# STEP 8: USER INPUT AND PREDICTION
# ============================================================================

print("\n" + "="*70)
print("PREDICT HOUSE PRICE")
print("="*70)

try:
    # Get user inputs
    print("\nPlease enter the following details:")
    print("-" * 70)
    
    age = float(input("Enter House Age (years): "))
    rooms = int(input("Enter Number of Rooms: "))
    size = float(input("Enter House Size (square feet): "))
    
    print("\nLocation Options:")
    print("  1 = Urban")
    print("  2 = Suburban")
    print("  3 = Rural")
    location = int(input("Select Location (1/2/3): "))
    
    print("\nIncome Level Options:")
    print("  1 = Low")
    print("  2 = Medium")
    print("  3 = High")
    income = int(input("Select Income Level (1/2/3): "))
    
    # Validate inputs
    if location not in [1, 2, 3]:
        location = 2  # Default to Suburban
    if income not in [1, 2, 3]:
        income = 2  # Default to Medium
    
    # Create input array
    user_input = np.array([[age, rooms, size, location, income]])
    
    # Make prediction
    predicted_price = model.predict(user_input)[0]
    
    # Display result
    print("\n" + "="*70)
    print("PREDICTION RESULT")
    print("="*70)
    print(f"\nHouse Details:")
    print(f"  Age: {age} years")
    print(f"  Rooms: {rooms}")
    print(f"  Size: {size} sq ft")
    print(f"  Location: {['Urban', 'Suburban', 'Rural'][location-1]}")
    print(f"  Income Level: {['Low', 'Medium', 'High'][income-1]}")
    print(f"\n>>> PREDICTED HOUSE PRICE: ₹ {predicted_price:,.2f}")
    print("="*70 + "\n")

except ValueError:
    print("\n❌ Error: Please enter valid numbers!")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("Program ended.\n")
