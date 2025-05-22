import numpy as np
import pandas as pd
import joblib

df =pd.read_csv(r"C:\\Users\ankit\\Desktop\\csv files\\diabetes.csv")
df.head()
df.isnull().sum()
df.shape
df.info()
df=df.dropna()

# Fill missing values with median
df.fillna(df.median(), inplace=True)





# Prepare features and target

x = df.drop(columns=['Outcome'])
y = df['Outcome']


from sklearn.model_selection import train_test_split
# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = rf.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(rf, 'diabetes_model.pkl')

# Print accuracy
print(f"rf accuracy: {rf.score(x_test, y_test):.2f}")