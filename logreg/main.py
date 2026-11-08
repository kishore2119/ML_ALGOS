import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_gradient(theta,X,y):
    m = len(y)
    return X.T @ (sigmoid(X @ theta) - y) / m

def gradient_descent(X,y,alpha=0.1,epocs=100,tolerance=1e-7):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term

    theta = np.zeros(X_b.shape[1])  # initialize theta
    for i in range(epocs):
        gradient = calculate_gradient(theta,X_b,y)
        theta -= alpha * gradient

        if np.linalg.norm(gradient) < tolerance:
            break

    return theta

def predict(X,theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
    return sigmoid(X_b @ theta) >= 0.5



X,y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

good_theta = gradient_descent(X_train_scaled,y_train,alpha=0.1)


y_pred_train = predict(X_train_scaled,good_theta)
y_pred_test = predict(X_test_scaled,good_theta)

training_accuracy = accuracy_score(y_train, y_pred_train)
testing_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {training_accuracy}")
print(f"Testing Accuracy: {testing_accuracy}")


# Decision Boundary Visualization

X_vis = X_train_scaled[:, :2]

# Train a separate model using only 2 features
theta_vis = gradient_descent(X_vis, y_train, alpha=0.1, epocs=1000)

# Scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(X_vis[y_train == 0, 0], X_vis[y_train == 0, 1],
            color='red', label='Malignant')

plt.scatter(X_vis[y_train == 1, 0], X_vis[y_train == 1, 1],
            color='blue', label='Benign')

# Decision boundary
x_values = np.linspace(X_vis[:, 0].min(), X_vis[:, 0].max(), 100)

y_values = -(theta_vis[0] + theta_vis[1] * x_values) / theta_vis[2]

plt.plot(x_values, y_values, color='black', label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')

plt.legend()
plt.grid()
plt.show()