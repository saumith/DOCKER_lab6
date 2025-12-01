# Diabetes Prediction System

A production-ready machine learning web application for predicting diabetes risk using Random Forest classification. Built with Flask, scikit-learn, and Docker.

---

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Random Forest Classifier with 200 decision trees
- 72.73% Accuracy on test data with 0.823 AUC-ROC score
- REST API with 5 endpoints for predictions and model insights
- Beautiful Web Interface with responsive design
- Fully Dockerized for easy deployment
- Feature Importance Analysis to understand key health factors
- Health Monitoring endpoint for production reliability
- Real-time Predictions with confidence scores
- Input Validation and error handling
- Mobile Responsive design

---

## Demo

### Web Interface
Access the application at: **http://localhost:8000**

### Quick API Test
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree": 0.627,
    "age": 50
  }'
```

**Response:**
```json
{
  "prediction": "Diabetic",
  "probability": "71.97%",
  "risk_level": "High Risk",
  "confidence": "71.97%"
}
```

---

## Tech Stack

### Backend
- **Python 3.11** - Programming language
- **Flask 3.0** - Web framework
- **scikit-learn 1.3.2** - Machine learning library
- **pandas 2.1.4** - Data manipulation
- **NumPy 1.24.3** - Numerical computing
- **joblib 1.3.2** - Model serialization

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with gradients and animations
- **JavaScript (Vanilla)** - Interactivity and AJAX requests

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Orchestration

### Dataset
- **Pima Indians Diabetes Database** - 768 samples from UCI ML Repository

---

## Installation

### Prerequisites
- Python 3.11+
- pip
- Docker & Docker Compose (optional)

### Option 1: Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python src/train_model.py

# 5. Run the application
python src/app.py
```

Visit: **http://localhost:8000**

### Option 2: Docker Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction

# 2. Train model (must be done before Docker build)
pip install pandas numpy scikit-learn joblib
python src/train_model.py

# 3. Build and run with Docker
docker-compose up --build
```

Visit: **http://localhost:8000**

---

## Usage

### Web Interface

1. **Home Page**: Navigate to http://localhost:8000
2. **Start Prediction**: Click "Start Prediction" button
3. **Enter Data**: Fill in the 8 health metrics
4. **Get Results**: View prediction, probability, and risk level

### Python API

```python
import requests

# Predict diabetes risk
response = requests.post(
    'http://localhost:8000/predict',
    json={
        'pregnancies': 6,
        'glucose': 148,
        'blood_pressure': 72,
        'skin_thickness': 35,
        'insulin': 0,
        'bmi': 33.6,
        'diabetes_pedigree': 0.627,
        'age': 50
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']}")
print(f"Risk Level: {result['risk_level']}")
```

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Home Page
```http
GET /
```
Returns the landing page with project information.

---

#### 2. Prediction Form
```http
GET /predict
```
Returns the HTML prediction form.

---

#### 3. Make Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "pregnancies": 6,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree": 0.627,
  "age": 50
}
```

**Response:**
```json
{
  "prediction": "Diabetic",
  "probability": "71.97%",
  "risk_level": "High Risk",
  "confidence": "71.97%",
  "probabilities": {
    "Non-Diabetic": "28.03%",
    "Diabetic": "71.97%"
  }
}
```

---

#### 4. Feature Importance
```http
GET /feature-importance
```

**Response:**
```json
{
  "Glucose": 0.3159,
  "BMI": 0.1789,
  "Age": 0.1363,
  "DiabetesPedigreeFunction": 0.1014,
  "Pregnancies": 0.0690,
  "Insulin": 0.0684,
  "BloodPressure": 0.0662,
  "SkinThickness": 0.0638
}
```

---

#### 5. Model Information
```http
GET /model-info
```

**Response:**
```json
{
  "model_type": "Random Forest Classifier",
  "n_estimators": 200,
  "max_depth": 10,
  "n_features": 8,
  "n_classes": 2,
  "feature_names": ["Pregnancies", "Glucose", ...]
}
```

---

#### 6. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "Random Forest"
}
```

---

## Model Details

### Algorithm
- **Type**: Random Forest Classifier
- **Framework**: scikit-learn
- **Trees**: 200 decision trees
- **Max Depth**: 10 levels
- **Features**: 8 health metrics

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 72.73% |
| **AUC-ROC Score** | 0.8230 |
| **Precision (Non-Diabetic)** | 82% |
| **Precision (Diabetic)** | 60% |
| **Recall (Non-Diabetic)** | 75% |
| **Recall (Diabetic)** | 69% |
| **Cross-Validation** | 76.87% ± 5.52% |

### Feature Importance

| Feature | Importance | Description |
|---------|------------|-------------|
| **Glucose** | 31.59% | Plasma glucose concentration |
| **BMI** | 17.89% | Body Mass Index |
| **Age** | 13.63% | Age in years |
| **Diabetes Pedigree** | 10.14% | Family history function |
| **Pregnancies** | 6.90% | Number of pregnancies |
| **Insulin** | 6.84% | 2-Hour serum insulin |
| **Blood Pressure** | 6.62% | Diastolic blood pressure |
| **Skin Thickness** | 6.38% | Triceps skin fold thickness |

### Dataset

**Pima Indians Diabetes Database**
- **Source**: UCI Machine Learning Repository
- **Samples**: 768 female patients
- **Features**: 8 diagnostic measurements
- **Target**: Binary (Diabetic / Non-Diabetic)
- **Split**: 80% training, 20% testing
- **Class Distribution**: 65% Non-Diabetic, 35% Diabetic

### Why Random Forest?

Advantages for this project:
- Works excellently with small datasets (768 samples)
- No feature scaling required
- Provides feature importance (interpretability)
- Robust to overfitting with ensemble approach
- Handles non-linear relationships
- Fast training (seconds vs. minutes for neural networks)
- Better accuracy than deep learning for tabular data

---

## Project Structure

```
diabetes-prediction/
│
├── src/                           # Source code
│   ├── app.py                    # Flask API application
│   ├── train_model.py            # Model training script
│   └── test_api.py               # API testing suite
│
├── templates/                     # HTML templates
│   ├── index.html                # Landing page
│   └── predict.html              # Prediction form
│
├── static/                        # Static files (CSS, JS)
│   └── (empty - inline styles)
│
├── models/                        # Trained models (generated)
│   ├── diabetes_rf_model.pkl     # Random Forest model (~5MB)
│   └── feature_names.pkl         # Feature names list
│
├── .venv/                         # Virtual environment
│
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose orchestration
├── requirements.txt               # Python dependencies
├── .dockerignore                  # Docker build exclusions
├── .gitignore                     # Git exclusions
│
└── README.md                      # This file
```

---

## Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker-compose build

# Run the container
docker-compose up

# Run in background (detached mode)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down

# Restart the container
docker-compose restart
```

### Docker Commands

```bash
# View running containers
docker ps

# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Execute commands in container
docker exec -it diabetes-prediction-rf bash

# View container logs
docker logs diabetes-prediction-rf

# Clean rebuild
docker-compose down -v
docker system prune -f
docker-compose build --no-cache
docker-compose up
```

---

## Testing

### Run All Tests

```bash
# Make sure the app is running first
python src/app.py

# In another terminal, run tests
python src/test_api.py
```

### Manual Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree": 0.627,
    "age": 50
  }'

# Test feature importance
curl http://localhost:8000/feature-importance

# Test model info
curl http://localhost:8000/model-info
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/diabetes-prediction.git

# Create branch
git checkout -b feature/new-feature

# Make changes and test
python src/train_model.py
python src/app.py
python src/test_api.py

# Commit and push
git add .
git commit -m "Add new feature"
git push origin feature/new-feature
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Disclaimer

**IMPORTANT**: This application is for **educational and research purposes only**. 

- This tool should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment.
- Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.
- Never disregard professional medical advice or delay seeking it because of information from this application.
- The predictions made by this model are based on statistical patterns and should not be considered as medical diagnoses.

---

## Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com