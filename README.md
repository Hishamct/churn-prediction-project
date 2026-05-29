# Customer Churn Prediction MLOps Project

## Project Overview

This project implements an end-to-end MLOps pipeline for customer churn prediction using a real-world telecom customer dataset.

The objective is to predict whether a customer is likely to leave the telecom service based on customer demographics, account information, and service usage.

---

## Technologies Used

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-Learn


### API Deployment

* FastAPI
* Uvicorn

### CI/CD

* GitHub Actions

### Testing

* Pytest
* Pylint


### Version Control

* Git
* GitHub

---

## Project Structure

```text
churn-prediction-project/
│
├── app/
├── data/
├── src/
├── tests/
├── .github/workflows/
├── requirements.txt
└── README.md
```

---

## Version Control Strategy

### Feature Branching

Feature branching was used to isolate development activities.

Examples:

* feature-documentation


Benefits:

* Safe development
* Easy collaboration
* Reduced risk of breaking production code

---

### GitHub Flow

Workflow followed:

1. Create feature branch
2. Develop feature
3. Commit changes
4. Push branch
5. Create Pull Request
6. Review and merge into main

---

### Advanced Git Operations

The following Git operations were practiced:

* Branch creation
* Branch merging
* Merge conflict resolution
* Rebase
* Cherry-pick
* Pull requests

---



## Testing Strategy

### Unit Testing

Implemented using pytest.

Purpose:

* Verify model training logic
* Validate functionality of individual components

---

### Integration Testing

Implemented using FastAPI TestClient.

Purpose:

* Validate API endpoints
* Ensure application components work together

---

### Code Quality Validation

Implemented using pylint.

Purpose:

* Maintain coding standards
* Detect potential issues early

---

## CI/CD Pipeline

GitHub Actions was used to automate the workflow.

Pipeline Steps:

1. Checkout repository
2. Install dependencies
3. Run pylint
4. Run pytest
5. Train model

Benefits:

* Automation
* Faster feedback
* Improved reliability

---

## Model Deployment

The trained model is exposed using FastAPI.

Endpoints:

### GET /

Health check endpoint.

### POST /predict

Returns customer churn predictions.

---

## Monitoring

Implemented:

* Request logging
* Prediction logging
* Basic drift detection logic

Purpose:

* Observe model behavior
* Detect abnormal input patterns

---


## Best Practices Applied

* Feature branching
* Pull request workflow
* Automated testing
* Continuous Integration
* Experiment tracking
* Reproducible pipelines
* Code quality checks
* API deployment


---

## Conclusion

This project demonstrates an end-to-end production-style MLOps workflow combining machine learning, software engineering, testing, version control, CI/CD, deployment, and monitoring practices.
