# ML Testing: Hugging Face Text Classification Model

## 📋 Overview

This project demonstrates **comprehensive ML model testing** for a Hugging Face text classification model (facebook/bart-large-mnli). Rather than just building an ML model, this focuses on **testing** the model across four critical dimensions:

1. **Functional Tests** - Does the model work as expected?
2. **Data Quality Tests** - Can it handle various data formats?
3. **Robustness Tests** - How does it react to weird/malicious inputs?
4. **Performance Tests** - How fast is it? Can it handle load?

---

## 🎯 What is ML Testing?

ML Testing is different from software testing. Instead of checking if code runs, we check if the **model predictions are correct and reliable**.

### Key Concepts (Level 0 - Beginner):

```
Traditional Testing:     |  ML Testing:
- Input/Output correct? | - Predictions correct?
- Error handling?       | - Handles edge cases?
- Performance okay?     | - Works on new data?
- Security?             | - Adversarial inputs?
```

---

## 📁 Project Structure

```
ml-testing-huggingface-classification/
├── tests/
│   ├── test_functional.py          # Test model outputs (predictions, scores)
│   ├── test_data_quality.py        # Test with various data types
│   ├── test_robustness.py          # Test with adversarial/weird inputs
│   └── test_performance.py         # Load testing with Locust
├── src/
│   ├── model.py                    # Model wrapper
│   └── data_validator.py           # Data validation utilities
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## 🔧 Installation & Setup

### Prerequisites:
- Python 3.8+
- pip

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/ml-testing-huggingface-classification.git
cd ml-testing-huggingface-classification
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/test_functional.py -v
pytest tests/test_data_quality.py -v
pytest tests/test_robustness.py -v
pytest tests/test_performance.py -v
```

---

## 1️⃣ FUNCTIONAL TESTS

### What does this test?
Functional tests check if the model **produces valid outputs in expected format**.

### Test Cases:
```python
# ✅ Valid input returns valid output
# ✅ Output has correct field names
# ✅ Confidence scores are between 0-1
# ✅ Labels are from expected categories
# ✅ Batch processing works
```

### Example Code Structure:
```python
def test_model_returns_valid_output():
    """
    Test that model returns properly formatted output.
    
    Returns:
        - List of predictions
        - Each prediction has 'label' and 'score'
        - Score is float between 0-1
    """
    # Arrange
    model = load_model()
    test_input = "This is a great product!"
    
    # Act
    result = model.predict(test_input)
    
    # Assert
    assert isinstance(result, list)
    assert 'label' in result[0]
    assert 'score' in result[0]
    assert 0 <= result[0]['score'] <= 1
```

---

## 2️⃣ DATA QUALITY TESTS

### What does this test?
Data quality tests check if the model can handle **different types of inputs** without crashing or behaving unexpectedly.

### Test Scenarios:
```
✅ Short text (1-5 words)
✅ Long text (500+ words)
✅ Multiple languages (English, Spanish, Hindi)
✅ Special characters and emojis
✅ HTML/XML content
✅ Numbers and punctuation
✅ Empty/null inputs
✅ URLs in text
✅ Mixed case text
```

### Example:
```python
def test_model_handles_long_text():
    """
    Test that model works with very long texts.
    
    This ensures the model doesn't fail on real-world data
    which often contains long documents.
    """
    model = load_model()
    long_text = "word " * 1000  # Very long text
    
    result = model.predict(long_text)
    assert result is not None
    assert len(result) > 0
```

---

## 3️⃣ ROBUSTNESS TESTS

### What does this test?
Robustness tests check if the model **resists adversarial/weird inputs** that could cause incorrect predictions.

### Attack Types:
```
🔴 Adversarial Attacks:
   - Add random words
   - Repeat characters
   - Inject contradictory phrases

🔴 Prompt Injection:
   - Add hidden instructions
   - "Ignore previous instructions..."

🔴 Gibberish & Noise:
   - Random characters
   - Base64 encoded text
   - URL-encoded strings

🔴 Extreme Cases:
   - Only numbers
   - Only punctuation
   - Unicode characters
   - Extremely long sequences
```

### Example:
```python
def test_model_handles_adversarial_input():
    """
    Test that model output changes minimally with small input changes.
    
    This measures robustness - a robust model should not drastically
    change predictions with small perturbations.
    """
    model = load_model()
    text1 = "This product is good"
    text2 = "This product is good and excellent and amazing"
    
    result1 = model.predict(text1)
    result2 = model.predict(text2)
    
    # Predictions should be similar (same label)
    assert result1[0]['label'] == result2[0]['label']
```

---

## 4️⃣ PERFORMANCE TESTS

### What does this test?
Performance tests measure **speed, throughput, and error rates** under load.

### Metrics Measured:
```
⏱️  Latency: How long does one prediction take?
📊 Throughput: How many predictions per second?
💥 Error Rate: What % of requests fail?
🔄 P95/P99: Time for slowest 5%/1% of requests?
```

### Tools Used:
- **Locust**: Simulates concurrent users making requests
- **JMeter**: Alternative load testing tool
- **Pytest**: For performance benchmarking

### Example Locust Test:
```python
from locust import HttpUser, task, between

class ModelUser(HttpUser):
    """
    Simulates a user making requests to the model endpoint.
    Locust will spawn multiple concurrent users.
    """
    wait_time = between(1, 5)  # Wait 1-5 seconds between requests
    
    @task
    def predict_sentiment(self):
        """
        Make a prediction request.
        Locust records response time and success/failure.
        """
        payload = {"text": "This is a test review"}
        self.client.post("/predict", json=payload)
```

### Running Load Tests:
```bash
# Start Locust UI
locust -f tests/test_performance.py

# Then open browser to http://localhost:8089
# Set number of users and spawn rate
```

---

## 🛠️ Tooling Explained

### 1. Pytest
```
Why: Industry standard for Python testing
What: Runs all tests, collects results, shows failures
Command: pytest tests/ -v
```

### 2. Great Expectations (Data Validation)
```
Why: Validates data schema before passing to model
What: Checks data types, ranges, null values
Example: Ensure all reviews are strings, ratings 1-5
```

### 3. Locust (Load Testing)
```
Why: Simulates many concurrent users
What: Measures latency, throughput, failure rates
Metric: "Can model handle 100 concurrent requests?"
```

### 4. Pandas & NumPy
```
Why: Data manipulation and analysis
What: Prepare test data, calculate statistics
```

---

## 📊 Expected Output Example

```
========================== test session starts ==========================
collected 15 items

tests/test_functional.py::test_model_returns_valid_output PASSED    [6%]
tests/test_functional.py::test_batch_processing_works PASSED        [13%]
tests/test_data_quality.py::test_handles_long_text PASSED           [20%]
tests/test_data_quality.py::test_handles_special_chars PASSED       [26%]
tests/test_robustness.py::test_adversarial_input_handling PASSED    [33%]
tests/test_robustness.py::test_gibberish_input PASSED               [40%]
tests/test_performance.py::test_latency_under_load PASSED           [46%]

======================== 7 passed in 2.34s =========================
```

---

## 🚀 Next Steps / Improvements

1. **Add Data Drift Detection** - Monitor if input data distribution changes
2. **Model Versioning** - Track which model version ran which tests
3. **Automated Alerts** - Notify if test failures
4. **CI/CD Integration** - Run tests automatically on every code push
5. **Visualization Dashboard** - Show test results over time

---

## 📚 Resources & References

- **Pytest Docs**: https://docs.pytest.org/
- **Hugging Face Models**: https://huggingface.co/models
- **ML Testing Paper**: https://arxiv.org/abs/1908.04626
- **Great Expectations**: https://greatexpectations.io/
- **Locust**: https://locust.io/

---

## 💡 Key Takeaways

✅ ML Testing = Testing model predictions (not just code)
✅ Four dimensions: Functional, Data Quality, Robustness, Performance
✅ Use Pytest for organizing tests
✅ Use Great Expectations for data validation
✅ Use Locust for load testing
✅ Always test on realistic, diverse data

---

## 🤝 Contributing

Feel free to add more test cases! This is meant to be a learning resource.

---

**Created as a Level-0 ML Testing Portfolio Project**
