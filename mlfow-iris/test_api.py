import requests
import json

# Base URL dari API
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("\n=== Testing Model Info ===")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n=== Testing Single Prediction ===")
    
    # Test data - Iris Virginica
    data = {
        "sepal_length": 6.7,
        "sepal_width": 3.1,
        "petal_length": 5.6,
        "petal_width": 2.4
    }
    
    print(f"Input Data: {json.dumps(data, indent=2)}")
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n=== Testing Batch Prediction ===")
    
    # Test data - multiple instances
    data = {
        "instances": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 6.7,
                "sepal_width": 3.1,
                "petal_length": 5.6,
                "petal_width": 2.4
            },
            {
                "sepal_length": 5.7,
                "sepal_width": 2.8,
                "petal_length": 4.1,
                "petal_width": 1.3
            }
        ]
    }
    
    print(f"Input Data: {json.dumps(data, indent=2)}")
    response = requests.post(f"{BASE_URL}/predict/batch", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing FastAPI Iris Classifier API")
    print("=" * 60)
    
    try:
        # Test root endpoint
        print("\n=== Testing Root Endpoint ===")
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Run all tests
        tests = [
            ("Health Check", test_health),
            ("Model Info", test_model_info),
            ("Single Prediction", test_single_prediction),
            ("Batch Prediction", test_batch_prediction)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"\n❌ Error in {test_name}: {str(e)}")
                results[test_name] = False
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        all_passed = all(results.values())
        print("\n" + "=" * 60)
        if all_passed:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Tidak dapat terhubung ke API!")
        print("Pastikan server FastAPI sudah berjalan di http://localhost:8000")
        print("Jalankan: uv run python fastapi_serve.py")

if __name__ == "__main__":
    main()
