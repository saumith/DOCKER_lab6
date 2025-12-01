
# ============================================================================
# FILE: src/test_api.py
# ============================================================================
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Health check passed!")
            return True
        else:
            print("❌ Health check failed!")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction_endpoint():
    print("\n" + "="*60)
    print("Testing Prediction Endpoint")
    print("="*60)
    
    test_cases = [
        {
            "name": "High Risk Patient",
            "data": {
                "pregnancies": 6, "glucose": 148, "blood_pressure": 72,
                "skin_thickness": 35, "insulin": 0, "bmi": 33.6,
                "diabetes_pedigree": 0.627, "age": 50
            }
        },
        {
            "name": "Low Risk Patient",
            "data": {
                "pregnancies": 1, "glucose": 85, "blood_pressure": 66,
                "skin_thickness": 29, "insulin": 0, "bmi": 26.6,
                "diabetes_pedigree": 0.351, "age": 31
            }
        }
    ]
    
    results = []
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        try:
            response = requests.post(f"{BASE_URL}/predict", json=test_case['data'])
            if response.status_code == 200:
                print(f"Result: {json.dumps(response.json(), indent=2)}")
                print("✅ Prediction successful!")
                results.append(True)
            else:
                print(f"❌ Failed: {response.text}")
                results.append(False)
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(False)
    
    return all(results)

def run_all_tests():
    print("\n" + "="*60)
    print("🏥 DIABETES PREDICTION API TEST SUITE")
    print("="*60)
    
    try:
        requests.get(BASE_URL, timeout=2)
    except:
        print(f"\n❌ Server not running at {BASE_URL}")
        print("Start with: python src/app.py")
        return
    
    test_results = [
        ("Health Check", test_health_endpoint()),
        ("Predictions", test_prediction_endpoint())
    ]
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    run_all_tests()