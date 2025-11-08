#!/usr/bin/env python3
"""
Test the Fraud Detection API
"""
import requests
import json
from datetime import datetime

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_model_info():
    """Test model info endpoint"""
    print("="*60)
    print("Testing Model Info Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_URL}/model_info")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Model Type: {data['model_type']}")
    print(f"Trained At: {data['trained_at']}")
    print(f"Features: {len(data['features'])} features")
    print()

def test_single_click():
    """Test single click scoring"""
    print("="*60)
    print("Testing Single Click Scoring")
    print("="*60)
    
    # Suspicious click (bot-like behavior)
    suspicious_click = {
        "ip": 999999,
        "app": 3,
        "device": 1,
        "os": 13,
        "channel": 497,
        "click_time": "2017-11-07 02:30:38"  # Night time = suspicious
    }
    
    response = requests.post(f"{API_URL}/score_click", json=suspicious_click)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    print(f"\n🔍 Click Analysis:")
    print(f"   Is Fraud: {result['is_fraud']}")
    print(f"   Fraud Probability: {result['fraud_probability']*100:.2f}%")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Confidence: {result['confidence']}")
    print()
    
    # Normal click
    normal_click = {
        "ip": 12345,
        "app": 5,
        "device": 2,
        "os": 19,
        "channel": 120,
        "click_time": "2017-11-07 14:30:38"  # Day time = less suspicious
    }
    
    response = requests.post(f"{API_URL}/score_click", json=normal_click)
    result = response.json()
    
    print(f"🔍 Normal Click Analysis:")
    print(f"   Is Fraud: {result['is_fraud']}")
    print(f"   Fraud Probability: {result['fraud_probability']*100:.2f}%")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Confidence: {result['confidence']}")
    print()

def test_batch_scoring():
    """Test batch scoring"""
    print("="*60)
    print("Testing Batch Scoring")
    print("="*60)
    
    batch_data = {
        "clicks": [
            {"ip": 111111, "app": 3, "device": 1, "os": 13, "channel": 497, "click_time": "2017-11-07 02:30:38"},
            {"ip": 222222, "app": 5, "device": 2, "os": 19, "channel": 120, "click_time": "2017-11-07 14:30:38"},
            {"ip": 333333, "app": 12, "device": 3, "os": 17, "channel": 134, "click_time": "2017-11-07 10:15:22"},
            {"ip": 444444, "app": 9, "device": 1, "os": 22, "channel": 259, "click_time": "2017-11-07 03:45:11"},
            {"ip": 555555, "app": 15, "device": 4, "os": 13, "channel": 477, "click_time": "2017-11-07 16:20:55"},
        ]
    }
    
    response = requests.post(f"{API_URL}/score_batch", json=batch_data)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    print(f"\n📊 Batch Results:")
    print(f"   Total Clicks: {result['total_clicks']}")
    print(f"   Fraud Detected: {result['fraud_detected']}")
    print(f"   Fraud Rate: {result['fraud_rate']*100:.2f}%")
    
    print(f"\n   Individual Results:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"   Click {i}: {'🚨 FRAUD' if pred['is_fraud'] else '✅ LEGIT'} "
              f"(prob: {pred['fraud_probability']*100:.1f}%, risk: {pred['risk_level']})")
    print()

def test_performance():
    """Test API response time"""
    print("="*60)
    print("Testing API Performance")
    print("="*60)
    
    import time
    
    click = {
        "ip": 123456,
        "app": 3,
        "device": 1,
        "os": 13,
        "channel": 497
    }
    
    # Warm-up request
    requests.post(f"{API_URL}/score_click", json=click)
    
    # Time multiple requests
    num_requests = 100
    start = time.time()
    
    for _ in range(num_requests):
        requests.post(f"{API_URL}/score_click", json=click)
    
    end = time.time()
    total_time = end - start
    avg_time = (total_time / num_requests) * 1000  # ms
    
    print(f"\n⚡ Performance Results:")
    print(f"   Total Requests: {num_requests}")
    print(f"   Total Time: {total_time:.2f} seconds")
    print(f"   Average Latency: {avg_time:.2f} ms")
    print(f"   Requests/Second: {num_requests/total_time:.2f}")
    print()

def main():
    """Run all tests"""
    print("\n" + "🚀 FRAUD DETECTION API TESTING SUITE" + "\n")
    
    try:
        test_health()
        test_model_info()
        test_single_click()
        test_batch_scoring()
        test_performance()
        
        print("="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("   Make sure the API is running:")
        print("   python src/api/fraud_api.py")
        print("   or")
        print("   uvicorn src.api.fraud_api:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    main()
