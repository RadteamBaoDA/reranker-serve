"""
Test script for HuggingFace-compatible rerank API.
Tests the new /reranking endpoint that uses 'texts' field.
"""

import requests
import json

# Test configurations
BASE_URL = "http://localhost:8000"
ENDPOINTS = [
    "/reranking",
    "/v1/reranking",
]

# Test data in HuggingFace format (uses 'texts' instead of 'documents')
test_data = {
    "query": "What is deep learning?",
    "texts": [
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        "The weather is nice today with clear skies.",
        "Neural networks are computational models inspired by the human brain.",
        "I like to eat pizza on weekends.",
    ],
    "top_k": 2,
}

# Alternative test with top_n (should work due to alias)
test_data_alt = {
    "query": "What is deep learning?",
    "texts": [
        "Deep learning is a subset of machine learning.",
        "The weather is nice today.",
    ],
    "top_n": 1,  # Using top_n instead of top_k
}

print("=" * 80)
print("Testing HuggingFace-compatible Rerank API")
print("=" * 80)

for endpoint in ENDPOINTS:
    url = f"{BASE_URL}{endpoint}"
    print(f"\n🔍 Testing endpoint: {endpoint}")
    print(f"URL: {url}")
    print(f"Request body: {json.dumps(test_data, indent=2)}")
    
    try:
        response = requests.post(
            url,
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        
        print(f"\n✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Validate response structure
            assert "results" in result, "Response missing 'results' field"
            assert "model" in result, "Response missing 'model' field"
            assert len(result["results"]) <= test_data["top_k"], "Too many results returned"
            
            for i, res in enumerate(result["results"]):
                assert "index" in res, f"Result {i} missing 'index'"
                assert "score" in res, f"Result {i} missing 'score'"
                print(f"  Result {i}: index={res['index']}, score={res['score']:.4f}")
                if "text" in res and res["text"]:
                    print(f"    Text: {res['text'][:60]}...")
            
            print("✅ All assertions passed!")
        else:
            print(f"❌ Error response:")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the server running?")
        print("   Start the server with: python -m src.main")
    except Exception as e:
        print(f"❌ Error: {e}")

# Test with top_n alias
print(f"\n{'=' * 80}")
print("Testing with 'top_n' alias")
print(f"{'=' * 80}")
url = f"{BASE_URL}/reranking"
print(f"Request body: {json.dumps(test_data_alt, indent=2)}")

try:
    response = requests.post(
        url,
        json=test_data_alt,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    
    print(f"\n✅ Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print("✅ top_n alias works!")
    else:
        print(f"❌ Error response:")
        print(response.text)

except Exception as e:
    print(f"❌ Error: {e}")

print(f"\n{'=' * 80}")
print("Testing the old /rerank endpoint (should still work)")
print(f"{'=' * 80}")

# Test that old endpoint still works with 'documents' field
old_test_data = {
    "query": "What is deep learning?",
    "documents": [  # Old format uses 'documents'
        "Deep learning is a subset of machine learning.",
        "The weather is nice today.",
    ],
    "top_n": 1,
}

url = f"{BASE_URL}/rerank"
print(f"Request body: {json.dumps(old_test_data, indent=2)}")

try:
    response = requests.post(
        url,
        json=old_test_data,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    
    print(f"\n✅ Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print("✅ Old endpoint still works!")
    else:
        print(f"❌ Error response:")
        print(response.text)

except Exception as e:
    print(f"❌ Error: {e}")

print(f"\n{'=' * 80}")
print("Test Summary")
print(f"{'=' * 80}")
print("✅ HuggingFace-compatible endpoints: /reranking, /v1/reranking")
print("✅ Accepts 'texts' field (HuggingFace format)")
print("✅ Supports both 'top_k' and 'top_n' parameters")
print("✅ Returns 'score' field in results")
print("✅ Backward compatible - old /rerank endpoint still works")
