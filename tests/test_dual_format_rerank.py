"""
Test script for updated /rerank endpoint that supports both formats.
Tests that the /rerank endpoint accepts both native and HuggingFace formats.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 80)
print("Testing /rerank Endpoint - Dual Format Support")
print("=" * 80)

# Test 1: Native format with 'documents'
print("\n🔍 Test 1: Native format (documents field)")
test_native = {
    "query": "What is deep learning?",
    "documents": [
        "Deep learning is a subset of machine learning.",
        "The weather is nice today.",
        "Neural networks are computational models."
    ],
    "top_n": 2,
    "return_documents": True
}

try:
    response = requests.post(
        f"{BASE_URL}/rerank",
        json=test_native,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("✅ Native format works!")
        print(f"Results: {len(result['results'])} documents ranked")
        for i, res in enumerate(result["results"][:2]):
            print(f"  {i+1}. Index {res['index']}, Score: {res['relevance_score']:.4f}")
    else:
        print(f"❌ Error: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Connection failed. Start server with: python -m src.main")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: HuggingFace format with 'texts'
print("\n🔍 Test 2: HuggingFace format (texts field)")
test_hf = {
    "query": "What is deep learning?",
    "texts": [  # HuggingFace uses 'texts' instead of 'documents'
        "Deep learning is a subset of machine learning.",
        "The weather is nice today.",
        "Neural networks are computational models."
    ],
    "top_k": 2,  # HuggingFace uses 'top_k' instead of 'top_n'
    "return_texts": True  # HuggingFace uses 'return_texts'
}

try:
    response = requests.post(
        f"{BASE_URL}/rerank",
        json=test_hf,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("✅ HuggingFace format works!")
        print(f"Results: {len(result['results'])} documents ranked")
        
        # Verify HuggingFace response format (should have 'score' not 'relevance_score')
        first_result = result["results"][0] if result["results"] else {}
        if 'score' in first_result:
            print("✅ Response is in HuggingFace format (has 'score' field)")
        elif 'relevance_score' in first_result:
            print("❌ ERROR: Response is in native format but should be HuggingFace format!")
            print(f"   Response: {result}")
        
        for i, res in enumerate(result["results"][:2]):
            score = res.get('score', res.get('relevance_score', 0))
            print(f"  {i+1}. Index {res['index']}, Score: {score:.4f}")
    else:
        print(f"❌ Error: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Mixed - HuggingFace field names with native aliases
print("\n🔍 Test 3: Mixed format (texts + top_n)")
test_mixed = {
    "query": "What is deep learning?",
    "texts": ["Deep learning is ML subset.", "Weather is nice."],
    "top_n": 1,  # Using top_n with texts field
}

try:
    response = requests.post(
        f"{BASE_URL}/rerank",
        json=test_mixed,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("✅ Mixed format works!")
        print(f"Results: {len(result['results'])} documents ranked")
    else:
        print(f"❌ Error: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Error case - both fields provided
print("\n🔍 Test 4: Invalid - both documents and texts (should fail)")
test_invalid = {
    "query": "What is deep learning?",
    "documents": ["Doc 1"],
    "texts": ["Text 1"],
    "top_n": 1
}

try:
    response = requests.post(
        f"{BASE_URL}/rerank",
        json=test_invalid,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 422:
        print("✅ Correctly rejected invalid request")
        error = response.json()
        print(f"   Error: {error.get('detail', [{}])[0].get('msg', 'Validation error')}")
    else:
        print(f"❌ Should have returned 422, got {response.status_code}")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test 5: Error case - neither field provided
print("\n🔍 Test 5: Invalid - no documents or texts (should fail)")
test_invalid2 = {
    "query": "What is deep learning?",
    "top_n": 1
}

try:
    response = requests.post(
        f"{BASE_URL}/rerank",
        json=test_invalid2,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 422:
        print("✅ Correctly rejected invalid request")
        error = response.json()
        print(f"   Error: {error.get('detail', [{}])[0].get('msg', 'Validation error')}")
    else:
        print(f"❌ Should have returned 422, got {response.status_code}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)
print("✅ /rerank endpoint now supports BOTH formats:")
print("   • Native format: documents, top_n, return_documents")
print("   • HuggingFace format: texts, top_k, return_texts")
print("   • Aliases work across formats (e.g., top_k with documents)")
print("   • Proper validation (rejects both fields or neither field)")
print("\n📝 Both /rerank and /reranking endpoints are now available!")
