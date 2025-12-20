"""
Test to verify the fix for 'str' object has no attribute get error.
Tests document text extraction with various response formats.
"""

import sys
sys.path.insert(0, '.')

from src.api.routes import extract_document_text

def test_extract_document_text():
    """Test the helper function with various input types."""
    
    print("Testing extract_document_text helper function:")
    print("=" * 60)
    
    # Test case 1: Dict with 'text' key (normal case)
    doc1 = {"text": "This is a document"}
    result1 = extract_document_text(doc1)
    assert result1 == "This is a document", f"Expected 'This is a document', got {result1!r}"
    print("✅ Dict with 'text' key: PASS")
    
    # Test case 2: Direct string (edge case that caused the error)
    doc2 = "Direct string document"
    result2 = extract_document_text(doc2)
    assert result2 == "Direct string document", f"Expected 'Direct string document', got {result2!r}"
    print("✅ Direct string: PASS")
    
    # Test case 3: Dict with 'content' key (alternative)
    doc3 = {"content": "Content field"}
    result3 = extract_document_text(doc3)
    assert result3 == "Content field", f"Expected 'Content field', got {result3!r}"
    print("✅ Dict with 'content' key: PASS")
    
    # Test case 4: None (when return_documents=False)
    doc4 = None
    result4 = extract_document_text(doc4)
    assert result4 is None, f"Expected None, got {result4!r}"
    print("✅ None value: PASS")
    
    # Test case 5: Empty dict
    doc5 = {}
    result5 = extract_document_text(doc5)
    assert result5 == "", f"Expected empty string, got {result5!r}"
    print("✅ Empty dict: PASS")
    
    # Test case 6: Dict with other keys (should return empty string)
    doc6 = {"other_key": "value"}
    result6 = extract_document_text(doc6)
    assert result6 == "", f"Expected empty string, got {result6!r}"
    print("✅ Dict with other keys: PASS")
    
    # Test case 7: Number (edge case)
    doc7 = 123
    result7 = extract_document_text(doc7)
    assert result7 == "123", f"Expected '123', got {result7!r}"
    print("✅ Number converted to string: PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! The fix resolves the error:")
    print("  'str' object has no attribute get")
    print("\nThe helper function now safely handles:")
    print("  ✅ Dict with 'text' key")
    print("  ✅ Dict with 'content' key")
    print("  ✅ Direct string values")
    print("  ✅ None values")
    print("  ✅ Other edge cases")

if __name__ == "__main__":
    try:
        test_extract_document_text()
        print("\n🎉 SUCCESS: Fix verified!")
    except AssertionError as e:
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
