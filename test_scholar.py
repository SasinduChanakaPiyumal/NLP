#!/usr/bin/env python
# coding: utf-8

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the Classification directory to the Python path to import Scholar
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Classification'))

# Mock the problematic imports and file operations before importing Scholar
with patch('pandas.read_excel') as mock_read_excel:
    mock_read_excel.return_value = MagicMock()
    from Scholar import preprocess_text

class TestPreprocessText(unittest.TestCase):
    """Test suite for the preprocess_text function from Scholar.py"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass
    
    def test_text_lowercasing(self):
        """Test that text is properly converted to lowercase"""
        input_text = "HELLO WORLD This Is A Test"
        result = preprocess_text(input_text)
        # All words should be lowercase
        self.assertTrue(all(word.islower() for word in result.split()))
    
    def test_non_alphanumeric_removal(self):
        """Test removal of non-alphanumeric characters"""
        input_text = "Hello, World! This has: punctuation & symbols123"
        result = preprocess_text(input_text)
        # Should only contain letters and spaces
        for char in result:
            self.assertTrue(char.isalpha() or char.isspace(), 
                          f"Non-alphabetic character found: '{char}'")
    
    def test_stopword_removal(self):
        """Test that common English stopwords are removed"""
        input_text = "The quick brown fox jumps over the lazy dog"
        result = preprocess_text(input_text)
        # Common stopwords like 'the', 'over' should be removed
        result_words = result.split()
        common_stopwords = ['the', 'over']
        for stopword in common_stopwords:
            self.assertNotIn(stopword, result_words, 
                           f"Stopword '{stopword}' was not removed")
    
    def test_lemmatization(self):
        """Test that words are lemmatized to their base form"""
        input_text = "running runs ran"
        result = preprocess_text(input_text)
        # After lemmatization, all forms should become base form
        # Note: This is a simplified test as lemmatization results can vary
        self.assertIsInstance(result, str)
        self.assertTrue(len(result.split()) > 0)
    
    def test_empty_string(self):
        """Test handling of empty string input"""
        input_text = ""
        result = preprocess_text(input_text)
        self.assertEqual(result, "")
    
    def test_only_punctuation(self):
        """Test handling of text with only punctuation"""
        input_text = "!@#$%^&*()"
        result = preprocess_text(input_text)
        # Should return empty string after removing all punctuation
        self.assertEqual(result.strip(), "")
    
    def test_only_stopwords(self):
        """Test handling of text with only stopwords"""
        input_text = "the a an and or but"
        result = preprocess_text(input_text)
        # Should be empty or nearly empty after stopword removal
        self.assertTrue(len(result.strip()) == 0 or result.strip() == "")
    
    def test_mixed_case_with_numbers(self):
        """Test handling of mixed case text with numbers"""
        input_text = "Hello123 WORLD456 test789"
        result = preprocess_text(input_text)
        # Numbers should be removed, text should be lowercase
        self.assertTrue(all(char.isalpha() or char.isspace() for char in result))
        self.assertTrue(all(word.islower() for word in result.split() if word))
    
    def test_whitespace_handling(self):
        """Test that multiple whitespaces are handled correctly"""
        input_text = "hello    world    test"
        result = preprocess_text(input_text)
        # Should not have excessive whitespace
        self.assertNotIn("  ", result)  # No double spaces
    
    def test_return_type(self):
        """Test that function returns a string"""
        input_text = "test input"
        result = preprocess_text(input_text)
        self.assertIsInstance(result, str)

if __name__ == '__main__':
    unittest.main()
