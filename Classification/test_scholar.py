#!/usr/bin/env python
"""
Unit tests for Scholar.py text preprocessing functionality.
"""

import unittest
import sys
import os

# Add the current directory to path to import Scholar functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the preprocessing function from Scholar.py
# Note: We need to import after ensuring NLTK resources are available
try:
    from Scholar import preprocess_text, ensure_nltk_resources
    ensure_nltk_resources()  # Ensure NLTK resources are downloaded
except ImportError as e:
    print(f"Error importing from Scholar.py: {e}")
    sys.exit(1)


class TestPreprocessText(unittest.TestCase):
    """Test cases for the preprocess_text function."""
    
    def test_basic_preprocessing(self):
        """Test basic text preprocessing with normal input."""
        text = "Hello world! This is a test."
        result = preprocess_text(text)
        self.assertIsInstance(result, str)
        self.assertNotIn("!", result)  # Punctuation should be removed
        self.assertNotIn("is", result)  # Stopwords should be removed
        self.assertIn("hello", result)  # Content words should remain
        self.assertIn("world", result)
        self.assertIn("test", result)
    
    def test_empty_string(self):
        """Test preprocessing of empty string."""
        result = preprocess_text("")
        self.assertEqual(result, "")
    
    def test_whitespace_only(self):
        """Test preprocessing of whitespace-only string."""
        result = preprocess_text("   \n\t  ")
        self.assertEqual(result, "")
    
    def test_punctuation_only(self):
        """Test preprocessing of punctuation-only text."""
        result = preprocess_text("!@#$%^&*()")
        self.assertEqual(result, "")
    
    def test_numbers_removed(self):
        """Test that numbers are removed during preprocessing."""
        text = "There are 123 items and 456 more."
        result = preprocess_text(text)
        self.assertNotIn("123", result)
        self.assertNotIn("456", result)
        self.assertIn("item", result)  # Should be lemmatized to singular
    
    def test_case_insensitive(self):
        """Test that text is converted to lowercase."""
        text = "UPPERCASE and lowercase"
        result = preprocess_text(text)
        self.assertNotIn("UPPERCASE", result)
        self.assertIn("uppercase", result)
        self.assertIn("lowercase", result)
    
    def test_stopwords_removal(self):
        """Test that common English stopwords are removed."""
        text = "the quick brown fox jumps over the lazy dog"
        result = preprocess_text(text)
        stopwords = ["the", "over"]
        for stopword in stopwords:
            self.assertNotIn(stopword, result)
        # Content words should remain
        content_words = ["quick", "brown", "fox", "jump", "lazy", "dog"]
        for word in content_words:
            self.assertIn(word, result)
    
    def test_lemmatization(self):
        """Test that words are properly lemmatized."""
        text = "running runs ran cats dogs houses"
        result = preprocess_text(text)
        # These should be lemmatized to base forms
        self.assertIn("run", result)  # running -> run, runs -> run
        self.assertIn("cat", result)  # cats -> cat
        self.assertIn("dog", result)  # dogs -> dog
        self.assertIn("house", result)  # houses -> house
    
    def test_none_input(self):
        """Test that None input raises ValueError."""
        with self.assertRaises(ValueError):
            preprocess_text(None)
    
    def test_non_string_input(self):
        """Test that non-string input raises ValueError."""
        with self.assertRaises(ValueError):
            preprocess_text(123)
        with self.assertRaises(ValueError):
            preprocess_text(["list", "input"])
    
    def test_special_characters(self):
        """Test handling of special characters and symbols."""
        text = "Hello@world.com #hashtag $money 50% off"
        result = preprocess_text(text)
        # Special characters should be removed
        self.assertNotIn("@", result)
        self.assertNotIn(".", result)
        self.assertNotIn("#", result)
        self.assertNotIn("$", result)
        self.assertNotIn("%", result)
        # Words should remain (some may be filtered as non-alphabetic)
        self.assertIn("hello", result)
        self.assertIn("world", result)
    
    def test_multiple_spaces(self):
        """Test handling of multiple consecutive spaces."""
        text = "word1    word2     word3"
        result = preprocess_text(text)
        # Result should not have multiple consecutive spaces
        self.assertNotIn("  ", result)
        words = result.split()
        self.assertIn("word", words[0])  # word1 -> word (lemmatized)
        self.assertEqual(len(words), 3)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
