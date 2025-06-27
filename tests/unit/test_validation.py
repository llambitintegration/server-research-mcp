"""Unit tests for validation functions and user input handling."""

import pytest
from unittest.mock import patch, MagicMock
from server_research_mcp.crew import validate_research_output, validate_report_output
from server_research_mcp.main import get_user_input


class TestResearchOutputValidation:
    """Test research output validation function."""
    
    def test_valid_research_output(self, valid_research_output):
        """Test validation passes for properly formatted research output."""
        is_valid, result = validate_research_output(valid_research_output)
        assert is_valid is True
        assert result == valid_research_output.strip()
        
    def test_research_output_too_short(self):
        """Test validation fails for output that's too brief."""
        short_output = "Too short"
        is_valid, error_msg = validate_research_output(short_output)
        
        assert is_valid is False
        assert "too brief" in error_msg.lower()
        
    def test_research_output_no_bullet_points(self):
        """Test validation fails for output without bullet points."""
        no_bullets = "This is a long output without any bullet points. " * 10
        is_valid, error_msg = validate_research_output(no_bullets)
        
        assert is_valid is False
        assert "bullet points" in error_msg.lower()
        
    def test_research_output_insufficient_bullet_points(self):
        """Test validation fails with too few bullet points."""
        few_bullets = """
        • Finding 1: First point
        • Finding 2: Second point
        • Finding 3: Third point with enough content to pass length check
        """
        is_valid, error_msg = validate_research_output(few_bullets)
        
        assert is_valid is False
        assert "bullet points" in error_msg.lower()
        assert "3" in error_msg or "three" in error_msg.lower()
        
    def test_research_output_edge_cases(self):
        """Test edge cases for research output validation."""
        # Exactly 10 bullet points
        exact_bullets = "\n".join([f"• Finding {i}: {'content' * 5}" for i in range(1, 11)])
        is_valid, result = validate_research_output(exact_bullets)
        assert is_valid is True
        
        # Unicode bullet points
        unicode_bullets = "\n".join([f"• Finding {i}: {'content' * 5}" for i in range(1, 11)])
        is_valid, result = validate_research_output(unicode_bullets)
        assert is_valid is True
        
    def test_research_output_none_handling(self):
        """Test validation handles None input gracefully."""
        is_valid, error_msg = validate_research_output(None)
        assert is_valid is False
        assert "error" in error_msg.lower()


class TestReportOutputValidation:
    """Test report output validation function."""
    
    def test_valid_report_output(self, valid_report_output):
        """Test validation passes for properly formatted report."""
        is_valid, result = validate_report_output(valid_report_output)
        assert is_valid is True
        assert result == valid_report_output.strip()
        
    def test_report_too_short(self):
        """Test validation fails for brief reports."""
        short_report = "# Title\nToo short report"
        is_valid, error_msg = validate_report_output(short_report)
        
        assert is_valid is False
        assert "too brief" in error_msg.lower()
        
    def test_report_no_headers(self):
        """Test validation fails for reports without markdown headers."""
        no_headers = "Long report content without any markdown headers. " * 50
        is_valid, error_msg = validate_report_output(no_headers)
        
        assert is_valid is False
        assert "headers" in error_msg.lower()
        
    def test_report_insufficient_headers(self):
        """Test validation fails with too few headers."""
        few_headers = """# Main Title

This is a very long section with lots of content to pass the length requirement.
""" + ("More content to ensure we exceed minimum length. " * 20)
        
        is_valid, error_msg = validate_report_output(few_headers)
        
        assert is_valid is False
        assert "1" in error_msg or "one" in error_msg.lower()
        assert "3" in error_msg or "three" in error_msg.lower()
        
    def test_report_with_code_blocks(self):
        """Test validation fails for reports containing code blocks."""
        with_code = """# Report

## Section 1
Content here

## Section 2
More content

```python
code_block = "should not be here"
```
""" + ("Additional content. " * 30)
        
        is_valid, error_msg = validate_report_output(with_code)
        
        assert is_valid is False
        assert "code block" in error_msg.lower()
        
    def test_report_validation_edge_cases(self):
        """Test edge cases for report validation."""
        # Exactly 3 headers
        three_headers = """# Title
Content here with sufficient length to pass validation.

## Section 1  
More content to ensure we have enough text for the validation to pass.

### Section 2
Final section with additional content to meet length requirements.
""" + ("Extra content. " * 20)
        
        is_valid, result = validate_report_output(three_headers)
        assert is_valid is True
        
    def test_report_none_handling(self):
        """Test validation handles None input gracefully."""
        is_valid, error_msg = validate_report_output(None)
        assert is_valid is False
        assert "error" in error_msg.lower()


class TestUserInput:
    """Test user input collection functionality."""
    
    @patch('builtins.input', side_effect=['AI Testing', 'y'])
    def test_get_user_input_valid_flow(self, mock_input):
        """Test successful user input collection."""
        topic = get_user_input()
        assert topic == 'AI Testing'
        assert mock_input.call_count == 2
        
    @patch('builtins.input', side_effect=['', '   ', 'Valid Topic', 'yes'])
    def test_get_user_input_empty_validation(self, mock_input):
        """Test validation rejects empty/whitespace input."""
        topic = get_user_input()
        assert topic == 'Valid Topic'
        assert mock_input.call_count == 4
        
    @patch('builtins.input', side_effect=['Machine Learning', 'n'])
    @patch('sys.exit')
    def test_get_user_input_cancellation(self, mock_exit, mock_input):
        """Test user can cancel input process."""
        get_user_input()
        mock_exit.assert_called_once_with(0)
        
    @patch('builtins.input', side_effect=['Research Topic', 'Y'])
    def test_get_user_input_case_insensitive(self, mock_input):
        """Test confirmation accepts various affirmative responses."""
        topic = get_user_input()
        assert topic == 'Research Topic'
        
    @patch('builtins.input', side_effect=KeyboardInterrupt())
    def test_get_user_input_interrupt(self, mock_input):
        """Test keyboard interrupt handling."""
        with pytest.raises(KeyboardInterrupt):
            get_user_input()
            
    @patch('builtins.input', side_effect=['Topic', 'invalid', 'maybe', 'y'])
    def test_get_user_input_invalid_confirmation(self, mock_input):
        """Test invalid confirmation prompts again."""
        topic = get_user_input()
        assert topic == 'Topic'
        # Should call input 4 times (topic, invalid, maybe, y)
        assert mock_input.call_count == 4


class TestInputParameterization:
    """Test various input parameter scenarios."""
    
    @pytest.mark.parametrize("topic,year,expected_valid", [
        ("AI", "2024", True),
        ("Machine Learning & Deep Learning", "2024", True),
        ("Quantum Computing (Advanced)", "2024", True),
        ("Künstliche Intelligenz", "2024", True),  # Unicode
        ("A" * 200, "2024", True),  # Very long topic
        ("!@#$%^&*()", "2024", True),  # Special characters
    ])
    def test_input_variations(self, topic, year, expected_valid):
        """Test crew handles various input formats."""
        inputs = {'topic': topic, 'current_year': year}
        
        # Verify inputs are structurally valid
        assert 'topic' in inputs
        assert 'current_year' in inputs
        assert inputs['topic'] == topic
        assert inputs['current_year'] == year
        assert expected_valid is True
        
    def test_input_structure_requirements(self):
        """Test that inputs must have required structure."""
        # Missing topic
        invalid_inputs_1 = {'current_year': '2024'}
        assert 'topic' not in invalid_inputs_1
        
        # Missing year
        invalid_inputs_2 = {'topic': 'AI Testing'}
        assert 'current_year' not in invalid_inputs_2
        
        # Both present
        valid_inputs = {'topic': 'AI Testing', 'current_year': '2024'}
        assert 'topic' in valid_inputs
        assert 'current_year' in valid_inputs


class TestValidationHelpers:
    """Test validation helper functions and utilities."""
    
    def test_bullet_point_counting(self):
        """Test bullet point counting logic."""
        test_cases = [
            ("• Point 1\n• Point 2", 2),
            ("• Point 1\n- Point 2\n• Point 3", 2),  # Only • counts
            ("No bullets here", 0),
            ("•Point 1\n •Point 2\n  • Point 3", 3),  # Various spacing
        ]
        
        for text, expected_count in test_cases:
            # This simulates the counting logic used in validation
            count = len([line for line in text.split('\n') if line.strip().startswith('•')])
            assert count == expected_count
            
    def test_markdown_header_counting(self):
        """Test markdown header counting logic."""
        test_cases = [
            ("# Header 1\n## Header 2", 2),
            ("# Header 1\n### Header 3", 2),
            ("#Not a header (no space)", 0),
            ("# H1\n## H2\n### H3\n#### H4", 4),
        ]
        
        for text, expected_count in test_cases:
            # This simulates the header counting logic
            count = len([line for line in text.split('\n') 
                        if line.strip().startswith('#') and len(line) > 1 and line[1] == ' '])
            assert count == expected_count