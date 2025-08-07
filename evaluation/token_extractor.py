#!/usr/bin/env python3
"""
Token Extractor for JSONL files

This script extracts tokens after a given term from each JSON object in a JSONL file
and calculates the average length of those tokens across the entire file.
"""

import json
import argparse
import sys
from typing import List, Dict, Any


def extract_tokens_after_term(json_obj: Dict[str, Any], term: str) -> str:
    """
    Extract tokens after a given term from a JSON object.
    
    Args:
        json_obj: The JSON object to search in
        term: The term to search for (e.g., "solution":)
    
    Returns:
        The string content after the term
    """
    # Convert JSON object to string for searching
    json_str = json.dumps(json_obj, ensure_ascii=False)
    
    # Find the exact term in the JSON string
    term_index = json_str.find(term)
    if term_index == -1:
        return ""
    
    # Extract everything after the term
    content_after_term = json_str[term_index + len(term):]
    
    # Remove leading/trailing whitespace
    content_after_term = content_after_term.strip()
    
    # If the content starts with a quote, find the matching closing quote
    if content_after_term.startswith('"'):
        # Find the closing quote (handle escaped quotes)
        quote_end = -1
        i = 1  # Skip the opening quote
        while i < len(content_after_term):
            if content_after_term[i] == '"' and content_after_term[i-1] != '\\':
                quote_end = i
                break
            i += 1
        
        if quote_end != -1:
            content_after_term = content_after_term[1:quote_end]
    
    return content_after_term


def process_jsonl_file(file_path: str, term: str) -> List[int]:
    """
    Process a JSONL file and extract token lengths after the given term.
    
    Args:
        file_path: Path to the JSONL file
        term: The term to search for
    
    Returns:
        List of token lengths for each JSON object
    """
    token_lengths = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    json_obj = json.loads(line)
                    extracted_content = extract_tokens_after_term(json_obj, term)
                    
                    if extracted_content:
                        # Split into tokens (words) and count them
                        tokens = extracted_content.split()
                        token_lengths.append(len(tokens))
                    else:
                        print(f"Line {line_num}: No content found after '{term}'")
                        
                except json.JSONDecodeError as e:
                    print(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    return token_lengths


def calculate_average_length(token_lengths: List[int]) -> float:
    """
    Calculate the average length of tokens.
    
    Args:
        token_lengths: List of token counts
    
    Returns:
        Average length as a float
    """
    if not token_lengths:
        return 0.0
    
    return sum(token_lengths) / len(token_lengths)


def main():
    parser = argparse.ArgumentParser(
        description="Extract tokens after a given term from JSONL file and calculate average length"
    )
    parser.add_argument(
        "--file_path", 
        required=True,
        help="Path to the JSONL file"
    )
    parser.add_argument(
        "--term", 
        required=True,
        help="The term to search for (e.g., 'solution:')"
    )
    
    args = parser.parse_args()
    
    print(f"Processing file: {args.file_path}")
    print(f"Searching for term: '{args.term}'")
    print("-" * 50)
    
    # Process the JSONL file
    token_lengths = process_jsonl_file(args.file_path, args.term)
    
    if token_lengths:
        avg_length = calculate_average_length(token_lengths)
        print("-" * 50)
        print(f"Total JSON objects processed: {len(token_lengths)}")
        print(f"Average token length after '{args.term}': {avg_length:.2f}")
        print(f"Total tokens found: {sum(token_lengths)}")
    else:
        print("No valid data found in the file.")


if __name__ == "__main__":
    main() 