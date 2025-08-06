#!/usr/bin/env python3
"""
Validation script for CoT parsing logic
Tests the parsing against the example fixture
"""

import json
import os

def parse_cot_from_raw(raw_answer):
    """Parse CoT structure from raw model answer"""
    raw = raw_answer.strip()
    
    # Primary heuristic: Split on the delimiter "####" (four hashes) if present
    if '####' in raw:
        parts = raw.split('####', 1)
        cot_text, ans_text = parts[0].strip(), parts[1].strip()
    elif "So the answer is" in raw:
        # Fallback heuristic: Split on "So the answer is"
        parts = raw.rsplit("So the answer is", 1)
        cot_text, ans_text = parts[0].strip(), parts[1].strip()
    elif "Answer:" in raw:
        # Fallback heuristic: Split on "Answer:"
        parts = raw.rsplit("Answer:", 1)
        cot_text, ans_text = parts[0].strip(), parts[1].strip()
    elif "Therefore" in raw:
        # Fallback heuristic: Split on "Therefore"
        parts = raw.rsplit("Therefore", 1)
        cot_text, ans_text = parts[0].strip(), "Therefore" + parts[1].strip()
    elif "The answer is" in raw:
        # Fallback heuristic: Split on "The answer is"
        parts = raw.rsplit("The answer is", 1)
        cot_text, ans_text = parts[0].strip(), "The answer is" + parts[1].strip()
    else:
        # Final fallback: Split on last newline
        lines = raw.splitlines()
        if len(lines) > 1:
            cot_text, ans_text = "\n".join(lines[:-1]), lines[-1]
        else:
            cot_text, ans_text = raw, raw
    
    # Convert to structured data
    cot_steps = [step.strip() for step in cot_text.split('\n') if step.strip()]
    final_answer = ans_text.strip()
    
    return {
        'cot_steps': cot_steps,
        'final_answer': final_answer,
        'cot_text': cot_text,
        'ans_text': ans_text
    }

def test_cot_parsing():
    """Test CoT parsing with the example fixture"""
    
    # Load the example fixture
    fixture_path = "test_fixtures/cot_example.json"
    if not os.path.exists(fixture_path):
        print(f"❌ Fixture not found: {fixture_path}")
        return False
    
    with open(fixture_path, 'r') as f:
        example = json.load(f)
    
    print("=== Testing CoT Parsing Logic ===")
    print(f"Question: {example['question'][:100]}...")
    print(f"Raw answer: {example['answer'][:100]}...")
    print()
    
    # Parse the answer
    cot_data = parse_cot_from_raw(example['answer'])
    
    print("=== Parsing Results ===")
    print(f"Final answer: {cot_data['final_answer']}")
    print(f"Number of steps: {len(cot_data['cot_steps'])}")
    print("CoT steps:")
    for i, step in enumerate(cot_data['cot_steps']):
        print(f"  {i+1}. {step}")
    print()
    
    # Validate against expected results
    print("=== Validation ===")
    expected_answer = example['expected_final_answer']
    expected_steps = example['expected_cot_steps']
    
    answer_correct = cot_data['final_answer'] == expected_answer
    steps_correct = cot_data['cot_steps'] == expected_steps
    
    print(f"Answer matches expected: {answer_correct}")
    print(f"Steps match expected: {steps_correct}")
    
    if answer_correct and steps_correct:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        if not answer_correct:
            print(f"  Expected answer: '{expected_answer}', got: '{cot_data['final_answer']}'")
        if not steps_correct:
            print(f"  Expected steps: {expected_steps}")
            print(f"  Got steps: {cot_data['cot_steps']}")
        return False

if __name__ == "__main__":
    success = test_cot_parsing()
    exit(0 if success else 1) 