"""
Test input processor with context documents
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.input_processor import InputProcessor
import json


def test_claim_with_pdf():
    """Test: Claim + PDF document"""
    print("\n" + "="*60)
    print("TEST 1: Claim + PDF Document")
    print("="*60)
    
    processor = InputProcessor()
    
    result = processor.process_with_context(
        claim_input="India's GDP grew 8% in 2024",
        claim_type='text',
        context_documents=[
            {
                'data': 'test_data/economic_report.pdf',
                'type': 'pdf',
                'name': 'Economic Report 2024'
            }
        ]
    )
    
    print("\nCLAIM:")
    print(f"  Text: {result['claim']['text']}")
    print(f"  Language: {result['claim']['language']}")
    
    print("\nCONTEXT DOCUMENTS:")
    for doc in result['context_documents']:
        print(f"  - {doc['metadata'].get('document_name', 'Unnamed')}")
        print(f"    Priority: {doc['priority']}")
        print(f"    Length: {len(doc['text'])} chars")
        print(f"    Preview: {doc['text'][:200]}...")
    
    print("\nMETADATA:")
    print(f"  Processing mode: {result['processing_mode']}")
    print(f"  Total context docs: {result['metadata']['num_context_docs']}")
    print(f"  Total context length: {result['metadata']['total_context_length']} chars")
    
    return result


def test_claim_with_multiple_docs():
    """Test: Claim + Multiple documents (mixed types)"""
    print("\n" + "="*60)
    print("TEST 2: Claim + Multiple Documents (Mixed Types)")
    print("="*60)
    
    processor = InputProcessor()
    
    result = processor.process_with_context(
        claim_input="Climate change is accelerating faster than predicted",
        claim_type='text',
        context_documents=[
            {
                'data': 'test_data/ipcc_report.pdf',
                'type': 'pdf',
                'name': 'IPCC Climate Report 2024'
            },
            {
                'data': 'test_data/research_paper.docx',
                'type': 'docx',
                'name': 'MIT Climate Study'
            },
            {
                'data': 'https://www.nature.com/articles/climate-2024',
                'type': 'url',
                'name': 'Nature Article on Climate'
            }
        ]
    )
    
    print(f"\nProcessing mode: {result['processing_mode']}")
    print(f"Number of context docs: {result['metadata']['num_context_docs']}")
    print(f"Context languages: {result['metadata']['context_languages']}")
    
    print("\nCONTEXT DOCUMENTS:")
    for i, doc in enumerate(result['context_documents'], 1):
        print(f"\n{i}. {doc['metadata'].get('document_name', 'Unnamed')}")
        print(f"   Type: {doc['source_type']}")
        print(f"   Language: {doc['language']}")
        print(f"   Priority: {doc['priority']}")
        print(f"   User Provided: {doc['is_user_provided']}")
        print(f"   Length: {len(doc['text'])} chars")
    
    return result


def test_voice_claim_with_image():
    """Test: Voice claim + Image document"""
    print("\n" + "="*60)
    print("TEST 3: Voice Claim + Image Document")
    print("="*60)
    
    processor = InputProcessor()
    
    # User speaks claim and provides image of data
    result = processor.process_with_context(
        claim_input={
            'data': 'test_data/user_claim.wav',
            'type': 'voice'
        },
        context_documents=[
            {
                'data': 'test_data/chart_screenshot.png',
                'type': 'image',
                'name': 'Sales Chart'
            }
        ]
    )
    
    print("\nCLAIM (from voice):")
    print(f"  Text: {result['claim']['text']}")
    print(f"  Transcribed from: {result['claim']['metadata']['audio_path']}")
    print(f"  Duration: {result['claim']['metadata']['duration_seconds']:.2f}s")
    
    print("\nCONTEXT (from image OCR):")
    print(f"  Text: {result['context_documents'][0]['text']}")
    print(f"  OCR confidence: {result['context_documents'][0]['metadata']['avg_confidence']:.2f}")
    
    return result


def test_text_only_claim():
    """Test: Text claim only (no context)"""
    print("\n" + "="*60)
    print("TEST 4: Text Claim Only (No Context)")
    print("="*60)
    
    processor = InputProcessor()
    
    result = processor.process(
        "Our company's revenue grew 50% in Q3 2024",
        'text'
    )
    
    print("\nCLAIM:")
    print(f"  Text: {result['text']}")
    print(f"  Language: {result['language']}")
    print(f"  Source Type: {result['source_type']}")
    
    print("\nMETADATA:")
    print(json.dumps(result['metadata'], indent=2))
    
    return result


def test_batch_processing():
    """Test: Batch processing multiple inputs"""
    print("\n" + "="*60)
    print("TEST 5: Batch Processing")
    print("="*60)
    
    processor = InputProcessor()
    
    inputs = [
        {'data': 'Climate change is real', 'type': 'text'},
        {'data': 'test_data/report.pdf', 'type': 'pdf'},
        {'data': 'test_data/audio.wav', 'type': 'voice'}
    ]
    
    results = processor.process_batch(inputs)
    
    print(f"\nProcessed {len(results)} inputs")
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"\n{i}. FAILED: {result['error']}")
        else:
            print(f"\n{i}. SUCCESS")
            print(f"   Type: {result['source_type']}")
            print(f"   Language: {result['language']}")
            print(f"   Text length: {len(result['text'])} chars")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("INPUT PROCESSOR WITH CONTEXT - TEST SUITE")
    print("="*60)
    
    # Run tests
    try:
        # Test 1: Claim + PDF
        # test_claim_with_pdf()
        
        # Test 2: Claim + Multiple docs
        # test_claim_with_multiple_docs()
        
        # Test 3: Voice + Image
        # test_voice_claim_with_image()
        
        # Test 4: Text only
        test_text_only_claim()
        
        # Test 5: Batch processing
        # test_batch_processing()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
