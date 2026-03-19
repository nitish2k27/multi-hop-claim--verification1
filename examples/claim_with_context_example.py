"""
Example: Processing Claim with User-Provided Context Documents

USE CASE: User provides BOTH claim + supporting documents
Scenario:
    User provides:
    - CLAIM: "Our company's revenue grew 50% in Q3"
    - DOCUMENTS:
        * financial_report_Q3.pdf
        * earnings_call_transcript.docx
        * competitor_analysis.xlsx
    
    User wants: Verify the claim USING these specific documents
"""

import sys
sys.path.append('..')

from src.preprocessing.input_processor import InputProcessor
import json


def example_1_claim_with_pdf():
    """Example 1: Claim + PDF Document"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Claim + PDF Document")
    print("="*60)
    
    processor = InputProcessor()
    
    # User provides claim + PDF report
    result = processor.process_with_context(
        claim_input="Our company's revenue grew 50% in Q3 2024",
        claim_type='text',
        context_documents=[
            {
                'data': 'data/user_uploads/Q3_financial_report.pdf',
                'type': 'pdf',
                'name': 'Q3 Financial Report'
            }
        ]
    )
    
    # Result structure
    print("\nCLAIM:")
    print(result['claim']['text'])
    print(f"Language: {result['claim']['language']}")
    
    print("\nCONTEXT DOCUMENTS:")
    for doc in result['context_documents']:
        print(f"- {doc['metadata'].get('document_name', 'Unnamed')}")
        print(f"  Type: {doc['source_type']}")
        print(f"  Length: {doc['metadata']['text_length']} chars")
        print(f"  Priority: {doc['priority']}")  # 'high' for user-provided
        print(f"  Preview: {doc['text'][:100]}...")
    
    return result


def example_2_claim_with_multiple_docs():
    """Example 2: Claim + Multiple Documents (Mixed Types)"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Claim + Multiple Documents (Mixed Types)")
    print("="*60)
    
    processor = InputProcessor()
    
    result = processor.process_with_context(
        claim_input="Climate change is accelerating faster than predicted",
        claim_type='text',
        context_documents=[
            {
                'data': 'data/user_uploads/ipcc_report.pdf',
                'type': 'pdf',
                'name': 'IPCC Climate Report 2024'
            },
            {
                'data': 'data/user_uploads/research_paper.docx',
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
    # Output: claim_with_context
    
    print(f"Number of context docs: {result['metadata']['num_context_docs']}")
    # Output: 3
    
    print(f"Context languages: {result['metadata']['context_languages']}")
    # Output: ['en']
    
    return result


def example_3_voice_claim_with_image():
    """Example 3: Voice Claim + Image Document"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Voice Claim + Image Document")
    print("="*60)
    
    processor = InputProcessor()
    
    # User speaks claim and provides image of data
    result = processor.process_with_context(
        claim_input={
            'data': 'data/user_uploads/user_claim.wav',
            'type': 'voice'
        },
        context_documents=[
            {
                'data': 'data/user_uploads/chart_screenshot.png',
                'type': 'image',
                'name': 'Sales Chart'
            }
        ]
    )
    
    print("\nCLAIM (from voice):")
    print(result['claim']['text'])
    print(f"Transcribed from: {result['claim']['metadata']['audio_path']}")
    
    print("\nCONTEXT (from image OCR):")
    print(result['context_documents'][0]['text'])
    print(f"OCR confidence: {result['context_documents'][0]['metadata']['avg_confidence']:.2f}")
    
    return result


def example_4_complete_workflow():
    """Example 4: Complete Workflow with RAG Integration"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Complete Workflow with RAG")
    print("="*60)
    
    from src.main_pipeline import FactVerificationPipeline
    
    # Initialize pipeline
    pipeline = FactVerificationPipeline()
    
    # Verify with context documents
    result = pipeline.verify_claim_with_context(
        claim_input="Our revenue grew 50% in Q3",
        context_documents=[
            {
                'data': 'data/user_uploads/financial_report.pdf',
                'type': 'pdf',
                'name': 'Q3 Financial Report'
            }
        ]
    )
    
    print("\nCLAIM:")
    print(result['claim'])
    
    print("\nVERDICT:")
    print(result['verification'].get('verdict', 'UNKNOWN'))
    
    print("\nEVIDENCE USED:")
    for i, ev in enumerate(result['evidence'][:3], 1):
        print(f"{i}. [{ev['source_type']}] {ev['text'][:100]}...")
        print(f"   Priority: {ev['priority']}, Source: {ev['source']}")
    
    return result


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("CLAIM WITH CONTEXT DOCUMENTS - EXAMPLES")
    print("="*60)
    
    # Example 1: Claim + PDF
    print("\nRunning Example 1...")
    # example_1_claim_with_pdf()
    
    # Example 2: Claim + Multiple docs
    print("\nRunning Example 2...")
    # example_2_claim_with_multiple_docs()
    
    # Example 3: Voice + Image
    print("\nRunning Example 3...")
    # example_3_voice_claim_with_image()
    
    # Example 4: Complete workflow
    print("\nRunning Example 4...")
    # example_4_complete_workflow()
    
    print("\n" + "="*60)
    print("HOW TO USE WITH RAG")
    print("="*60)
    print("""
    # The context documents are marked with:
    # - priority: 'high' (user-provided = high priority)
    # - is_user_provided: True
    
    # In your RAG system, you would:
    
    1. Index the context documents with HIGH priority
       for doc in result['context_documents']:
           if doc['is_user_provided']:
               rag_system.index(doc['text'], priority='high')
    
    2. When retrieving, prioritize user-provided docs
       retrieved_docs = rag_system.retrieve(
           query=result['claim']['text'],
           prioritize_user_docs=True
       )
    
    3. Verify claim against these specific documents
       verification_result = verify_claim(
           claim=result['claim']['text'],
           evidence=retrieved_docs
       )
    """)
    
    print("\n✓ Examples complete. Uncomment function calls to run.")


if __name__ == "__main__":
    main()
