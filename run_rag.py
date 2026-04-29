#!/usr/bin/env python3
"""
FinSight AI RAG System Runner
Simple script to demonstrate RAG functionality
"""

from agents.rag_agent import run


def main():
    """Demo queries for the RAG system."""
    
    # Example questions
    test_queries = [
        'What are Microsoft total revenues?',
        'Describe Amazon cloud services',
        'What risk factors does Meta mention?',
        'How does Alphabet report advertising revenue?',
    ]
    
    print("=" * 60)
    print("FinSight AI RAG System - Demo")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuestion: {query}")
        print("-" * 60)
        
        try:
            state = {'query': query}
            result = run(state)
            
            # Display answer
            print(result['rag_result'])
            
            # Show sources
            sources = result.get('sources', [])
            if sources:
                print(f"\nSources: {len(set(sources))} unique document(s)")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()


if __name__ == "__main__":
    main()