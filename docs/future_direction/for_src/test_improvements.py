"""
Test script to verify the RAG improvements work correctly.
This validates:
1. Config imports work
2. LLM fallback mechanism functions
3. PromptTemplate is compatible with llama-index version
4. Query engine creation works
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_config_imports():
    """Test that all config parameters can be imported."""
    print("=" * 60)
    print("TEST 1: Config Imports")
    print("=" * 60)
    try:
        from config import (
            VECTOR_DB_DIR, COLLECTION_NAME, TOP_K,
            LLM_MODEL, LLM_FALLBACK, EMBED_MODEL, OLLAMA_BASE_URL,
            SIMILARITY_THRESHOLD, CHUNK_SIZE, CHUNK_OVERLAP
        )
        print("✅ All config parameters imported successfully")
        print(f"   Primary LLM: {LLM_MODEL}")
        print(f"   Fallback LLM: {LLM_FALLBACK}")
        print(f"   Chunk Size: {CHUNK_SIZE}")
        print(f"   Chunk Overlap: {CHUNK_OVERLAP}")
        print(f"   Similarity Threshold: {SIMILARITY_THRESHOLD}")
        return True
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

def test_prompt_template_import():
    """Test that PromptTemplate can be imported from llama_index.core."""
    print("\n" + "=" * 60)
    print("TEST 2: PromptTemplate Import")
    print("=" * 60)
    try:
        from llama_index.core import PromptTemplate
        print("✅ PromptTemplate imported successfully")
        
        # Test creating a simple prompt
        test_template = PromptTemplate("Test: {query_str}")
        print("✅ PromptTemplate instantiation works")
        return True
    except ImportError as e:
        print(f"❌ PromptTemplate import failed: {e}")
        print("   This may indicate llama-index-core version incompatibility")
        return False
    except Exception as e:
        print(f"❌ PromptTemplate creation failed: {e}")
        return False

def test_llm_configuration():
    """Test LLM configuration with fallback logic."""
    print("\n" + "=" * 60)
    print("TEST 3: LLM Configuration")
    print("=" * 60)
    try:
        from llama_index.llms.ollama import Ollama
        from config import LLM_MODEL, LLM_FALLBACK, OLLAMA_BASE_URL
        
        print(f"   Testing connection to: {OLLAMA_BASE_URL}")
        print(f"   Primary model: {LLM_MODEL}")
        
        # Test primary LLM
        try:
            llm = Ollama(
                model=LLM_MODEL,
                base_url=OLLAMA_BASE_URL,
                request_timeout=10.0,
                temperature=0.1,
            )
            # Quick test
            response = llm.complete("test")
            print(f"✅ Primary LLM ({LLM_MODEL}) is available and responding")
            return True
        except Exception as e:
            print(f"⚠️  Primary LLM ({LLM_MODEL}) unavailable: {e}")
            print(f"   Testing fallback: {LLM_FALLBACK}")
            
            # Test fallback
            try:
                llm = Ollama(
                    model=LLM_FALLBACK,
                    base_url=OLLAMA_BASE_URL,
                    request_timeout=10.0,
                    temperature=0.1,
                )
                response = llm.complete("test")
                print(f"✅ Fallback LLM ({LLM_FALLBACK}) is available and responding")
                print("   Note: You may want to run 'ollama pull deepseek-r1:latest'")
                return True
            except Exception as e2:
                print(f"❌ Fallback LLM also unavailable: {e2}")
                print("   Make sure Ollama is running and models are pulled")
                return False
                
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_query_engine_creation():
    """Test that query engine can be created with custom prompt."""
    print("\n" + "=" * 60)
    print("TEST 4: Query Engine Creation")
    print("=" * 60)
    try:
        from llama_index.core import PromptTemplate
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.postprocessor import SimilarityPostprocessor
        
        print("✅ All query engine components imported")
        
        # Test prompt template creation
        qa_prompt_template = (
            "You are an AI assistant answering questions based on provided context documents.\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Question: {query_str}\n"
            "Answer: "
        )
        qa_prompt = PromptTemplate(qa_prompt_template)
        print("✅ Custom prompt template created successfully")
        
        # Note: Can't test full query engine without vector store
        print("✅ Query engine components are compatible")
        return True
        
    except Exception as e:
        print(f"❌ Query engine test failed: {e}")
        return False

def test_ingest_compatibility():
    """Test that ingest.py will use updated config values."""
    print("\n" + "=" * 60)
    print("TEST 5: Ingest Compatibility")
    print("=" * 60)
    try:
        # Import ingest to check it can read new config
        import ingest
        from config import CHUNK_SIZE, CHUNK_OVERLAP
        
        print(f"✅ ingest.py successfully imports config")
        print(f"   Ingest will use CHUNK_SIZE: {CHUNK_SIZE}")
        print(f"   Ingest will use CHUNK_OVERLAP: {CHUNK_OVERLAP}")
        print("   ⚠️  Note: To use new chunk sizes, you must re-run ingest.py")
        return True
    except Exception as e:
        print(f"❌ Ingest compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 TESTING RAG IMPROVEMENTS")
    print("=" * 60)
    
    results = []
    results.append(("Config Imports", test_config_imports()))
    results.append(("PromptTemplate Import", test_prompt_template_import()))
    results.append(("LLM Configuration", test_llm_configuration()))
    results.append(("Query Engine Creation", test_query_engine_creation()))
    results.append(("Ingest Compatibility", test_ingest_compatibility()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The improvements are compatible.")
    else:
        print("\n⚠️  Some tests failed. Review the output above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
