import sys
import os
import requests
import django

sys.stdout.reconfigure(encoding='utf-8')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cyber_physical_banking.settings')
django.setup()

from django.conf import settings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from ml_models.rag_pipeline import _get_rag_config, _get_or_build_profile, query_dataset_rag
from api.services import chat_with_dataset_service

print("==================================================")
print("1. CHECK CONFIGURATION & COMPATIBILITY")
print("==================================================")
cfg = _get_rag_config()
for k, v in cfg.items():
    print(f"  {k}: {v}")

print("\n==================================================")
print("2. VERIFY OLLAMA CONNECTIVITY & MODELS")
print("==================================================")
try:
    r = requests.get(f"{cfg['base_url']}/api/tags", timeout=5)
    print(f"  Ollama HTTP GET /api/tags -> Status: {r.status_code}")
    models = [m.get('name') for m in r.json().get('models', [])]
    print(f"  Available models in Ollama: {models}")
    assert any('nomic-embed-text' in m for m in models), "nomic-embed-text not found!"
    assert any('llama3' in m for m in models), "llama3 not found!"
    print("  [SUCCESS] Both nomic-embed-text and llama3 are present in Ollama.")
except Exception as e:
    print(f"  [ERROR] Ollama verification failed: {e}")
    sys.exit(1)

print("\n==================================================")
print("3. INDEPENDENT EMBEDDING GENERATION TEST")
print("==================================================")
try:
    emb = OllamaEmbeddings(model=cfg["embedding_model"], base_url=cfg["base_url"])
    sample_text = "Summarize the average balance and transaction count across customers."
    vector = emb.embed_query(sample_text)
    print(f"  Embedding model: '{cfg['embedding_model']}'")
    print(f"  Vector generated successfully! Dimension: {len(vector)}")
    assert len(vector) == 768, f"Expected 768 dimensions, got {len(vector)}"
    print("  [SUCCESS] nomic-embed-text generated valid 768-dim embeddings.")
except Exception as e:
    print(f"  [ERROR] Embedding generation failed: {e}")
    sys.exit(1)

print("\n==================================================")
print("4. CHROMADB PERSISTENT INSERTION & RETRIEVAL TEST")
print("==================================================")
try:
    from langchain_chroma import Chroma
    test_collection = "test_verification_collection"
    vs = Chroma(
        collection_name=test_collection,
        embedding_function=emb,
        persist_directory=cfg["chroma_persist_dir"],
    )
    doc_id = vs.add_documents([
        Document(
            page_content="Customer balance average is $50,699.12 and transaction count is 257.77.",
            metadata={"source": "test_verification"}
        )
    ])
    print(f"  Document inserted with ID: {doc_id}")
    results = vs.similarity_search("What is the customer balance average?", k=1)
    print(f"  Semantic query returned: '{results[0].page_content}'")
    assert "50,699.12" in results[0].page_content
    print("  [SUCCESS] ChromaDB vector insertion and semantic retrieval working perfectly.")
except Exception as e:
    print(f"  [ERROR] ChromaDB test failed: {e}")
    sys.exit(1)

print("\n==================================================")
print("5. DATASET PROFILE GENERATION & CACHING TEST")
print("==================================================")
dataset_path = os.path.join(cfg["base_dir"], "media", "datasets", "cyber_physical_customers.csv")
assert os.path.exists(dataset_path), f"Dataset not found at {dataset_path}"
try:
    profile = _get_or_build_profile(dataset_path, emb)
    print(f"  Profile built/retrieved successfully! Total characters: {len(profile)}")
    assert "=== Dataset Profile: cyber_physical_customers.csv ===" in profile
    assert "balance" in profile
    assert "transaction_count" in profile
    print("  [SUCCESS] Dataset profile cached in ChromaDB.")
except Exception as e:
    print(f"  [ERROR] Profile building/retrieval failed: {e}")
    sys.exit(1)

print("\n==================================================")
print("6. END-TO-END RAG PIPELINE & LLM ANSWER TEST")
print("==================================================")
query = "Summarize the average balance and transaction count across customers."
try:
    answer = query_dataset_rag(dataset_path, query)
    print(f"  Query: '{query}'")
    print(f"  Answer from Llama 3:\n{answer}\n")
    # Verify data correctness: actual dataset values are ~50699 and ~257
    assert "50699" in answer or "50,699" in answer or "50700" in answer, "Average balance not reflected in answer!"
    assert "257" in answer or "258" in answer, "Average transaction count not reflected in answer!"
    print("  [SUCCESS] RAG pipeline completed successfully without any HTTP 501 or embedding errors.")
except Exception as e:
    print(f"  [ERROR] RAG pipeline failed: {e}")
    sys.exit(1)

print("\n==================================================")
print("7. DJANGO SERVICE LAYER API TEST")
print("==================================================")
try:
    service_result = chat_with_dataset_service("cyber_physical_customers.csv", query)
    print(f"  Service response success: {service_result['success']}")
    assert service_result['success'] is True
    print("  [SUCCESS] Django chat service returned successful answer.")
except Exception as e:
    print(f"  [ERROR] Django chat service failed: {e}")
    sys.exit(1)

print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")

