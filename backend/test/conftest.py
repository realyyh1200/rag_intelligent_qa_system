import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import SessionLocal, engine
from services.rag_service import RAGService
from services.qdrant_service import QdrantService
from models.rag import RAGFile, RAGChunk
from models.user import User
from sqlalchemy import text


TEST_USER_ID = 999
TEST_USER_USERNAME = "benchmark_user"
TEST_USER_EMAIL = "benchmark@test.com"


@pytest.fixture(scope="session")
def db_session():
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture(scope="session")
def test_user(db_session):
    existing_user = db_session.query(User).filter(User.id == TEST_USER_ID).first()
    
    if existing_user:
        user = existing_user
    else:
        user = User(
            id=TEST_USER_ID,
            username=TEST_USER_USERNAME,
            email=TEST_USER_EMAIL,
            hashed_password="$2b$12$test_hashed_password_for_benchmark",
            is_active=True
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
    
    yield user


@pytest.fixture(scope="session")
def qdrant_service():
    return QdrantService()


@pytest.fixture(scope="session")
def setup_test_data(db_session, test_user, qdrant_service):
    print(f"\n[Setup] Creating test data for user_id={test_user.id}")
    
    rag_service = RAGService(db_session, test_user.id)
    
    test_doc_path = Path(__file__).parent / "data" / "test_document.txt"
    
    if not test_doc_path.exists():
        print(f"[Setup] Test document not found at {test_doc_path}")
        yield
        return
    
    try:
        result = rag_service.process_files([str(test_doc_path)])
        print(f"[Setup] Test document processed: {result}")
    except Exception as e:
        print(f"[Setup] Error processing test document: {e}")
    
    yield
    
    print(f"\n[Teardown] Cleaning up test data for user_id={test_user.id}")
    
    try:
        db_session.query(RAGChunk).filter(RAGChunk.file_id.in_(
            db_session.query(RAGFile.id).filter(RAGFile.user_id == test_user.id)
        )).delete(synchronize_session=False)
        db_session.query(RAGFile).filter(RAGFile.user_id == test_user.id).delete(synchronize_session=False)
        db_session.commit()
        print("[Teardown] MySQL data cleaned")
    except Exception as e:
        print(f"[Teardown] Error cleaning MySQL: {e}")
    
    try:
        from core.config import settings
        qdrant_service._client.delete(
            collection_name=settings.QDRANT_RAG_COLLECTION,
            points_filter={
                "must": [
                    {"key": "user_id", "match": {"value": test_user.id}}
                ]
            }
        )
        print("[Teardown] Qdrant data cleaned")
    except Exception as e:
        print(f"[Teardown] Error cleaning Qdrant: {e}")


@pytest.fixture(scope="function")
def rag_service(db_session, test_user):
    return RAGService(db_session, test_user.id)


@pytest.fixture(scope="session")
def benchmark_test_cases():
    from test.data.test_cases import RAG_BENCHMARK_TEST_CASES
    return RAG_BENCHMARK_TEST_CASES


@pytest.fixture(scope="session")
def ragas_config():
    try:
        from ragas import RunConfig
        return RunConfig(
            timeout=60,
            max_retries=3,
            max_wait=120,
            max_workers=4,
        )
    except ImportError:
        return None


@pytest.fixture(scope="session")
def ragas_llm():
    from core.config import settings
    
    try:
        from langchain_anthropic import ChatAnthropic
        
        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            temperature=0,
            timeout=60,
        )
    except ImportError:
        try:
            from langchain_openai import ChatOpenAI
            
            return ChatOpenAI(
                model=settings.ANTHROPIC_MODEL if "gpt" in settings.ANTHROPIC_MODEL.lower() else "gpt-3.5-turbo",
                temperature=0,
                timeout=60,
            )
        except ImportError:
            return None


def pytest_configure(config):
    try:
        import ragas
        from ragas import RunConfig
        
        run_config = RunConfig(
            timeout=60,
            max_retries=3,
            max_wait=120,
            max_workers=4,
        )
        
        ragas.run_config = run_config
        
        print("\n[Ragas] Configuration loaded successfully")
    except ImportError:
        print("\n[Ragas] Not installed, skipping ragas configuration")
