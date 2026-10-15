"""Explicit CI-only PostgreSQL/Qdrant roundtrip. Never use local fallback storage."""
import os
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main():
    if os.environ.get("CI_ISOLATED_SERVICES") != "1":
        raise RuntimeError("Run only against disposable CI services with CI_ISOLATED_SERVICES=1")
    from sqlalchemy import text
    from qdrant_client import QdrantClient, models
    from app.config import get_settings
    from app.db.postgres import get_engine
    settings = get_settings()
    with get_engine().connect() as connection:
        assert connection.execute(text("SELECT 1")).scalar_one() == 1
    client = QdrantClient(url=settings.qdrant_url, timeout=10, check_compatibility=False)
    name = "ci_probe_" + uuid4().hex
    created = False
    try:
        client.create_collection(name, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        created = True
        client.upsert(name, points=[models.PointStruct(id=1, vector=[1.0, 0.0], payload={"probe": name})], wait=True)
        point = client.retrieve(name, ids=[1], with_payload=True)[0]
        assert point.payload["probe"] == name
        assert client.count(name, exact=True).count == 1
    finally:
        if created:
            client.delete_collection(name)
        client.close()
    print("CI_SERVICE_SMOKE_OK: postgres reachable; qdrant server roundtrip passed")


if __name__ == "__main__":
    main()
