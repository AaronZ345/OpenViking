"""Tests for rebuild admin endpoint."""

import httpx
import pytest

from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier
from tests.server.test_admin_api import ROOT_KEY
from tests.server.test_admin_api import admin_app as _admin_app_fixture
from tests.server.test_admin_api import admin_client as _admin_client_fixture
from tests.server.test_admin_api import admin_service as _admin_service_fixture

admin_service = _admin_service_fixture
admin_app = _admin_app_fixture
admin_client = _admin_client_fixture


async def test_rebuild_requires_admin_role(admin_client: httpx.AsyncClient):
    resp = await admin_client.post(
        "/api/v1/admin/rebuild",
        json={"uri": "viking://resources/demo", "mode": "vectors_only"},
    )
    assert resp.status_code == 401


async def test_rebuild_rejects_unsupported_uri(admin_client: httpx.AsyncClient):
    resp = await admin_client.post(
        "/api/v1/admin/rebuild",
        json={"uri": "viking://unknown/demo", "mode": "vectors_only"},
        headers={"X-API-Key": ROOT_KEY},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "UNSUPPORTED_URI"


@pytest.mark.asyncio
async def test_rebuild_resource_vectors_only_wait_true(monkeypatch):
    from openviking.server.routers.admin import RebuildRequest, rebuild

    seen = {}

    class FakeRebuildService:
        async def execute(self, *, uri, mode, wait, reason, ctx):
            seen["uri"] = uri
            seen["mode"] = mode
            seen["wait"] = wait
            seen["reason"] = reason
            seen["ctx"] = ctx
            return {
                "status": "completed",
                "uri": uri,
                "object_type": "resource",
                "mode": mode,
                "rebuilt_records": 1,
                "scanned_records": 1,
                "unsupported_records": 0,
                "failed_records": 0,
                "duration_ms": 12,
                "warnings": [],
            }

    ctx = RequestContext(
        user=UserIdentifier(account_id="test", user_id="alice", agent_id="default"),
        role=Role.ROOT,
    )
    request = RebuildRequest(uri="viking://resources/demo", mode="vectors_only", wait=True)

    monkeypatch.setattr(
        "openviking.server.routers.admin.get_rebuild_service",
        lambda: FakeRebuildService(),
    )

    response = await rebuild(request=request, ctx=ctx)

    assert response.status == "ok"
    assert response.result["status"] == "completed"
    assert response.result["object_type"] == "resource"
    assert response.result["rebuilt_records"] == 1
    assert seen["uri"] == "viking://resources/demo"
    assert seen["mode"] == "vectors_only"
    assert seen["wait"] is True
    assert seen["ctx"] == ctx


@pytest.mark.asyncio
async def test_rebuild_resource_vectors_only_wait_false(monkeypatch):
    from openviking.server.routers.admin import RebuildRequest, rebuild

    class FakeRebuildService:
        async def execute(self, *, uri, mode, wait, reason, ctx):
            return {
                "task_id": "rbld_123",
                "status": "accepted",
                "uri": uri,
                "object_type": "resource",
                "mode": mode,
            }

    ctx = RequestContext(
        user=UserIdentifier(account_id="test", user_id="alice", agent_id="default"),
        role=Role.ROOT,
    )
    request = RebuildRequest(uri="viking://resources/demo", mode="vectors_only", wait=False)

    monkeypatch.setattr(
        "openviking.server.routers.admin.get_rebuild_service",
        lambda: FakeRebuildService(),
    )

    response = await rebuild(request=request, ctx=ctx)

    assert response.status == "ok"
    assert response.result["status"] == "accepted"
    assert response.result["task_id"] == "rbld_123"
    assert response.result["object_type"] == "resource"


@pytest.mark.asyncio
async def test_rebuild_memory_rejects_semantic_and_vectors(monkeypatch):
    from openviking.service.rebuild_service import RebuildService

    service = RebuildService()

    with pytest.raises(Exception) as exc_info:
        await service.execute(
            uri="viking://user/default/memories/events",
            mode="semantic_and_vectors",
            wait=True,
            reason=None,
            ctx=RequestContext(
                user=UserIdentifier(account_id="test", user_id="alice", agent_id="default"),
                role=Role.ROOT,
            ),
        )

    assert getattr(exc_info.value, "code", None) == "UNSUPPORTED_MODE"


@pytest.mark.asyncio
async def test_rebuild_service_infers_skill_supports_semantic_and_vectors():
    from openviking.service.rebuild_service import RebuildService

    service = RebuildService()
    service._validate_mode("skill", "semantic_and_vectors")


@pytest.mark.asyncio
async def test_rebuild_fetch_existing_record_uses_fetch_by_uri(monkeypatch):
    from openviking.service.rebuild_service import RebuildService

    class FakeVikingDB:
        def __init__(self):
            self.fetch_calls = []
            self.lookup_calls = []

        async def fetch_by_uri(self, uri, *, ctx):
            self.fetch_calls.append((uri, ctx))
            return {"uri": uri, "level": 2, "abstract": "from-fetch"}

        async def get_context_by_uri(self, uri, owner_space=None, level=None, limit=1, *, ctx):
            self.lookup_calls.append((uri, owner_space, level, limit, ctx))
            return [{"uri": uri, "level": level, "active_count": 1}]

    fake_service = type("Svc", (), {"vikingdb_manager": FakeVikingDB()})()
    monkeypatch.setattr("openviking.service.rebuild_service.get_service", lambda: fake_service)

    service = RebuildService()
    ctx = RequestContext(
        user=UserIdentifier(account_id="test", user_id="alice", agent_id="default"),
        role=Role.ROOT,
    )

    record = await service._fetch_existing_record(
        uri="viking://resources/demo.txt",
        level=2,
        ctx=ctx,
    )

    assert record["abstract"] == "from-fetch"
    assert fake_service.vikingdb_manager.fetch_calls
    assert not fake_service.vikingdb_manager.lookup_calls


@pytest.mark.asyncio
async def test_rebuild_resource_vectors_only_continues_after_single_record_failure(monkeypatch):
    from openviking.service.rebuild_service import RebuildService, _RebuildCounters
    from openviking_cli.exceptions import OpenVikingError

    class FakeVikingFS:
        async def tree(self, uri, output="original", show_all_hidden=True, ctx=None):
            return [
                {"uri": "viking://resources/demo/bad.txt", "isDir": False},
                {"uri": "viking://resources/demo/good.txt", "isDir": False},
            ]

    async def fake_read_directory_abstract(self, uri, *, ctx):
        return ""

    async def fake_read_directory_overview(self, uri, *, ctx):
        return ""

    async def fake_best_file_summary(self, uri, *, ctx):
        return f"summary:{uri.rsplit('/', 1)[-1]}"

    async def fake_best_resource_file_vector_text(self, uri, summary, ctx):
        return summary

    seen = []

    async def fake_upsert_context(self, **kwargs):
        seen.append(kwargs["uri"])
        if kwargs["uri"].endswith("bad.txt"):
            raise OpenVikingError("boom", code="PROCESSING_ERROR")

    monkeypatch.setattr("openviking.service.rebuild_service.get_viking_fs", lambda: FakeVikingFS())
    monkeypatch.setattr(RebuildService, "_read_directory_abstract", fake_read_directory_abstract)
    monkeypatch.setattr(RebuildService, "_read_directory_overview", fake_read_directory_overview)
    monkeypatch.setattr(RebuildService, "_best_file_summary", fake_best_file_summary)
    monkeypatch.setattr(
        RebuildService,
        "_best_resource_file_vector_text",
        fake_best_resource_file_vector_text,
    )
    monkeypatch.setattr(RebuildService, "_upsert_context", fake_upsert_context)

    service = RebuildService()
    counters = _RebuildCounters()
    ctx = RequestContext(
        user=UserIdentifier(account_id="test", user_id="alice", agent_id="default"),
        role=Role.ROOT,
    )

    await service._rebuild_resource_vectors(
        uri="viking://resources/demo",
        counters=counters,
        ctx=ctx,
    )

    assert seen == [
        "viking://resources/demo/bad.txt",
        "viking://resources/demo/good.txt",
    ]
    assert counters.failed_records == 1
    assert counters.rebuilt_records == 1


@pytest.mark.asyncio
async def test_rebuild_semantic_processor_runs_with_skip_vectorization(monkeypatch):
    from openviking.service.rebuild_service import RebuildService

    seen = {}

    class FakeSemanticProcessor:
        async def on_dequeue(self, payload):
            seen["payload"] = payload

    monkeypatch.setattr(
        "openviking.service.rebuild_service.SemanticProcessor",
        FakeSemanticProcessor,
    )

    service = RebuildService()
    ctx = RequestContext(
        user=UserIdentifier(account_id="test", user_id="alice", agent_id="default"),
        role=Role.ROOT,
    )

    await service._run_semantic_processor(
        uri="viking://resources/demo",
        context_type="resource",
        ctx=ctx,
    )

    import json

    msg = json.loads(seen["payload"]["data"])
    assert msg["skip_vectorization"] is True
