"""
Tests for HITLService.

Tests the Human-in-the-Loop service functionality including request handling,
timeout management, and replanning approval workflows.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.hitl_request import (
    HITLRequest, HITLResponse, HITLRequestType, HITLRequestStatus
)
from roma.application.services.hitl_service import HITLService
from roma.domain.context import TaskContext


@pytest.fixture
def hitl_service():
    """Create HITL service instance."""
    return HITLService(
        enabled=True,
        default_timeout_seconds=300
    )


@pytest.fixture
def disabled_hitl_service():
    """Create disabled HITL service instance."""
    return HITLService(
        enabled=False,
        default_timeout_seconds=300
    )


@pytest.fixture
def sample_task():
    """Create a sample task node."""
    return TaskNode(
        task_id=str(uuid4()),
        goal="Complex analysis requiring human oversight",
        task_type=TaskType.THINK,
        status=TaskStatus.NEEDS_REPLAN,
        node_type=NodeType.PLAN
    )


@pytest.fixture
def sample_context():
    """Create a sample task context."""
    return TaskContext(
        task=TaskNode(
            task_id=str(uuid4()),
            goal="Root task",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        ),
        overall_objective="Complete analysis",
        execution_id="test-hitl-execution-id",
        execution_metadata={}
    )


@pytest.fixture
def sample_failed_children():
    """Create sample failed child tasks."""
    parent_id = str(uuid4())
    return [
        TaskNode(
            task_id=str(uuid4()),
            goal="Failed child 1",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.FAILED,
            parent_id=parent_id
        ),
        TaskNode(
            task_id=str(uuid4()),
            goal="Failed child 2",
            task_type=TaskType.THINK,
            status=TaskStatus.FAILED,
            parent_id=parent_id
        )
    ]


class TestHITLService:
    """Test HITLService functionality."""

    def test_initialization_enabled(self, hitl_service):
        """Test HITL service initialization when enabled."""
        assert hitl_service.enabled is True
        assert hitl_service.default_timeout_seconds == 300
        assert len(hitl_service._pending_requests) == 0

    def test_initialization_disabled(self, disabled_hitl_service):
        """Test HITL service initialization when disabled."""
        assert disabled_hitl_service.enabled is False
        assert disabled_hitl_service.default_timeout_seconds == 300
        assert len(disabled_hitl_service._pending_requests) == 0

    @pytest.mark.asyncio
    async def test_request_replanning_approval_disabled_service(self, disabled_hitl_service, sample_task, sample_context, sample_failed_children):
        """Test that disabled HITL service returns None immediately."""
        result = await disabled_hitl_service.request_replanning_approval(
            node=sample_task,
            context=sample_context,
            failed_children=sample_failed_children,
            failure_reason="Test failure"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_request_replanning_approval_creates_request(self, hitl_service, sample_task, sample_context, sample_failed_children):
        """Test that enabled HITL service creates and tracks requests."""
        # Mock the _wait_for_response method to return immediately
        async def mock_wait_for_response(request_id, timeout):
            return HITLResponse(
                request_id=request_id,
                status=HITLRequestStatus.APPROVED,
                response_data={}
            )

        with patch.object(hitl_service, '_wait_for_response', side_effect=mock_wait_for_response):
            result = await hitl_service.request_replanning_approval(
                node=sample_task,
                context=sample_context,
                failed_children=sample_failed_children,
                failure_reason="Test failure"
            )

        # Verify request was created and response received
        assert result is not None
        assert isinstance(result, HITLResponse)
        assert result.status == HITLRequestStatus.APPROVED

    @pytest.mark.asyncio
    async def test_request_task_guidance_creates_request(self, hitl_service, sample_task, sample_context):
        """Test task guidance request creation."""
        # Mock the _wait_for_response method
        async def mock_wait_for_response(request_id, timeout):
            return HITLResponse(
                request_id=request_id,
                status=HITLRequestStatus.APPROVED,
                response_data={"guidance": "Use alternative approach"}
            )

        with patch.object(hitl_service, '_wait_for_response', side_effect=mock_wait_for_response):
            result = await hitl_service.request_task_guidance(
                node=sample_task,
                context=sample_context,
                guidance_request="How should we proceed with this complex task?"
            )

        # Verify response
        assert result is not None
        assert result.status == HITLRequestStatus.APPROVED
        assert result.response_data["guidance"] == "Use alternative approach"

    def test_create_hitl_request(self, hitl_service, sample_task):
        """Test HITL request creation."""
        request = hitl_service._create_hitl_request(
            request_type=HITLRequestType.REPLANNING_APPROVAL,
            task_id=sample_task.task_id,
            request_data={"test": "data"}
        )

        assert isinstance(request, HITLRequest)
        assert request.request_type == HITLRequestType.REPLANNING_APPROVAL
        assert request.task_id == sample_task.task_id
        assert request.request_data["test"] == "data"
        assert request.status == HITLRequestStatus.PENDING

    @pytest.mark.asyncio
    async def test_wait_for_response_timeout(self, hitl_service):
        """Test timeout handling in wait_for_response."""
        request_id = str(uuid4())

        # Should timeout after short duration
        result = await hitl_service._wait_for_response(request_id, timeout_seconds=0.1)

        assert result is not None
        assert result.status == HITLRequestStatus.TIMEOUT
        assert result.request_id == request_id

    @pytest.mark.asyncio
    async def test_provide_response_updates_pending_request(self, hitl_service, sample_task):
        """Test providing response to pending request."""
        # Create a request
        request = hitl_service._create_hitl_request(
            request_type=HITLRequestType.REPLANNING_APPROVAL,
            task_id=sample_task.task_id,
            request_data={}
        )

        # Add to pending requests
        hitl_service._pending_requests[request.request_id] = request

        # Provide response
        response = HITLResponse(
            request_id=request.request_id,
            status=HITLRequestStatus.APPROVED,
            response_data={"approved": True}
        )

        success = await hitl_service.provide_response(response)

        assert success is True
        # Request should be removed from pending after response
        assert request.request_id not in hitl_service._pending_requests

    @pytest.mark.asyncio
    async def test_provide_response_nonexistent_request(self, hitl_service):
        """Test providing response to non-existent request."""
        fake_response = HITLResponse(
            request_id=str(uuid4()),
            status=HITLRequestStatus.APPROVED,
            response_data={}
        )

        success = await hitl_service.provide_response(fake_response)

        assert success is False

    def test_get_pending_requests(self, hitl_service, sample_task):
        """Test getting list of pending requests."""
        # Initially should be empty
        pending = hitl_service.get_pending_requests()
        assert len(pending) == 0

        # Add some requests
        request1 = hitl_service._create_hitl_request(
            request_type=HITLRequestType.REPLANNING_APPROVAL,
            task_id=sample_task.task_id,
            request_data={}
        )
        request2 = hitl_service._create_hitl_request(
            request_type=HITLRequestType.TASK_GUIDANCE,
            task_id=sample_task.task_id,
            request_data={}
        )

        hitl_service._pending_requests[request1.request_id] = request1
        hitl_service._pending_requests[request2.request_id] = request2

        # Should return both requests
        pending = hitl_service.get_pending_requests()
        assert len(pending) == 2
        assert request1 in pending
        assert request2 in pending

    def test_get_request_by_id(self, hitl_service, sample_task):
        """Test retrieving specific request by ID."""
        request = hitl_service._create_hitl_request(
            request_type=HITLRequestType.REPLANNING_APPROVAL,
            task_id=sample_task.task_id,
            request_data={}
        )

        hitl_service._pending_requests[request.request_id] = request

        # Should retrieve the correct request
        retrieved = hitl_service.get_request_by_id(request.request_id)
        assert retrieved == request

        # Should return None for non-existent ID
        fake_id = str(uuid4())
        assert hitl_service.get_request_by_id(fake_id) is None

    @pytest.mark.asyncio
    async def test_cancel_request(self, hitl_service, sample_task):
        """Test cancelling a pending request."""
        request = hitl_service._create_hitl_request(
            request_type=HITLRequestType.REPLANNING_APPROVAL,
            task_id=sample_task.task_id,
            request_data={}
        )

        hitl_service._pending_requests[request.request_id] = request

        # Cancel the request
        success = await hitl_service.cancel_request(request.request_id)

        assert success is True
        assert request.request_id not in hitl_service._pending_requests

        # Cancelling non-existent request should return False
        fake_id = str(uuid4())
        success = await hitl_service.cancel_request(fake_id)
        assert success is False

    def test_get_service_stats(self, hitl_service, sample_task):
        """Test getting service statistics."""
        # Add some requests to test stats
        request1 = hitl_service._create_hitl_request(
            request_type=HITLRequestType.REPLANNING_APPROVAL,
            task_id=sample_task.task_id,
            request_data={}
        )
        request2 = hitl_service._create_hitl_request(
            request_type=HITLRequestType.TASK_GUIDANCE,
            task_id=sample_task.task_id,
            request_data={}
        )

        hitl_service._pending_requests[request1.request_id] = request1
        hitl_service._pending_requests[request2.request_id] = request2

        stats = hitl_service.get_service_stats()

        assert stats["enabled"] is True
        assert stats["pending_requests"] == 2
        assert stats["default_timeout_seconds"] == 300
        assert "request_types" in stats
        assert HITLRequestType.REPLANNING_APPROVAL.value in stats["request_types"]
        assert HITLRequestType.TASK_GUIDANCE.value in stats["request_types"]

    def test_clear_pending_requests(self, hitl_service, sample_task):
        """Test clearing all pending requests."""
        # Add some requests
        request1 = hitl_service._create_hitl_request(
            request_type=HITLRequestType.REPLANNING_APPROVAL,
            task_id=sample_task.task_id,
            request_data={}
        )
        request2 = hitl_service._create_hitl_request(
            request_type=HITLRequestType.TASK_GUIDANCE,
            task_id=sample_task.task_id,
            request_data={}
        )

        hitl_service._pending_requests[request1.request_id] = request1
        hitl_service._pending_requests[request2.request_id] = request2

        assert len(hitl_service._pending_requests) == 2

        # Clear all requests
        hitl_service.clear_pending_requests()

        assert len(hitl_service._pending_requests) == 0

    @pytest.mark.asyncio
    async def test_request_replanning_approval_with_custom_timeout(self, hitl_service, sample_task, sample_context, sample_failed_children):
        """Test replanning approval with custom timeout."""
        # Mock the _wait_for_response method to verify timeout parameter
        async def mock_wait_for_response(request_id, timeout):
            assert timeout == 60  # Custom timeout
            return HITLResponse(
                request_id=request_id,
                status=HITLRequestStatus.TIMEOUT,
                response_data={}
            )

        with patch.object(hitl_service, '_wait_for_response', side_effect=mock_wait_for_response):
            result = await hitl_service.request_replanning_approval(
                node=sample_task,
                context=sample_context,
                failed_children=sample_failed_children,
                failure_reason="Test failure",
                timeout_seconds=60
            )

        assert result.status == HITLRequestStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, hitl_service, sample_task, sample_context):
        """Test handling multiple concurrent HITL requests."""
        tasks = []

        # Mock _wait_for_response to simulate different response times
        async def mock_wait_for_response(request_id, timeout):
            # Simulate different response times
            await asyncio.sleep(0.1)
            return HITLResponse(
                request_id=request_id,
                status=HITLRequestStatus.APPROVED,
                response_data={"request_id": request_id}
            )

        with patch.object(hitl_service, '_wait_for_response', side_effect=mock_wait_for_response):
            # Create multiple concurrent requests
            for i in range(3):
                task = hitl_service.request_task_guidance(
                    node=sample_task,
                    context=sample_context,
                    guidance_request=f"Guidance request {i}"
                )
                tasks.append(task)

            # Wait for all requests to complete
            results = await asyncio.gather(*tasks)

        # All requests should complete successfully
        assert len(results) == 3
        for result in results:
            assert result.status == HITLRequestStatus.APPROVED
            assert "request_id" in result.response_data