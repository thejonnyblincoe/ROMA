"""
KnowledgeStore Service - ROMA v2.0 Application Layer.

Thread-safe service for managing KnowledgeRecord value objects.
"""

import asyncio
from collections import OrderedDict
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timezone

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.knowledge_record import KnowledgeRecord
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.result_envelope import ResultEnvelope
from roma.application.services.artifact_service import ArtifactService

logger = logging.getLogger(__name__)


class KnowledgeStoreService:
    """
    Thread-safe service for managing task knowledge records.

    Stores immutable KnowledgeRecord value objects with LRU caching.
    """

    def __init__(self, artifact_service: Optional[ArtifactService] = None):
        """
        Initialize knowledge store service.

        Args:
            artifact_service: Optional service for storing artifacts
        """
        self._records: Dict[str, KnowledgeRecord] = {}
        self._lock = asyncio.Lock()
        self._artifact_service = artifact_service

        # LRU cache for frequently accessed records
        self._cache: OrderedDict[str, KnowledgeRecord] = OrderedDict()
        self._cache_max_size = 100

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("KnowledgeStoreService initialized")

    async def add_or_update_record(
        self,
        node: TaskNode,
        envelope: Optional[ResultEnvelope] = None
    ) -> KnowledgeRecord:
        """
        Add or update a knowledge record from TaskNode.

        Args:
            node: TaskNode to store
            envelope: Optional ResultEnvelope with artifacts

        Returns:
            Created or updated KnowledgeRecord
        """
        async with self._lock:
            # Store artifacts if envelope provided
            artifact_refs = []
            if envelope and envelope.artifacts and self._artifact_service:
                try:
                    artifact_refs = await self._artifact_service.store_envelope_artifacts(
                        envelope
                    )
                    logger.debug(f"Stored {len(artifact_refs)} artifacts for task {node.task_id}")
                except Exception as e:
                    logger.error(f"Failed to store artifacts for task {node.task_id}: {e}")

            # Create or update record
            existing_record = self._records.get(node.task_id)

            if existing_record:
                # Update existing record
                record = existing_record.update_status(node.status)
                if artifact_refs:
                    record = record.add_artifacts(artifact_refs)
                # Update result if provided
                if envelope:
                    record = record.model_copy(update={
                        "result": envelope,
                        "updated_at": datetime.now(timezone.utc),
                        "version": record.version + 1
                    })
            else:
                # Create new record
                record = KnowledgeRecord.create(
                    task_id=node.task_id,
                    goal=node.goal,
                    task_type=node.task_type,
                    status=node.status,
                    artifacts=artifact_refs,
                    parent_task_id=node.parent_id,
                    result=envelope
                )

            self._records[record.task_id] = record
            self._update_cache(record.task_id, record)

            logger.info(f"KnowledgeStore: Added/Updated record for {node.task_id}")
            return record

    async def get_record(self, task_id: str) -> Optional[KnowledgeRecord]:
        """
        Get record by ID with caching.

        Args:
            task_id: Task ID to retrieve

        Returns:
            KnowledgeRecord if found, None otherwise
        """
        async with self._lock:
            # Check cache first (now thread-safe)
            if task_id in self._cache:
                self._cache.move_to_end(task_id)  # Mark as recently used
                self._cache_hits += 1
                return self._cache[task_id]

            # Load from storage
            record = self._records.get(task_id)
            if record:
                self._update_cache(task_id, record)
            self._cache_misses += 1
            return record

    async def get_child_records(self, parent_id: str) -> List[KnowledgeRecord]:
        """Get all child records of a parent task."""
        async with self._lock:
            return [r for r in self._records.values() if r.parent_task_id == parent_id]

    async def get_records_by_status(self, status: TaskStatus) -> List[KnowledgeRecord]:
        """Get all records with specific status."""
        async with self._lock:
            return [r for r in self._records.values() if r.status == status]

    async def add_child_relationship(self, parent_id: str, child_id: str) -> bool:
        """
        Add child relationship to parent record.

        Args:
            parent_id: Parent task ID
            child_id: Child task ID

        Returns:
            True if relationship added, False if parent not found
        """
        async with self._lock:
            parent_record = self._records.get(parent_id)
            if not parent_record:
                return False

            # Update parent record with new child
            updated_parent = parent_record.add_child(child_id)
            self._records[parent_id] = updated_parent
            self._update_cache(parent_id, updated_parent)

            logger.debug(f"Added child {child_id} to parent {parent_id}")
            return True

    async def get_completed_records(self) -> List[KnowledgeRecord]:
        """Get all completed records."""
        return await self.get_records_by_status(TaskStatus.COMPLETED)

    async def get_failed_records(self) -> List[KnowledgeRecord]:
        """Get all failed records."""
        return await self.get_records_by_status(TaskStatus.FAILED)

    def _update_cache(self, key: str, record: KnowledgeRecord) -> None:
        """
        Update LRU cache with record.

        ⚠️  CRITICAL: This method MUST only be called while holding self._lock.
        It is NOT thread-safe on its own and assumes the caller has exclusive access.
        """
        if key in self._cache:
            # Update existing cache entry with new record and mark as recently used
            self._cache[key] = record
            self._cache.move_to_end(key)
        else:
            # Remove oldest if at capacity
            if len(self._cache) >= self._cache_max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                logger.debug(f"Evicted {oldest_key} from cache")

            self._cache[key] = record

    async def clear(self) -> None:
        """Clear all records and cache."""
        async with self._lock:
            self._records.clear()
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.info("KnowledgeStore: All records cleared")

    async def get_summary_stats(self) -> Dict[str, Any]:
        """Get knowledge store statistics."""
        async with self._lock:
            statuses = [r.status for r in self._records.values()]
            task_types = [r.task_type for r in self._records.values()]

            total_cache_requests = self._cache_hits + self._cache_misses
            cache_hit_rate = (self._cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0

            return {
                "total_records": len(self._records),
                "cache_size": len(self._cache),
                "cache_hit_rate": round(cache_hit_rate, 2),
                "status_breakdown": {
                    status.value: statuses.count(status)
                    for status in set(statuses)
                },
                "task_type_breakdown": {
                    task_type.value: task_types.count(task_type)
                    for task_type in set(task_types)
                },
                "records_with_artifacts": len([r for r in self._records.values() if r.has_artifacts()]),
                "records_with_children": len([r for r in self._records.values() if r.has_children()]),
                "records_with_results": len([r for r in self._records.values() if r.has_result()]),
            }

    async def get_records_with_artifacts(self, task_id: str, include_siblings: bool = True) -> List[str]:
        """
        Get artifact references for task context building.

        Args:
            task_id: Task ID to get artifacts for
            include_siblings: Whether to include sibling artifacts

        Returns:
            List of artifact storage keys
        """
        async with self._lock:
            artifacts = []

            # Get task's artifacts
            record = self._records.get(task_id)
            if record:
                artifacts.extend(record.artifacts)

                # Get sibling artifacts if requested
                if include_siblings and record.parent_task_id:
                    siblings = [r for r in self._records.values() if r.parent_task_id == record.parent_task_id]
                    for sibling in siblings:
                        if sibling.task_id != task_id and sibling.has_artifacts():
                            artifacts.extend(sibling.artifacts)

            return artifacts