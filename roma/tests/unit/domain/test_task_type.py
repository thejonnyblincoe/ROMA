"""
Unit tests for TaskType enum.

Tests the MECE task classification with RETRIEVE, WRITE, THINK.
"""

import pytest

from src.roma.domain.value_objects.task_type import TaskType


class TestTaskTypeEnum:
    """Test TaskType enum values and behavior."""
    
    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert TaskType.RETRIEVE == "RETRIEVE"
        assert TaskType.WRITE == "WRITE" 
        assert TaskType.THINK == "THINK"
    
    def test_string_conversion(self):
        """Test string conversion."""
        assert str(TaskType.RETRIEVE) == "RETRIEVE"
        assert str(TaskType.WRITE) == "WRITE"
        assert str(TaskType.THINK) == "THINK"
    
    def test_all_task_types_exist(self):
        """Test that all five MECE task types are present."""
        task_types = list(TaskType)
        assert len(task_types) == 5
        assert TaskType.RETRIEVE in task_types
        assert TaskType.WRITE in task_types
        assert TaskType.THINK in task_types
        assert TaskType.CODE_INTERPRET in task_types
        assert TaskType.IMAGE_GENERATION in task_types


class TestTaskTypeFromString:
    """Test TaskType.from_string() method."""
    
    def test_valid_string_conversion(self):
        """Test valid string to TaskType conversion."""
        assert TaskType.from_string("RETRIEVE") == TaskType.RETRIEVE
        assert TaskType.from_string("WRITE") == TaskType.WRITE
        assert TaskType.from_string("THINK") == TaskType.THINK
    
    def test_case_insensitive_conversion(self):
        """Test case-insensitive string conversion."""
        assert TaskType.from_string("retrieve") == TaskType.RETRIEVE
        assert TaskType.from_string("write") == TaskType.WRITE
        assert TaskType.from_string("think") == TaskType.THINK
        assert TaskType.from_string("Retrieve") == TaskType.RETRIEVE
        assert TaskType.from_string("WRITE") == TaskType.WRITE
    
    def test_invalid_string_raises_error(self):
        """Test that invalid string raises ValueError.""" 
        with pytest.raises(ValueError, match="Invalid task type 'INVALID'"):
            TaskType.from_string("INVALID")
        
        with pytest.raises(ValueError, match="Invalid task type 'search'"):
            TaskType.from_string("search")  # No backward compatibility in v2
        
        with pytest.raises(ValueError):
            TaskType.from_string("")
    
    def test_error_message_includes_valid_types(self):
        """Test that error message includes valid types."""
        try:
            TaskType.from_string("INVALID")
        except ValueError as e:
            error_msg = str(e)
            assert "RETRIEVE" in error_msg
            assert "WRITE" in error_msg 
            assert "THINK" in error_msg


class TestTaskTypeProperties:
    """Test TaskType convenience properties."""
    
    def test_is_retrieve_property(self):
        """Test is_retrieve property."""
        assert TaskType.RETRIEVE.is_retrieve is True
        assert TaskType.WRITE.is_retrieve is False
        assert TaskType.THINK.is_retrieve is False
        assert TaskType.CODE_INTERPRET.is_retrieve is False
        assert TaskType.IMAGE_GENERATION.is_retrieve is False
    
    def test_is_write_property(self):
        """Test is_write property."""
        assert TaskType.RETRIEVE.is_write is False
        assert TaskType.WRITE.is_write is True
        assert TaskType.THINK.is_write is False
        assert TaskType.CODE_INTERPRET.is_write is False
        assert TaskType.IMAGE_GENERATION.is_write is False
    
    def test_is_think_property(self):
        """Test is_think property."""
        assert TaskType.RETRIEVE.is_think is False
        assert TaskType.WRITE.is_think is False
        assert TaskType.THINK.is_think is True
        assert TaskType.CODE_INTERPRET.is_think is False
        assert TaskType.IMAGE_GENERATION.is_think is False
    
    def test_is_code_interpret_property(self):
        """Test is_code_interpret property."""
        assert TaskType.RETRIEVE.is_code_interpret is False
        assert TaskType.WRITE.is_code_interpret is False
        assert TaskType.THINK.is_code_interpret is False
        assert TaskType.CODE_INTERPRET.is_code_interpret is True
        assert TaskType.IMAGE_GENERATION.is_code_interpret is False
    
    def test_is_image_generation_property(self):
        """Test is_image_generation property."""
        assert TaskType.RETRIEVE.is_image_generation is False
        assert TaskType.WRITE.is_image_generation is False
        assert TaskType.THINK.is_image_generation is False
        assert TaskType.CODE_INTERPRET.is_image_generation is False
        assert TaskType.IMAGE_GENERATION.is_image_generation is True


class TestTaskTypeMECEFramework:
    """Test that TaskType implements MECE (Mutually Exclusive, Collectively Exhaustive) framework."""
    
    def test_mutually_exclusive(self):
        """Test that task types are mutually exclusive."""
        retrieve = TaskType.RETRIEVE
        write = TaskType.WRITE
        think = TaskType.THINK
        code_interpret = TaskType.CODE_INTERPRET
        image_generation = TaskType.IMAGE_GENERATION
        
        # Each type should only match its own property
        assert retrieve.is_retrieve and not retrieve.is_write and not retrieve.is_think and not retrieve.is_code_interpret and not retrieve.is_image_generation
        assert write.is_write and not write.is_retrieve and not write.is_think and not write.is_code_interpret and not write.is_image_generation
        assert think.is_think and not think.is_retrieve and not think.is_write and not think.is_code_interpret and not think.is_image_generation
        assert code_interpret.is_code_interpret and not code_interpret.is_retrieve and not code_interpret.is_write and not code_interpret.is_think and not code_interpret.is_image_generation
        assert image_generation.is_image_generation and not image_generation.is_retrieve and not image_generation.is_write and not image_generation.is_think and not image_generation.is_code_interpret
    
    def test_collectively_exhaustive(self):
        """Test that task types are collectively exhaustive."""
        all_types = {TaskType.RETRIEVE, TaskType.WRITE, TaskType.THINK, TaskType.CODE_INTERPRET, TaskType.IMAGE_GENERATION}
        enum_types = set(TaskType)
        
        # All enum values should be in our expected set
        assert enum_types == all_types
        
        # Every task type should satisfy exactly one property
        for task_type in TaskType:
            property_count = sum([
                task_type.is_retrieve,
                task_type.is_write, 
                task_type.is_think,
                task_type.is_code_interpret,
                task_type.is_image_generation
            ])
            assert property_count == 1, f"TaskType {task_type} should satisfy exactly one property"


class TestTaskTypeComparisons:
    """Test TaskType equality and comparisons."""
    
    def test_enum_equality(self):
        """Test TaskType equality."""
        assert TaskType.RETRIEVE == TaskType.RETRIEVE
        assert TaskType.WRITE != TaskType.RETRIEVE
        assert TaskType.THINK != TaskType.WRITE
    
    def test_string_equality(self):
        """Test TaskType to string equality."""
        assert TaskType.RETRIEVE == "RETRIEVE"
        assert TaskType.WRITE == "WRITE"
        assert TaskType.THINK == "THINK"
    
    def test_enum_identity(self):
        """Test TaskType identity."""
        retrieve1 = TaskType.RETRIEVE
        retrieve2 = TaskType.RETRIEVE
        assert retrieve1 is retrieve2  # Same instance


class TestTaskTypeUsagePatterns:
    """Test common usage patterns for TaskType."""
    
    def test_in_collections(self):
        """Test TaskType usage in sets and lists."""
        task_types = [TaskType.RETRIEVE, TaskType.WRITE, TaskType.THINK]
        task_set = {TaskType.RETRIEVE, TaskType.WRITE, TaskType.THINK}
        
        assert TaskType.RETRIEVE in task_types
        assert TaskType.WRITE in task_set
        assert len(set(task_types)) == 3  # All unique
    
    def test_switch_pattern(self):
        """Test switch-like pattern with TaskType."""
        def process_task_type(task_type: TaskType) -> str:
            if task_type == TaskType.RETRIEVE:
                return "data_acquisition"
            elif task_type == TaskType.WRITE:
                return "content_generation" 
            elif task_type == TaskType.THINK:
                return "analysis_reasoning"
            else:
                return "unknown"
        
        assert process_task_type(TaskType.RETRIEVE) == "data_acquisition"
        assert process_task_type(TaskType.WRITE) == "content_generation"
        assert process_task_type(TaskType.THINK) == "analysis_reasoning"
    
    def test_filtering_pattern(self):
        """Test filtering tasks by type."""
        task_data = [
            {"id": "1", "type": TaskType.RETRIEVE, "goal": "Get data"},
            {"id": "2", "type": TaskType.THINK, "goal": "Analyze"}, 
            {"id": "3", "type": TaskType.WRITE, "goal": "Create report"},
            {"id": "4", "type": TaskType.RETRIEVE, "goal": "More data"}
        ]
        
        retrieve_tasks = [t for t in task_data if t["type"].is_retrieve]
        think_tasks = [t for t in task_data if t["type"].is_think] 
        write_tasks = [t for t in task_data if t["type"].is_write]
        
        assert len(retrieve_tasks) == 2
        assert len(think_tasks) == 1
        assert len(write_tasks) == 1


class TestTaskTypeSerializationDeserialization:
    """Test TaskType serialization and deserialization patterns."""
    
    def test_json_serialization(self):
        """Test TaskType can be serialized to JSON-compatible format."""
        import json
        
        task_data = {
            "task_id": "123",
            "task_type": TaskType.RETRIEVE.value,
            "goal": "Test goal"
        }
        
        json_str = json.dumps(task_data)
        parsed_data = json.loads(json_str)
        
        assert parsed_data["task_type"] == "RETRIEVE"
        
        # Reconstruct TaskType
        reconstructed_type = TaskType.from_string(parsed_data["task_type"])
        assert reconstructed_type == TaskType.RETRIEVE
    
    def test_yaml_like_serialization(self):
        """Test TaskType works with YAML-like string representations."""
        configs = {
            "retrieve_agent": {
                "task_type": "RETRIEVE",
                "model": "gpt-4"
            },
            "think_agent": {
                "task_type": "THINK", 
                "model": "claude-3"
            }
        }
        
        for agent_name, config in configs.items():
            task_type = TaskType.from_string(config["task_type"])
            assert task_type in [TaskType.RETRIEVE, TaskType.THINK]