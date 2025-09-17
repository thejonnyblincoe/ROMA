"""
Tests for Toolkit Validation System.

Tests the optional validation mixin and BinanceToolkit validation implementation.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import Tuple, Optional

from src.roma.infrastructure.toolkits.base_agno_toolkit import (
    BaseAgnoToolkit,
    ToolkitValidationMixin
)
from src.roma.infrastructure.toolkits.custom.crypto.binance_toolkit import BinanceToolkit


class TestToolkitValidationMixin:
    """Test the optional validation mixin functionality."""
    
    class MockValidationToolkit(BaseAgnoToolkit, ToolkitValidationMixin):
        """Mock toolkit implementing validation mixin."""
        
        def __init__(self, config):
            super().__init__(config)
            self._health_status = True
            self._health_warning = None
            self._health_error = None
        
        async def validate_health(self) -> Tuple[bool, Optional[str], Optional[str]]:
            """Mock health validation."""
            return self._health_status, self._health_warning, self._health_error
        
        def set_health_status(self, healthy: bool, warning: str = None, error: str = None):
            """Set mock health status for testing."""
            self._health_status = healthy
            self._health_warning = warning
            self._health_error = error
    
    @pytest.fixture
    def validation_toolkit(self):
        """Create mock validation toolkit."""
        config = {"name": "test_validation_toolkit", "type": "test"}
        return self.MockValidationToolkit(config)
    
    @pytest.mark.asyncio
    async def test_validation_configuration_success(self, validation_toolkit):
        """Test successful configuration validation."""
        is_valid, warning, error = await validation_toolkit.validate_configuration()
        
        assert is_valid is True
        assert warning is None
        assert error is None
    
    @pytest.mark.asyncio
    async def test_validation_configuration_empty_name(self):
        """Test configuration validation with empty name."""
        config = {"name": "", "type": "test"}
        
        # Should raise ValueError during initialization
        with pytest.raises(ValueError, match="Toolkit name cannot be empty"):
            self.MockValidationToolkit(config)
    
    @pytest.mark.asyncio
    async def test_validation_tools_success(self, validation_toolkit):
        """Test successful tools validation."""
        is_valid, warning, error = await validation_toolkit.validate_tools()
        
        assert is_valid is True
        # Mock toolkit has validate_health method, so tools validation passes without warning
    
    @pytest.mark.asyncio
    async def test_validation_health_success(self, validation_toolkit):
        """Test successful health validation."""
        validation_toolkit.set_health_status(True)
        
        is_healthy, warning, error = await validation_toolkit.validate_health()
        
        assert is_healthy is True
        assert warning is None
        assert error is None
    
    @pytest.mark.asyncio
    async def test_validation_health_with_warning(self, validation_toolkit):
        """Test health validation with warning."""
        validation_toolkit.set_health_status(True, warning="Performance degraded")
        
        is_healthy, warning, error = await validation_toolkit.validate_health()
        
        assert is_healthy is True
        assert warning == "Performance degraded"
        assert error is None
    
    @pytest.mark.asyncio
    async def test_validation_health_failure(self, validation_toolkit):
        """Test health validation failure."""
        validation_toolkit.set_health_status(False, error="Service unavailable")
        
        is_healthy, warning, error = await validation_toolkit.validate_health()
        
        assert is_healthy is False
        assert error == "Service unavailable"
    
    @pytest.mark.asyncio
    async def test_run_full_validation_all_pass(self, validation_toolkit):
        """Test full validation suite with all checks passing."""
        validation_toolkit.set_health_status(True)
        
        results = await validation_toolkit.run_full_validation()
        
        assert results["toolkit_name"] == "test_validation_toolkit"
        assert results["overall_healthy"] is True
        assert "validation_timestamp" in results
        
        # Check individual results
        assert results["checks"]["health"]["passed"] is True
        assert results["checks"]["configuration"]["passed"] is True
        assert results["checks"]["tools"]["passed"] is True
    
    @pytest.mark.asyncio
    async def test_run_full_validation_with_failure(self, validation_toolkit):
        """Test full validation suite with health check failing."""
        validation_toolkit.set_health_status(False, error="Critical failure")
        
        results = await validation_toolkit.run_full_validation()
        
        assert results["overall_healthy"] is False
        assert results["checks"]["health"]["passed"] is False
        assert results["checks"]["health"]["error"] == "Critical failure"


class TestBinanceToolkitValidation:
    """Test BinanceToolkit validation implementation."""
    
    @pytest.fixture
    def binance_toolkit(self):
        """Create BinanceToolkit for testing."""
        config = {
            "name": "binance_test",
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "default_market_type": "spot"
        }
        return BinanceToolkit(config)
    
    @pytest.mark.asyncio
    async def test_binance_validation_not_created(self, binance_toolkit):
        """Test validation when toolkit is not created."""
        # Don't call create() - toolkit should report as not created
        is_healthy, warning, error = await binance_toolkit.validate_health()
        
        assert is_healthy is False
        assert error == "Toolkit not created/initialized"
    
    @pytest.mark.asyncio
    async def test_binance_validation_success(self, binance_toolkit):
        """Test successful BinanceToolkit validation."""
        await binance_toolkit.create()
        
        is_healthy, warning, error = await binance_toolkit.validate_health()
        
        # Should be healthy since it's a mock implementation
        assert is_healthy is True
        assert warning is None
        assert error is None
    
    @pytest.mark.asyncio
    async def test_binance_full_validation(self, binance_toolkit):
        """Test full validation suite on BinanceToolkit."""
        await binance_toolkit.create()
        
        results = await binance_toolkit.run_full_validation()
        
        assert results["toolkit_name"] == "binance_test"
        assert results["overall_healthy"] is True
        assert "validation_timestamp" in results
        
        # All checks should pass
        for check_name, check_result in results["checks"].items():
            assert check_result["passed"] is True, f"Check {check_name} failed"
    
    @pytest.mark.asyncio
    async def test_binance_has_tools_available(self, binance_toolkit):
        """Test that BinanceToolkit has tools available."""
        await binance_toolkit.create()
        
        tools = binance_toolkit.get_available_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Check for expected tools
        expected_tools = ["get_current_price", "get_order_book", "get_klines", "get_symbol_ticker_change"]
        for tool in expected_tools:
            assert tool in tools, f"Expected tool {tool} not found in available tools"
    
    @pytest.mark.asyncio
    async def test_binance_tools_are_callable(self, binance_toolkit):
        """Test that BinanceToolkit tools are actually callable."""
        await binance_toolkit.create()
        
        tools = binance_toolkit.get_available_tools()
        
        for tool_name in tools:
            tool_method = getattr(binance_toolkit, tool_name, None)
            assert tool_method is not None, f"Tool method {tool_name} not found"
            assert callable(tool_method), f"Tool method {tool_name} is not callable"
    
    @pytest.mark.asyncio
    async def test_binance_sample_tool_execution(self, binance_toolkit):
        """Test that BinanceToolkit can execute a sample tool."""
        await binance_toolkit.create()
        
        # Test get_current_price tool
        result = await binance_toolkit.get_current_price("BTCUSDT")
        
        assert result is not None
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "symbol" in result
        assert result["symbol"] == "BTCUSDT"


class TestToolkitValidationIntegration:
    """Test validation integration with toolkit manager."""
    
    @pytest.mark.asyncio 
    async def test_toolkit_supports_validation(self):
        """Test that toolkits can be checked for validation support."""
        # BinanceToolkit supports validation
        config = {"name": "binance_test", "symbols": ["BTCUSDT"]}
        binance_toolkit = BinanceToolkit(config)
        
        # Should be instance of validation mixin
        assert isinstance(binance_toolkit, ToolkitValidationMixin)
        
        # Should have validation methods
        assert hasattr(binance_toolkit, 'validate_health')
        assert hasattr(binance_toolkit, 'validate_configuration')
        assert hasattr(binance_toolkit, 'validate_tools')
        assert hasattr(binance_toolkit, 'run_full_validation')
    
    @pytest.mark.asyncio
    async def test_non_validating_toolkit(self):
        """Test toolkit that doesn't implement validation."""
        from src.roma.infrastructure.toolkits.base_agno_toolkit import BaseAgnoToolkit
        
        # Regular toolkit without validation mixin
        config = {"name": "simple_toolkit", "type": "test"}
        simple_toolkit = BaseAgnoToolkit(config)
        
        # Should NOT be instance of validation mixin
        assert not isinstance(simple_toolkit, ToolkitValidationMixin)
        
        # Should NOT have validation methods
        assert not hasattr(simple_toolkit, 'validate_health')
    
    def test_validation_check_interface(self):
        """Test that validation methods follow the expected interface."""
        # All validation methods should return Tuple[bool, Optional[str], Optional[str]]
        from inspect import signature
        
        config = {"name": "test", "symbols": ["BTCUSDT"]}
        toolkit = BinanceToolkit(config)
        
        # Check method signatures
        health_sig = signature(toolkit.validate_health)
        config_sig = signature(toolkit.validate_configuration) 
        tools_sig = signature(toolkit.validate_tools)
        
        # All should be async methods with no parameters (except self)
        assert len(health_sig.parameters) == 0  # self is implicit
        assert len(config_sig.parameters) == 0
        assert len(tools_sig.parameters) == 0