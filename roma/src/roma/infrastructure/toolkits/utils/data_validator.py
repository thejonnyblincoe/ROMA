"""
Data Validator for ROMA v2 Toolkits.

Clean validation utilities following Single Responsibility Principle.
Provides data validation for common toolkit operations.
"""

from typing import Any, Dict, List, Optional, Union
import re


class DataValidator:
    """
    Stateless utility class for data validation.
    
    Single Responsibility: Validate data structures and formats
    Open/Closed: Extensible via static methods
    """
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """
        Validate trading symbol format.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            True if valid symbol format
        """
        if not symbol or not isinstance(symbol, str):
            return False
            
        # Basic symbol validation (letters and numbers, 2-20 chars)
        return bool(re.match(r'^[A-Z0-9]{2,20}$', symbol.upper()))
        
    @staticmethod
    def validate_price_data(data: Dict[str, Any]) -> bool:
        """
        Validate price data structure.
        
        Args:
            data: Price data dictionary
            
        Returns:
            True if valid price data structure
        """
        required_fields = {"symbol", "price"}
        
        if not isinstance(data, dict):
            return False
            
        # Check required fields exist
        if not all(field in data for field in required_fields):
            return False
            
        # Validate price is numeric
        try:
            float(data["price"])
            return True
        except (ValueError, TypeError):
            return False
            
    @staticmethod
    def validate_order_book(data: Dict[str, Any]) -> bool:
        """
        Validate order book data structure.
        
        Args:
            data: Order book data dictionary
            
        Returns:
            True if valid order book structure
        """
        if not isinstance(data, dict):
            return False
            
        required_fields = {"bids", "asks"}
        if not all(field in data for field in required_fields):
            return False
            
        # Validate bids and asks are lists
        if not isinstance(data["bids"], list) or not isinstance(data["asks"], list):
            return False
            
        return True
        
    @staticmethod
    def validate_time_range(start_time: Optional[int], end_time: Optional[int]) -> bool:
        """
        Validate time range parameters.
        
        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            
        Returns:
            True if valid time range
        """
        if start_time is not None and end_time is not None:
            # Both provided - validate order
            return start_time <= end_time
            
        # Single values or None are valid
        return True
        
    @staticmethod
    def validate_pagination(limit: Optional[int], offset: Optional[int] = None) -> bool:
        """
        Validate pagination parameters.
        
        Args:
            limit: Record limit
            offset: Record offset (optional)
            
        Returns:
            True if valid pagination parameters
        """
        if limit is not None:
            if not isinstance(limit, int) or limit <= 0 or limit > 10000:
                return False
                
        if offset is not None:
            if not isinstance(offset, int) or offset < 0:
                return False
                
        return True