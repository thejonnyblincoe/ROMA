"""Response Builder Utilities (exact v1 implementation)
===========================

Standardized response construction for consistent API responses across toolkits.
Provides uniform response formats for success, error, and data storage scenarios.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["ResponseBuilder"]


class ResponseBuilder:
    """Stateful utility class for building standardized API responses (exact v1 pattern).

    Automatically injects toolkit information into all responses when initialized
    with toolkit context, eliminating the need for manual injection.
    """

    def __init__(self, toolkit_info: dict[str, Any] | None = None) -> None:
        """Initialize ResponseBuilder with toolkit information (exact v1 pattern).

        Args:
            toolkit_info: Dictionary containing toolkit identification info
                         (toolkit_name, toolkit_category, toolkit_type, toolkit_icon)
        """
        self.toolkit_info = toolkit_info or {}

    def success_response(
        self,
        data: Any = None,
        message: str = "Operation completed successfully",
        **additional_fields: Any,
    ) -> dict[str, Any]:
        """Create a standardized success response with automatic toolkit information injection (exact v1).

        Args:
            data: Response data payload
            message: Success message
            **additional_fields: Additional fields to include in response

        Returns:
            dict: Standardized success response with toolkit info automatically injected
        """
        response = {"success": True, "message": message, "fetched_at": int(time.time())}

        if data is not None:
            response["data"] = data

        # Add toolkit information
        response.update(self.toolkit_info)

        # Add any additional fields
        response.update(additional_fields)

        return response

    def error_response(
        self, error: str, error_type: str = "general_error", **additional_fields: Any
    ) -> dict[str, Any]:
        """Create standardized error response (exact v1 pattern)."""
        response = {
            "success": False,
            "error": error,
            "error_type": error_type,
            "fetched_at": int(time.time()),
        }

        # Add toolkit information
        response.update(self.toolkit_info)

        # Add any additional fields
        response.update(additional_fields)

        return response

    def file_response(
        self, file_path: str, message: str = "Data stored to file", **additional_fields: Any
    ) -> dict[str, Any]:
        """Create standardized file storage response (exact v1 pattern)."""
        response = {
            "success": True,
            "message": message,
            "file_path": str(file_path),
            "fetched_at": int(time.time()),
        }

        # Add file info if available
        try:
            path_obj = Path(file_path)
            if path_obj.exists():
                response["file_size"] = path_obj.stat().st_size
                response["file_name"] = path_obj.name
        except Exception:
            pass

        # Add toolkit information
        response.update(self.toolkit_info)

        # Add any additional fields
        response.update(additional_fields)

        return response

    def api_error_response(
        self,
        api_endpoint: str,
        api_message: str,
        status_code: int | None = None,
        **additional_fields: Any,
    ) -> dict[str, Any]:
        """Create standardized API error response (exact v1 pattern)."""
        response = {
            "success": False,
            "error": api_message,
            "error_type": "api_error",
            "api_endpoint": api_endpoint,
            "fetched_at": int(time.time()),
        }

        if status_code is not None:
            response["status_code"] = status_code

        # Add toolkit information
        response.update(self.toolkit_info)

        # Add any additional fields
        response.update(additional_fields)

        return response

    def build_data_response_with_storage(
        self,
        data: Any,
        storage_check_func: Callable[[Any], bool],
        storage_func: Callable[[Any, str], str],
        filename_prefix: str,
        success_message: str,
        **additional_fields: Any,
    ) -> dict[str, Any]:
        """Build response with automatic storage decision (exact v1 pattern)."""
        if storage_check_func(data):
            file_path = storage_func(data, filename_prefix)
            return self.file_response(file_path, success_message, **additional_fields)
        else:
            return self.success_response(data, success_message, **additional_fields)
