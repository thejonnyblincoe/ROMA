"""
Filename Generator for ROMA v2 Toolkits.

Clean filename generation following Single Responsibility Principle.
Provides standardized filename generation for data storage.
"""

import time


class FileNameGenerator:
    """
    Stateless utility class for generating standardized filenames.

    Single Responsibility: Generate consistent filenames
    Open/Closed: Extensible via static methods
    """

    @staticmethod
    def generate_data_filename(
        prefix: str,
        extension: str = "json",
        include_timestamp: bool = True,
        timestamp: int | None = None,
    ) -> str:
        """
        Generate standardized data filename.

        Args:
            prefix: Filename prefix
            extension: File extension (without dot)
            include_timestamp: Whether to include timestamp
            timestamp: Custom timestamp (uses current if None)

        Returns:
            Generated filename string
        """
        if include_timestamp:
            ts = timestamp or int(time.time())
            return f"{prefix}_{ts}.{extension}"
        else:
            return f"{prefix}.{extension}"

    @staticmethod
    def generate_symbol_filename(
        symbol: str, data_type: str, extension: str = "json", include_timestamp: bool = True
    ) -> str:
        """
        Generate filename for symbol-specific data.

        Args:
            symbol: Trading symbol
            data_type: Type of data (price, orderbook, etc.)
            extension: File extension (without dot)
            include_timestamp: Whether to include timestamp

        Returns:
            Generated filename string
        """
        prefix = f"{data_type}_{symbol.upper()}"
        return FileNameGenerator.generate_data_filename(prefix, extension, include_timestamp)

    @staticmethod
    def generate_market_filename(
        market_type: str, data_type: str, extension: str = "json", include_timestamp: bool = True
    ) -> str:
        """
        Generate filename for market-specific data.

        Args:
            market_type: Market type (spot, futures, etc.)
            data_type: Type of data
            extension: File extension (without dot)
            include_timestamp: Whether to include timestamp

        Returns:
            Generated filename string
        """
        prefix = f"{market_type}_{data_type}"
        return FileNameGenerator.generate_data_filename(prefix, extension, include_timestamp)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to be filesystem-safe.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename string
        """
        # Replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        safe_filename = filename

        for char in unsafe_chars:
            safe_filename = safe_filename.replace(char, "_")

        # Remove multiple underscores
        while "__" in safe_filename:
            safe_filename = safe_filename.replace("__", "_")

        # Remove leading/trailing underscores and spaces
        safe_filename = safe_filename.strip("_").strip()

        return safe_filename
