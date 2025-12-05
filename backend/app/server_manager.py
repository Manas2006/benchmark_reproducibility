"""
Server Manager for CoT Analysis Servers

This module manages CoT analysis server processes, including cleanup of completed servers.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ServerManager:
    """
    Manages CoT analysis server processes.
    
    Tracks running servers and provides cleanup functionality for completed servers.
    """
    
    def __init__(self):
        """Initialize the server manager"""
        self._servers: Dict[str, Dict[str, Any]] = {}
    
    def register_server(self, server_id: str, server_info: Dict[str, Any]):
        """
        Register a new server.
        
        Args:
            server_id: Unique identifier for the server
            server_info: Dictionary containing server metadata
        """
        self._servers[server_id] = server_info
        if logger:
            logger.debug(f"Registered server: {server_id}")
    
    def unregister_server(self, server_id: str):
        """
        Unregister a server.
        
        Args:
            server_id: Unique identifier for the server
        """
        if server_id in self._servers:
            del self._servers[server_id]
            if logger:
                logger.debug(f"Unregistered server: {server_id}")
    
    def cleanup_completed_servers(self) -> int:
        """
        Clean up completed CoT analysis servers.
        
        This method checks for servers that have completed and removes them
        from the tracking dictionary.
        
        Returns:
            Number of servers that were cleaned up
        """
        cleaned_count = 0
        servers_to_remove = []
        
        for server_id, server_info in self._servers.items():
            # Check if server is marked as completed
            if server_info.get("status") == "completed":
                servers_to_remove.append(server_id)
        
        # Remove completed servers
        for server_id in servers_to_remove:
            self.unregister_server(server_id)
            cleaned_count += 1
        
        if cleaned_count > 0 and logger:
            logger.info(f"Cleaned up {cleaned_count} completed server(s)")
        
        return cleaned_count
    
    def get_server_count(self) -> int:
        """
        Get the number of tracked servers.
        
        Returns:
            Number of currently tracked servers
        """
        return len(self._servers)
    
    def get_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all tracked servers.
        
        Returns:
            Dictionary of server_id -> server_info
        """
        return self._servers.copy()


# Singleton instance
_server_manager_instance: Optional[ServerManager] = None


def get_server_manager() -> ServerManager:
    """
    Get the singleton ServerManager instance.
    
    Returns:
        The global ServerManager instance
    """
    global _server_manager_instance
    if _server_manager_instance is None:
        _server_manager_instance = ServerManager()
    return _server_manager_instance

