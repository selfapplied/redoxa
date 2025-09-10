"""
Path utilities for Redoxa project
"""
import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the absolute path to the project root directory.
    
    Looks for pyproject.toml to identify the project root.
    """
    current = Path(__file__).resolve()
    
    # Walk up the directory tree looking for pyproject.toml
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    
    # Fallback: assume we're in src/orchestrator/redoxa/ and go up 3 levels
    return current.parent.parent.parent


def get_out_dir() -> Path:
    """Get the absolute path to the .out directory."""
    return get_project_root() / ".out"


def get_db_path(db_name: str) -> str:
    """Get the absolute path to a database file in .out directory.
    
    Args:
        db_name: Name of the database file (e.g., "vm.db", "ce1_seed.db")
        
    Returns:
        Absolute path to the database file
    """
    out_dir = get_out_dir()
    out_dir.mkdir(exist_ok=True)
    return str(out_dir / db_name)
