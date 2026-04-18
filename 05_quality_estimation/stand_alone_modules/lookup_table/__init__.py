"""Lookup-table package exports."""

__all__ = ["main"]


def main(argv=None):
    """Lazily load the CLI entry point to avoid runpy module warnings."""
    from stand_alone_modules.lookup_table.cli import main as cli_main

    return cli_main(argv)
