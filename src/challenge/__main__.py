"""
Command-line interface entry point for skeleton-challenge.

This module enables running the application via `python -m challenge`.
It provides a simple CLI for starting the API server and other operations.
"""

import argparse
import logging
import sys

import uvicorn

from challenge import __version__
from challenge.api.main import app


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:  # noqa: S104
    """
    Start the FastAPI application server.

    Args:
        host: Server host address
        port: Server port number
        reload: Enable auto-reload on code changes (development mode)

    """
    logging.info("Starting skeleton-challenge API v%s", __version__)
    logging.info("Server: http://%s:%d", host, port)
    logging.info("API Documentation: http://%s:%d/api/docs", host, port)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main() -> int:
    """
    Run the main CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)

    """
    parser = argparse.ArgumentParser(
        description=f"Skeleton Challenge v{__version__} - Clean Architecture Python Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"skeleton-challenge {__version__}",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # API server command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104
        help="Server host address (default: 0.0.0.0)",
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number (default: 8000)",
    )
    api_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development mode)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Handle commands
    if args.command == "api":
        run_api(host=args.host, port=args.port, reload=args.reload)
        return 0

    # If no command provided, show help and default to API server
    if not args.command:
        print(f"skeleton-challenge v{__version__}")
        print("\nNo command specified. Starting API server...")
        print("Use --help to see available commands\n")
        run_api()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
