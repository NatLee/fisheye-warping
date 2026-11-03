"""Run the command as ``python -m fisheyewarping``."""
from .cli import main

if __name__ == '__main__':  # worker processes on Windows and macOS import this module again
    main()
