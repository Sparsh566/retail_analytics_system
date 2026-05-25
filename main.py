# main.py
# Root entry point launcher for the Object Detection and Tracking System.
# Imports and executes the MainController package from the src core.

from src.app import MainController

if __name__ == "__main__":
    app = MainController()
    app.run()
