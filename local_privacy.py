"""Disable optional model-library telemetry before native runtimes initialize."""
import os


def configure_local_models():
    os.environ['ORT_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    # This suppresses Ultralytics connectivity probes and analytics, while its
    # explicit official-weight downloader remains available for first use.
    os.environ['YOLO_OFFLINE'] = 'true'
