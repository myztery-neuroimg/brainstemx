"""
BrainStemX – modular brain-stem lesion-detection toolkit.

Core APIs (always available):
    from brainstemx.pipeline import process_subject
    from brainstemx.postprocess import analyse

Optional APIs (require extras):
    # pip install brainstemx[reports]
    from brainstemx.report_generator import generate as generate_report

    # pip install brainstemx[web]
    from brainstemx.web_visualiser import main as run_web_ui
"""
__version__ = "0.2.0"

# Core exports - always available
from .pipeline import process_subject
from .postprocess import analyse

# Optional exports - wrapped to provide helpful errors
def generate_report(*args, **kwargs):
    """Generate AI-powered radiology report. Requires: pip install brainstemx[reports]"""
    from .report_generator import generate
    return generate(*args, **kwargs)

def run_web_ui(*args, **kwargs):
    """Launch web visualization UI. Requires: pip install brainstemx[web]"""
    from .web_visualiser import main
    return main(*args, **kwargs)
