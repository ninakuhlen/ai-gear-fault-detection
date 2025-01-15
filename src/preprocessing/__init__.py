# mein_paket/__init__.py
from .add_centrifugal_force import add_centrifugal_force
from .add_time import add_time
from .add_timedelta import add_timedelta
from .apply_threshold import apply_threshold
from .calculate_dft import calculate_dft
from .clean_outliers import clean_outliers
from .discard_data import discard_data
from .mean import mean
from .median import median
from .random_resample import random_resample
from .scale_robust import scale_robust
from .split_by_gradient_direction import split_by_gradient_direction
from .step_resample import step_resample

import importlib

# # Funktion, die die Funktionalität tatsächlich kapselt und verfügbar macht
# def funktion1():
#     # Laden Sie die tatsächliche Funktion dynamisch
#     actual_function = getattr(importlib.import_module('.modul1', package=__name__), 'funktion1')
#     return actual_function()

# def funktion2():
#     actual_function = getattr(importlib.import_module('.modul2', package=__name__), 'funktion2')
#     return actual_function()

# # Optional: Definieren Sie __all__, um anzugeben, was importiert werden kann
# __all__ = ['funktion1', 'funktion2']
