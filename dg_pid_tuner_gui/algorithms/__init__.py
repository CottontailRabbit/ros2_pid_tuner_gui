from .base import Tuner, TunerResult, TunerConfig
from .relay_feedback import RelayFeedbackTuner
from .ziegler_nichols import ZieglerNicholsTuner
from .cohen_coon import CohenCoonTuner
from .imc import ImcTuner
from .chr import ChrTuner
from .bayesian_opt import BayesianOptTuner

ALGORITHMS = {
    'Ziegler-Nichols (open-loop, FOPDT)': ZieglerNicholsTuner,
    'Cohen-Coon (FOPDT)':                  CohenCoonTuner,
    'Chien-Hrones-Reswick (CHR, 0% OS)':   ChrTuner,
    'Internal Model Control (IMC)':        ImcTuner,
    'Relay Feedback (Astrom-Hagglund)':    RelayFeedbackTuner,
    'Bayesian Optimization (GP+EI)':       BayesianOptTuner,
}
