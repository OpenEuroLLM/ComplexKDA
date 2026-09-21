from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.complex_kda.configuration_complex_kda import ComplexKDAConfig
from fla.models.complex_kda.modeling_complex_kda import ComplexKDAForCausalLM, ComplexKDAModel

AutoConfig.register(ComplexKDAConfig.model_type, ComplexKDAConfig, exist_ok=True)
AutoModel.register(ComplexKDAConfig, ComplexKDAModel, exist_ok=True)
AutoModelForCausalLM.register(ComplexKDAConfig, ComplexKDAForCausalLM, exist_ok=True)

__all__ = ['ComplexKDAConfig', 'ComplexKDAForCausalLM', 'ComplexKDAModel']
