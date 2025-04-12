from . import chatgpt
from . import gpt2
from . import gpt3
from . import dummy
from . import together
from . import maritalk
from . import openai_compatible_models
from . import gpt
MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "gpt": gpt.GPTLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "dummy": dummy.DummyLM,
    "chatgpt": openai_compatible_models.OpenaiAPI,
    "together": openai_compatible_models.TogetherAPI,
    "maritalk": openai_compatible_models.MaritalkAPI,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
