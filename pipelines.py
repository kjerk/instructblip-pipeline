from typing import Optional

from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline

available_pipelines = ['instructblip-7b', 'instructblip-13b']

def get_pipeline(name: str, params: dict) -> Optional[AbstractMultimodalPipeline]:
	if name == 'instructblip-7b':
		from .instructblip_pipeline import InstructVicuna_7b_Pipeline
		return InstructVicuna_7b_Pipeline(params)
	if name == 'instructblip-13b':
		from .instructblip_pipeline import InstructVicuna_13b_Pipeline
		return InstructVicuna_13b_Pipeline(params)
	return None

def get_pipeline_from_model_name(model_name: str, params: dict) -> Optional[AbstractMultimodalPipeline]:
	if 'vicuna' not in model_name.lower():
		return None
	if '7b' in model_name.lower():
		from .instructblip_pipeline import InstructVicuna_7b_Pipeline
		return InstructVicuna_7b_Pipeline(params)
	if '13b' in model_name.lower():
		from .instructblip_pipeline import InstructVicuna_13b_Pipeline
		return InstructVicuna_13b_Pipeline(params)
	return None

if __name__ == '__main__':
	print('__main__ not allowed in modules')
