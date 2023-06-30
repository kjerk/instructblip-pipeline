from typing import List

import torch
from PIL import Image
from torch.hub import load_state_dict_from_url

from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline
from modules import shared
from modules.text_generation import encode
from .instructblip.blip2_vicuna_embedder import Blip2VicunaInstructEmbedder
from .instructblip.blip_image_processor import BlipImageEvalProcessor

_available_checkpoints = [
	"https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth",
	"https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna13b_trimmed.pth",
]

class InstructBlipPipeline(AbstractMultimodalPipeline):
	def __init__(self, params: dict) -> None:
		super().__init__()
		self.image_preprocessor = BlipImageEvalProcessor(image_size=224)
		self.image_embedder = None  # type: Blip2VicunaInstructEmbedder
	
	@staticmethod
	def placeholder_token_id() -> int:
		return 1
	
	@staticmethod
	def image_start() -> str:
		return "<Img>"
	
	@staticmethod
	def image_end() -> str:
		return "</Img>"
	
	@staticmethod
	def num_image_embeds() -> int:
		return 32
	
	@staticmethod
	def embed_tokens(input_ids: torch.Tensor) -> torch.Tensor:
		return shared.model.model.embed_tokens(input_ids).to(shared.model.device, dtype=shared.model.dtype)
	
	@staticmethod
	def placeholder_embeddings() -> torch.Tensor:
		placeholders = encode("<ImgContent>", add_bos_token=False, add_special_tokens=False)[0]
		return InstructBlipPipeline.embed_tokens(placeholders.to(shared.model.device, dtype=torch.int64)).to(dtype=shared.model.dtype)
	
	def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
		image_tensors = torch.stack([self.image_preprocessor(image) for image in images])
		image_emb = self.image_embedder.embed_textless(image_tensors)
		return image_emb.to(shared.model.device, dtype=shared.model.dtype)

class InstructVicuna_7b_Pipeline(InstructBlipPipeline):
	def __init__(self, params: dict) -> None:
		super().__init__(params)
		
		weights = load_state_dict_from_url(_available_checkpoints[0], map_location="cpu", check_hash=False, progress=True)
		
		if "model" in weights:
			weights = weights["model"]
		
		self.image_embedder = Blip2VicunaInstructEmbedder(
			llm_hidden_size=4096,
		)
		
		self.image_embedder.load_state_dict(weights, strict=False)
		del weights
	
	@staticmethod
	def name() -> str:
		return "instruct-gpt-7b"

class InstructVicuna_13b_Pipeline(InstructBlipPipeline):
	def __init__(self, params: dict) -> None:
		super().__init__(params)
		
		weights = load_state_dict_from_url(_available_checkpoints[1], map_location="cpu", check_hash=False, progress=True)
		
		if "model" in weights:
			weights = weights["model"]
		
		self.image_embedder = Blip2VicunaInstructEmbedder(
			llm_hidden_size=5120,
		)
		
		self.image_embedder.load_state_dict(weights, strict=False)
		del weights
	
	@staticmethod
	def name() -> str:
		return "instruct-gpt-13b"

if __name__ == "__main__":
	print("__main__ not allowed in modules")
