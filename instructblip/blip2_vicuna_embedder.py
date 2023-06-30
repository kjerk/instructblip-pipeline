import torch
from transformers import BertTokenizer

from .Qformer import BertConfig, BertLMHeadModel
from .blip_utils import LayerNorm

# Heavily adapted from https://github.com/salesforce/LAVIS/blob/76c556e51876146ffd78018fe13233f8f4cf624a/lavis/models/blip2_models/blip2_vicuna_instruct.py
# and https://github.com/salesforce/LAVIS/blob/76c556e51876146ffd78018fe13233f8f4cf624a/lavis/models/blip2_models/blip2.py
# Flattened and simplified.

class EmbeddingMode:
	WITH_TEXT = 0
	NO_TEXT = 1

class Blip2VicunaInstructEmbedder(torch.nn.Module):
	
	def __init__(
		self,
		llm_hidden_size=5120,
		img_size=224,
		max_txt_len=128,
		num_query_token=32,
		embedding_mode=EmbeddingMode.NO_TEXT,
	):
		super().__init__()
		
		self.embedding_mode = embedding_mode
		
		self.max_txt_len = max_txt_len
		
		self.visual_encoder, self.ln_vision = self.init_vision_encoder(img_size, 0, False, 'fp32')
		self.visual_encoder.eval()
		
		self.Qformer, self.query_tokens = self.init_qformer(num_query_token, self.visual_encoder.num_features)
		
		if self.embedding_mode == EmbeddingMode.NO_TEXT:
			self.Qformer.bert.embeddings.word_embeddings = None
			self.Qformer.bert.embeddings.position_embeddings = None
			for layer in self.Qformer.bert.encoder.layer:
				layer.output = None
				layer.intermediate = None
		elif self.embedding_mode == EmbeddingMode.WITH_TEXT:
			self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left")
			self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
			self.Qformer.resize_token_embeddings(len(self.tokenizer))
		
		self.Qformer.cls = None
		
		self.llm_proj = torch.nn.Linear(self.Qformer.config.hidden_size, llm_hidden_size)
		
		self.visual_encoder = self.visual_encoder.to('cpu')
		self.ln_vision = self.ln_vision.to('cpu')
		self.Qformer = self.Qformer.to('cuda')
		self.llm_proj = self.llm_proj.to('cpu')
	
		# Torch modules with parameters:
		# llm_proj, ln_vision, query_tokens, Qformer.*
	
	def init_qformer(self, num_query_token, vision_width, cross_attention_freq=2):
		encoder_config = BertConfig.from_pretrained("bert-base-uncased")
		
		encoder_config.encoder_width = vision_width
		encoder_config.add_cross_attention = True
		encoder_config.cross_attention_freq = cross_attention_freq
		encoder_config.query_length = num_query_token
		
		Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
		
		query_tokens = torch.nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
		query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
		
		for name, param in Qformer.named_parameters():
			param.requires_grad = False
			del param.grad
		
		Qformer.eval()
		
		return Qformer, query_tokens
	
	def init_vision_encoder(self, img_size, drop_path_rate, use_grad_checkpoint, precision):
		from .eva_vit import create_eva_vit_g
		visual_encoder = create_eva_vit_g(img_size, drop_path_rate, use_grad_checkpoint, precision)
		ln_vision = LayerNorm(visual_encoder.num_features)
		
		# Freeze vision encoder
		for name, param in visual_encoder.named_parameters():
			param.requires_grad = False
			del param.grad
		for name, param in ln_vision.named_parameters():
			param.requires_grad = False
			del param.grad
		
		visual_encoder.eval()
		
		return visual_encoder, ln_vision
	
	def embed_images(self, image_tensors, current_prompt: str = None):
		if self.embedding_mode == EmbeddingMode.WITH_TEXT and current_prompt is not None:
			return self.embed_with_prompt(image_tensors, current_prompt)
		elif self.embedding_mode == EmbeddingMode.NO_TEXT:
			return self.embed_textless(image_tensors)
		else:
			raise ValueError("Invalid embedding mode")
	
	def embed_textless(self, image_tensor):
		batch_size = image_tensor.size(0)
		
		with torch.no_grad():
			image_tensors = image_tensor.to('cpu', dtype=torch.float32)
			
			image_embeds = self.ln_vision(self.visual_encoder(image_tensors)).to('cuda')
			image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to('cuda')
			query_tokens = self.query_tokens.expand(batch_size, -1, -1).to('cuda')
			
			query_output = self.Qformer.bert(
				query_embeds=query_tokens,
				encoder_hidden_states=image_embeds,
				encoder_attention_mask=image_atts,
				return_dict=True,
			)
			
			last_state = query_output.last_hidden_state[:, :query_tokens.size(1), :]
			last_state = last_state.to('cpu', dtype=torch.float32)
			
			inputs_llm = self.llm_proj(last_state)
		
		return inputs_llm
	
	def embed_with_prompt(self, image_tensor, current_prompt: str =""):
		# Currently unused, but this is the default mode for instructblip in LAVIS.
		# Leaving this for future compatibility w/ textgen, but unneccesary for now.
		bs = image_tensor.size(0)
		
		prompt = current_prompt * bs
		
		query_tokens = self.query_tokens.expand(bs, -1, -1)
		
		text_qformer = self.tokenizer(
			prompt,
			padding='longest',
			truncation=True,
			max_length=self.max_txt_len,
			return_tensors="pt",
		).to(image_tensor.device)
		
		query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_tensor.device)
		qformer_atts = torch.cat([query_atts, text_qformer.attention_mask], dim=1)
		
		with torch.no_grad():
			image_tensor = image_tensor.to('cpu', dtype=torch.float32)
			image_embeds = self.ln_vision(self.visual_encoder(image_tensor))
			
			image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to('cuda')
			
			query_output = self.Qformer.bert(
				text_qformer.input_ids.to('cuda'),
				attention_mask=qformer_atts.to('cuda'),
				query_embeds=query_tokens.to('cuda'),
				encoder_hidden_states=image_embeds.to('cuda'),
				encoder_attention_mask=image_atts,
				return_dict=True,
			)
			
			last_state = query_output.last_hidden_state[:, :query_tokens.size(1), :]
			last_state = last_state.to('cpu', dtype=torch.float32)
			inputs_llm = self.llm_proj(last_state)
		
		return inputs_llm

if __name__ == '__main__':
	print('__main__ not allowed in modules')