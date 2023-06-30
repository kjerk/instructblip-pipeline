from torchvision import transforms
from torchvision.transforms import Normalize, InterpolationMode

class BlipImageEvalProcessor:
	
	def __init__(self, image_size=384, mean=None, std=None):
		if mean is None:
			mean = (0.48145466, 0.4578275, 0.40821073)
		if std is None:
			std = (0.26862954, 0.26130258, 0.27577711)
		
		self.normalize = Normalize(mean, std)
		
		self.transform = transforms.Compose(
			[
				transforms.Resize(
					(image_size, image_size), interpolation=InterpolationMode.BICUBIC
				),
				transforms.ToTensor(),
				self.normalize,
			]
		)
	
	def __call__(self, item):
		return self.transform(item)

if __name__ == '__main__':
	print('__main__ not allowed in modules')
