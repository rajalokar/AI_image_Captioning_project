import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, output_scores=True, return_dict_in_generate=True)
    caption = processor.decode(out.sequences[0], skip_special_tokens=True)

    # Extract average confidence
    scores = out.scores
    confidence = torch.stack([torch.nn.functional.softmax(score, dim=-1).max() for score in scores])
    avg_confidence = float(confidence.mean().item())

    return caption, round(avg_confidence * 100, 2)
