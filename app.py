from flask import Flask, request, jsonify, render_template
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
from PIL import Image
import torch

app = Flask(__name__)

# Load the pre-trained image captioning model and tokenizer
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the text generation model (GPT-2)
text_generator = pipeline("text-generation", model="gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)

def generate_basic_caption(image_path):
    # Open the image file
    img = Image.open(image_path)

    # Preprocess the image
    pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate the basic caption
    output_ids = caption_model.generate(pixel_values, max_length=50, num_beams=5, temperature=1.0)
    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return caption

def enhance_caption_with_gpt2(basic_caption):
    prompt = f"Generate a cool, engaging, and social media-friendly caption with emojis for the following image description: {basic_caption}"
    
    response = text_generator(prompt, max_length=60, num_return_sequences=1, temperature=0.7)
    
    enhanced_caption = response[0]['generated_text'].strip()
    
    return enhanced_caption

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_caption', methods=['POST'])
def generate_caption_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'}), 400
    
    img = request.files['image']
    img_path = "./temp_image.png"
    img.save(img_path)
    
    basic_caption = generate_basic_caption(img_path)
    cool_caption = enhance_caption_with_gpt2(basic_caption)
    
    return jsonify({'caption': cool_caption}), 200

if __name__ == '__main__':
    app.run(debug=True)
