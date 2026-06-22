import os
import pandas as pd
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
from datasets import load_dataset

# Setup folder directories
pristine_dir = "./dataset/pristine_images"
rural_dir = "./dataset/rural_images"
os.makedirs(pristine_dir, exist_ok=True)
os.makedirs(rural_dir, exist_ok=True)

print(" Step 1: Downloading PathVQA subset from Hugging Face...")
# Streams dataset directly without needing a massive local clone
dataset = load_dataset("flaviagiammarino/path-vqa", split="train", streaming=True)

eval_data = []
count = 0
target_samples = 50  # Balanced count for fast hackathon evaluation

print(" Step 2: Running degradation pipeline...")
for item in dataset:
    question = item['question']
    answer = item['answer'].strip().lower()
    
    # Filter strictly for binary answers to ensure deterministic grading
    if answer in ['yes', 'no']:
        try:
            # Convert the streamed image object to RGB PIL Image
            img = item['image'].convert('RGB')
            
            filename = f"sample_{count}.jpg"
            pristine_path = os.path.join(pristine_dir, filename)
            rural_path = os.path.join(rural_dir, filename)
            
            # Save Pristine Control Image
            img.save(pristine_path, format="JPEG", quality=100)
            
            # Apply stacked transformations mimicking rural telepathology
            # A: Microscope eyepiece glare (Overexposure)
            deg_img = ImageEnhance.Brightness(img).enhance(1.3)
            # B: Low-end smartphone camera sensor (Loss of contrast)
            deg_img = ImageEnhance.Contrast(deg_img).enhance(0.8)
            # C: Unfocused/Dirty lens (Gaussian Blur)
            deg_img = deg_img.filter(ImageFilter.GaussianBlur(radius=1.2))
            
            # D: WhatsApp / 3G Network compression artifacts
            buffer = BytesIO()
            deg_img.save(buffer, format="JPEG", quality=15)
            
            # Save Degraded Target Image
            with open(rural_path, "wb") as f:
                f.write(buffer.getvalue())
                
            # Append unified metadata
            eval_data.append({
                "image_filename": filename,
                "prompt": f"Examine the image carefully. Answer the following question with exactly one word, either 'Yes' or 'No'. Question: {question}",
                "ground_truth": answer.capitalize()
            })
            
            count += 1
            print(f" -> Prepared image {count}/{target_samples}")
            
            if count >= target_samples:
                break
        except Exception as e:
            continue

# Save the structured spreadsheet mapping file
df = pd.DataFrame(eval_data)
df.to_csv("./dataset/metadata.csv", index=False)
print("\n Step 3: Success! Check your local './dataset' directory.")
print("You now have two image folders and one 'metadata.csv' file.")