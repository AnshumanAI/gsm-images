import pandas as pd

GITHUB_USERNAME = "AnshumanAI" 

print("Loading original metadata...")
df = pd.read_csv("./dataset/metadata.csv")

# Generate Public URLs for Pristine Images
print("Generating GitHub URLs for Pristine...")
df_pristine = df.copy()
df_pristine['image_url'] = df_pristine['image_filename'].apply(
    lambda x: f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/gsm-images/main/pristine_images/{x}"
)
df_pristine.to_csv("./dataset/pristine_final.csv", index=False)

# Generate Public URLs for Rural Images
print("Generating GitHub URLs for Rural...")
df_rural = df.copy()
df_rural['image_url'] = df_rural['image_filename'].apply(
    lambda x: f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/gsm-images/main/rural_images/{x}"
)
df_rural.to_csv("./dataset/rural_final.csv", index=False)

print(" Done! You now have pristine_final.csv and rural_final.csv")