from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np

def calculate_density(char, font, img_size=(16, 16)):
    img = Image.new("L", img_size, color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((img_size[0] - w) / 2, (img_size[1] - h) / 2), char, font=font, fill=0)
    return 1 - np.mean(np.array(img) / 255)

def preprocess_image(image, contrast_factor=2.0):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(contrast_factor)

def map_feature_to_char(feature, max_feature, char_set):
    normalized_feature = (feature ** 0.5) / (max_feature ** 0.5)
    index = int(normalized_feature * (len(char_set) - 1))
    return char_set[min(index, len(char_set) - 1)]

def main():
    image_path = "test_d.jpg"
    image = Image.open(image_path).convert("RGB")
    region_width, region_height = 4, 4
    image = preprocess_image(image, contrast_factor=1.8)
    width, height = image.size
    width -= width % region_width
    height -= height % region_height
    image = image.resize((width, height))
    num_regions_x = width // region_width
    num_regions_y = height // region_height
    #char_set = " .,;:'-_=+!`~<>{}[]()@#$%&*abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    char_set = "　一二三四五六七八九十上下工懿侕觞霞龔"
    font_path = "/home/byayoi/workspace/dieloli/data/font/SarasaMonoSC-Regular.ttf"
    font = ImageFont.truetype(font_path, size=16)
    sorted_char_set = sorted(char_set, key=lambda c: calculate_density(c, font),reverse=1)
    region_features = []
    for y in range(num_regions_y):
        row_features = []
        for x in range(num_regions_x):
            region = image.crop((x * region_width, y * region_height,
                                 (x + 1) * region_width, (y + 1) * region_height))
            feature = np.mean(np.array(region.convert("L")))
            row_features.append(feature)
        region_features.append(row_features)
    max_feature = max(max(row) for row in region_features) * 0.8
    ascii_art = ""
    for row in region_features:
        ascii_art += "".join(map_feature_to_char(f, max_feature, sorted_char_set) for f in row) + "\n"
    print(ascii_art)

if __name__ == "__main__":
    main()

