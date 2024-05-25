from pdf2image import convert_from_path
import glob
from PIL import Image

chunk_size = 32
chunk_cnt_per_side = 100

if __name__ == "__main__":
    for file in glob.glob("*.pdf"):
        png = convert_from_path(file, dpi=300, poppler_path="C:\\Users\\alexm\\Downloads\\poppler-23.11.0\\Library\\bin")
        for img in png:
            base_width = chunk_size * chunk_cnt_per_side
            img = img.resize((base_width, base_width), Image.LANCZOS)
            img.save(file.split('.')[0]+".png", 'PNG')
            print(file + " done")
