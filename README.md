# BoopNet

A U-Net-inspired convolutional encoder-decoder to extract floor plans from commercial real estate
brochures in the form of PDF documents.


### Version

Both the package and the model are at v0.1.

### Requirements

- `imagemagick`
- `pip`

### Usage

```
from boopnet import extract_from_pdf

extract_from_pdf("/home/boop/stuff/brochure.pdf")
```

This will yield one PNG file for each page where a floor plan ( or more ) are found. Floor plan 
images are tightly cropped.

You may pass `destination="/home/boop/floor_plans"` to this function should you wish for the PNGs
to be exported into a different directory.

```
from boopnet import extract_all_from_directory

extract_all_from_directory("/home/stuff/pdfs")
```

This will work through all PDFs in the specified directory. Again, you may specify a 
`destination` directory if you want.





### Installation

#### ImageMagick

```
sudo apt-get install imagemagick
```

or 

```
brew install imagemagick
```


#### Python dependencies

```
pip install -r requirements.txt
```




### To Do

- For pages containing multiple, distinct floor plans, export these as individual images
- Allow user to specify resolution, DPI, and quality of image to pull from the PDFs
- Train the model on more and better data to improve result quality
- Create a `setup.py` file

### Authors

Brittney Johnson and Quentin Caudron
