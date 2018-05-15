import os
from glob import glob
from subprocess import call
from uuid import uuid1

from skimage import io

from boopnet.utilities import preprocess, postprocess, segmentation_model
model = segmentation_model()


def extract_from_pdf(pdf, destination=None):
    """
    Given the absolute path of a PDF file, extract all floor plans from it.

    Plans are rendered as PNG images, and land in the directory specified by the
    destination kwarg. If none is provided, they appear in the same directory
    as the original PDF file.
    """

    # If no destination is explicitly passed, use the directory of the PDF
    if destination is None:
        destination = os.path.split(pdf)[0]

    # If one is specified, trim the final slash
    else:
        if destination[-1] == "/":
            destination = destination[:-1]

    # Generate a temporary filename
    filename = "/tmp/{}".format(uuid1())

    # Extract pages from PDF in PNG format
    print("\nConverting {} to single-page PNGs.".format(pdf))
    imagemagick_commands = [
        "convert",
        "-resize", "2000x",  # high res
        "-density", "300x300",  # still high res
        "-quality", "100",  # no compression
        pdf,  # source
        "{}__page_%03d.png".format(filename)  # destination
    ]
    call(imagemagick_commands)

    # Iterate over each image, exporting floor plans
    for image_file in glob("{}__page_*.png".format(filename)):

        print("Processing image {}".format(image_file))

        # Read in the image
        image = io.imread(image_file)

        # Process it to neural network specs
        processed, rescale_factor, rotated = preprocess(image)

        # Pass it through the model
        mask = model.predict(processed.reshape(1, *processed.shape))

        # Postprocess the mask
        bounding_box = postprocess(image, mask, rescale_factor, rotated)

        # If a floor plan is found, write it to disk as a PNG
        if bounding_box is not None:
            filepath = os.path.join(
                destination,
                os.path.split(pdf)[1] + "__" + image_file.split("__page_")[1]
            )
            print("  Floor plan found ! Writing {}".format(filepath))
            io.imsave(filepath, bounding_box)
        
        # Otherwise, keep calm and carry on
        else:
            print("  No floor plan found; continuing.")

        # Remove temporary PNG files of extracted PDF images
        for temp_file in glob("{}__page_*.png".format(filename)):
            call(["rm", temp_file])

def extract_all_from_directory(directory, destination=None):
    """
    Given an absolute path, extract all floor plans from all PDF brochures in that path.

    See the docstring for extract_from_pdf for more information.
    """

    # If no directory is set, target the same directory as contains the PDF files
    if destination is None:
        destination = directory

    # If one is specified, trim the final slash
    else:
        if destination[-1] == "/":
            destination = destination[:-1]

    # For each PDF in that directory, call extract_from_pdf
    for pdf in glob("{}/*.pdf".format(directory)) + glob("{}/*.PDF".format(directory)):
        extract_from_pdf(pdf, destination=destination)
