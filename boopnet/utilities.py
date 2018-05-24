import numpy as np

from skimage import transform, filters, morphology, color

from keras.models import load_model
from keras import backend as K


def preprocess(image):
    """
    Given an RGB(A) image, ensure its resolution matches that of the 
    neural network model, and squash the alpha channel, if any.
    """
    
    # Ensure image is portrait in orientation
    if image.shape[1] > image.shape[0]:
        rotated = True
        image = transform.rotate(image, 90, resize=True)
    else:
        rotated = False
    
    # Rescale image to 400x304 pixels
    original_size = image.shape
    rescale_factor = (408/image.shape[0], 312/image.shape[1])
    image = transform.rescale(image, rescale_factor)

    # We don't like transparency
    if image.shape[2] == 4:
        image = color.rgba2rgb(image)
    
    return (image, rescale_factor, rotated)


def postprocess(original_image, mask, rescale_factor, rotated, sensitivity=0.7):
    """
    Given an original image, and a ( probably ) smaller mask returned by the
    convolutional encoder-decoder, along with some metadata such as the image's
    rescale factor, and whether it was rotated, return a tight crop of the original
    image containing the floor plan only.
    """
    
    # Remove singleton dimension to yield a (400, 304) image
    # from the predicted (400, 304, 1) image
    mask = mask.squeeze()
    
    # First, assess whether the image contains significant information or just noise
    if mask.max() < sensitivity:  # If not, set it to zero and return that
        return None
        
    else:  # If it contains info, clean it up
        
        # Begin by thresholding using Otsu's method
        mask = mask > filters.threshold_otsu(mask)
        
        # Remove small connected components
        mask = morphology.remove_small_objects(mask, 500)

        # If we've removed everything, it was noise
        if mask.sum() < 1:
            return None

        # Compute the convex hull
        hull = morphology.convex_hull_image(mask)

        # Grab the minimal bounding box around the hull
        x = hull.max(0)
        y = hull.max(1)
        x_min = np.argmax(x[1:] * (1-x[:-1]))
        x_max = np.argmax((1-x[1:]) * x[:-1]) + 2
        y_min = np.argmax(y[1:] * (1-y[:-1]))
        y_max = np.argmax((1-y[1:]) * y[:-1]) + 2

        # Rescale bounding box coordinates
        x_min = int(np.floor(x_min / rescale_factor[1]))
        x_max = int(np.ceil(x_max / rescale_factor[1]))
        y_min = int(np.floor(y_min / rescale_factor[0]))
        y_max = int(np.ceil(y_max / rescale_factor[0]))
        
        # Rotate if needed
        if rotated:
            y_min, y_max, x_max, x_min = (
                x_min, 
                x_max, 
                original_image.shape[1] - y_min, 
                original_image.shape[1] - y_max
            )
       
        # Crop the image to the bounding box
        bounding_box = original_image[y_min:y_max, x_min:x_max, :]

        # If the image contains an alpha channel, squash it
        if bounding_box.shape[2] == 4:
            bounding_box = color.rgba2rgb(bounding_box)

        return bounding_box


def dice_coef(y_true, y_pred):
    """
    Intersection-over-Union coefficient ( between 0 and 1, higher is better ).
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    """
    Turns the dice coefficient into a loss function to be minimised.
    """
    return -dice_coef(y_true, y_pred)


def segmentation_model():
    """
    Loads the U-net model from HDF5 file into a compiled Keras model.
    """
    return load_model(
    "boopnet.hdf5",
    compile=True,
    custom_objects={
        "dice_coef": dice_coef, 
        "dice_coef_loss": dice_coef_loss, 
        "K": K
    })
