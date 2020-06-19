import numpy as np
from skimage import measure, draw
from skimage.morphology import label
from astropy.io import fits
from matplotlib.colors import LogNorm
from skimage import measure, draw
from skimage.morphology import label
import matplotlib.patches as patch
from matplotlib.patches import Ellipse
import math

properties_from_regionsprops = ['area', 'bbox', 'bbox_area', 
                                                 'centroid', 'convex_area',
                                                 'eccentricity', 'equivalent_diameter',
                                                 'extent', 'inertia_tensor', 
                                                 'inertia_tensor_eigvals', 'local_centroid',
                                                 'major_axis_length', 'max_intensity',
                                                 'mean_intensity', 'min_intensity',
                                                 'minor_axis_length', 'moments',
                                                 'moments_central', 'moments_hu',
                                                 'moments_normalized', 'orientation',
                                                 'perimeter', 'weighted_centroid', 
                                                 'weighted_local_centroid', 'weighted_moments',
                                                 'weighted_moments_central', 'weighted_moments_hu',
                                                 'weighted_moments_normalized' 
                                                 ]

def get_longest_contour(image, contour_value) :
  '''
  Input:
  image: array of numbers
  contour_value: float that is the value of pixels in the image we want to pull out
  Returns longest contour at a given value, length means the contour likely
  encloses the BCG'''

  contours = measure.find_contours(image, contour_value)
  longest_contour = max(contours, key = lambda i: i.size)

  return longest_contour

def get_pixels_in_contour(image_shape, contour) :
  '''Uses grid_points_in_poly to get the boolean for pixels inside the contour
      Returns an nxm array of True and False'''

  return measure.grid_points_in_poly(image_shape, contour)

def visualize_substructure(image_data, substructure_contour, axis,
                          annotate=False,
                          plot_only_region=False) :
  '''Visualize a cluster with imshow on an axis, options to draw contours and
  to annotate.'''

  if not plot_only_region: axis.imshow(image_data, norm=LogNorm())

  axis.plot(substructure_contour[:,1],substructure_contour[:,0], lw=2., color='red')
  if plot_only_region :
    _, region = get_region_properties(image_data, substructure_contour)
    axis.imshow(region)

  if annotate :
    axis.annotate(annotate, (0.5,0.9),fontsize='xx-large',
                             color='white', xycoords='axes fraction')

def zoom_image(image, zoom_factor=1) :
  '''Zoom to the center of the image by some factor'''

  return image[int( image.shape[0]/2 )-int( image.shape[0]/(zoom_factor*2) ):\
               int( image.shape[0]/2 )+int( image.shape[0]/(zoom_factor*2) ),\
               int( image.shape[1]/2 )-int( image.shape[1]/(zoom_factor*2) ):\
               int( image.shape[1]/2 )+int( image.shape[1]/(zoom_factor*2) ),\
               ]

print(get_pixels_in_contour)

def get_region_properties(image, contour) : 
  '''Get properties of region carved out by longest contour using
  skimage.measure regionprops.  Returns the regionprops object and the region to 
  plot/visually compare'''

  region_img = get_pixels_in_contour(image.shape, contour)
  label_image = label(region_img)
  region_properties = measure.regionprops(label_image,intensity_image=image)
  selected_region = region_img.astype(np.uint8)
  return region_properties, selected_region

def visualize_cluster(filename, axis, draw_contour=True,
                      get_contour_method=get_longest_contour,
                      get_contour_kwargs={'contour_value':1e-4},
                      zoom_factor=1, annotate=False,
                      plot_only_region=False) :
  '''Visualize a cluster with imshow on an axis, options to draw contours and
  to annotate.'''
  image_data = fits.getdata(filename)
  image_data = zoom_image(image_data, zoom_factor=zoom_factor)
  if not plot_only_region: axis.imshow(image_data, norm=LogNorm())

  if draw_contour :
    contour = get_contour_method(image_data, **get_contour_kwargs)
    axis.plot(contour[:,1],contour[:,0], lw=2., color='red')
    if plot_only_region :
      # Below, selected_region correpsonds to the pixels in the contour
      _, selected_region = get_region_properties(image_data, contour)
      axis.imshow(selected_region)
  if annotate :
    axis.annotate(annotate, (0.5,0.9),fontsize='xx-large',
                             color='white', xycoords='axes fraction')

def get_region_properties_dictionary(filename,
                                     get_contour_method=get_longest_contour,
                                     get_contour_kwargs={'contour_value': 1e-4},
                                     properties=properties_from_regionsprops) :
  '''Create a dictionary of the region properties per cluster.  Properties are from
  the skimage.measure.regionsprops documentation'''
  image_data = zoom_image(fits.getdata(filename))

  longest_contour =  get_contour_method(image_data, **get_contour_kwargs)
  region_properties, _ = get_region_properties(image_data, longest_contour)
  region_properties_dict = {prop: region_properties[0][prop] for prop in properties}

  return region_properties_dict

def radial_profile(filename, radial_bins, angular_range=None):
    """The function to output the radial profile of a given npz file.

        Parameters:
            filename (str):  filename of npz file
            field (str): field name for profile, default is 'star_sigma'
            radial_bins (arr): bins for the radial profile
            angular_range (tuple): min, max of the angular range
        Returns:
            radial_profile (arr): array of profile values
    """
    image_data = zoom_image(fits.getdata(filename))

    center = (image_data.shape[0]/2,image_data.shape[0]/2)
    y, x = np.indices((image_data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    bin_labels = np.digitize(r, radial_bins)
    #r = r.astype(np.int)

    tbin = np.bincount(bin_labels.ravel(), image_data.ravel())
    nr = np.bincount(bin_labels.ravel())
    radial_profile = tbin / nr
    return radial_profile

def get_isocontour_regions(image, contour_value) : 
  '''
  Returns contours at a given value, separating substructures from the contour enclosing the BCG
  Input: 
  image: array of numbers
  contour_value: float that is the value of pixels in the image we want to pull out

  Return:
  substructures: list of arrays
'''
  
  contours = measure.find_contours(image, contour_value)  
  sorted(contours, key = lambda i: i.size)
  return contours[1:]

