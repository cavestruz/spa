import numpy as np
from skimage import measure, draw
from skimage.morphology import label


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
    # 
    _, region = get_region_properties(image_data, contour)
    axis.imshow(region)

  if annotate : 
    axis.annotate(annotate, (0.5,0.9),fontsize='xx-large',
                             color='white', xycoords='axes fraction') 


