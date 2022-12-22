
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import io
from IPython.display import clear_output
import numpy as np
from ipywidgets import interact, fixed
from glob import glob
import os

def start_plot():
    global metric_values, multires_iterations
    metric_values = []
    multires_iterations = []

def end_plot():
    global metric_values, multires_iterations
    del metric_values
    del multires_iterations
    plt.close()

def plot_values(registration_method):
    global metric_values, multires_iterations
    metric_values.append(registration_method.GetMetricValue())                                       
    clear_output(wait=True)
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))

def read_image(image_path):
    image = io.imread(image_path)
    image = sitk.GetImageFromArray(image[:,0,:,:])
    return image

def align_stacks_itk(moving_image,fixed_image,initial = True,verbose=False,dim=3):
    if initial:
        initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                        moving_image, 
                                                        sitk.Euler3DTransform(), 
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    else:
        initial_transform = sitk.Transform(dim, sitk.sitkIdentity)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=10, numberOfIterations=1000, convergenceMinimumValue=1e-10, convergenceWindowSize=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    if verbose:
        registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                sitk.Cast(moving_image, sitk.sitkFloat32))
    return final_transform

def align_stacks(moving_image,fixed_image):
    moving_image = sitk.GetImageFromArray(moving_image)
    fixed_image = sitk.GetImageFromArray(fixed_image)
    transformation = align_stacks_itk(moving_image,fixed_image,initial = True,verbose=False,dim=3)

def display_images_with_color(image_z, alpha, fixed, moving):
    create_figure_with_color(image_z, fixed, moving)
    plt.show()

def create_figure_with_color(image_z, fixed, moving):
    fixed = fixed/np.max(fixed)
    moving = moving/np.max(moving)
    plt.figure(figsize=(10,10))
    img = np.array([fixed[image_z],moving[image_z],np.zeros(fixed.shape[1:])])
    img = np.swapaxes(img, 0,2)
    plt.imshow(img);
    plt.axis('off')
