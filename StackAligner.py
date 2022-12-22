import os
from glob import glob
import SimpleITK as sitk
from skimage import io
import numpy as np
from alignment_tools import align_stacks_itk,create_figure_with_color
from tifffile import imsave
from scipy import ndimage

class StackAligner:
    def __init__(self,image_path):
        self.image_path = image_path
        self.average_image_path = os.path.join(self.image_path,'average_images')
        self.transformation_path = os.path.join(self.image_path,'transformations')
        self.transformed_image_path = os.path.join(self.image_path,'transformed_image')
        self.inspection_path = os.path.join(self.image_path,'inspection')
        os.makedirs(self.average_image_path,exist_ok=True)
        os.makedirs(self.transformation_path,exist_ok=True)
        os.makedirs(self.transformed_image_path,exist_ok=True)
        os.makedirs(self.inspection_path,exist_ok=True)
        self.transformations = sorted(glob(os.path.join(self.transformation_path,'*.tfm')))
        self.tifs = sorted(glob(os.path.join(self.image_path,'*.tif')))
        self.transformed = sorted(glob(os.path.join(self.transformed_image_path,'*.tif')))
        self.ntifs = len(self.tifs)
        self.chunk_size=500
        self.n_chunks = np.floor(self.ntifs/self.chunk_size).astype(int)

    def np_to_itk(self,image):
        return sitk.GetImageFromArray(image)
    
    def itk_to_np(self,image):
        return sitk.GetArrayViewFromImage(image)

    def load_average_image(self,offset=0):
        average_path = os.path.join(self.average_image_path,f'average_images_{offset}.tif')
        if not os.path.exists(average_path):
            self.save_average_image()
        return io.imread(average_path).astype(np.int16)
    
    def load_fixed_image(self,offset=0):
        average_image = self.load_average_image(offset)
        return self.np_to_itk(average_image)
    
    def get_tifi_name(self,tifi):
        return os.path.basename(self.tifs[tifi])[:-4]
    
    def get_tifi_transformation_path(self,tifi):
        tifi_name = self.get_tifi_name(tifi)
        return os.path.join(self.transformation_path,tifi_name+'.tfm')
    
    def get_path_to_tifi(self,tifi):
        return os.path.join(self.image_path,self.tifs[tifi])

    def load_tifi(self,tifi):
        image_path = self.get_path_to_tifi(tifi)
        return io.imread(image_path)

    def load_transformed_image(self,tifi):
        image_path = self.transformed[tifi]
        return io.imread(image_path)[:,0,:,:]

    
    def load_moving_image(self,tifi,channel=1):
        image = self.load_tifi(tifi)
        return self.np_to_itk(image[:,channel-1,:,:])
    
    def load_itk_image(self,image_path,channel=1):
        image = io.imread(image_path)[:,channel-1,:,:]
        return self.np_to_itk(image)
    
    def get_offsets(self):
        return np.array(list(range(self.n_chunks)))*self.chunk_size
    
    def load_transformation(self,transformationi):
        transformation_path = self.transformations[transformationi]
        return sitk.ReadTransform(transformation_path)
    
    def load_image_and_transform(self,transformationi):
        transformation_path = self.transformations[transformationi]
        file_name = os.path.basename(transformation_path)
        image_path = os.path.join(self.image_path,file_name[:-4]+'.tif')
        moving_image_ch1 = self.load_itk_image(image_path,channel=1)
        moving_image_ch2 = self.load_itk_image(image_path,channel=2)
        transformation = sitk.ReadTransform(transformation_path)
        return moving_image_ch1,moving_image_ch2,transformation
    
    def apply_transformation(self,fixed,moving,transformation):
        return sitk.Resample(moving, fixed, transformation, sitk.sitkLinear, 0.0, moving.GetPixelID())
    
    def load_tifi_transformation(self,tifi):
        transformation_path = self.get_tifi_transformation_path(tifi)
        return sitk.ReadTransform(transformation_path)
    
    def create_average_image(self,offset = 0):
        images = []
        fixed_image = self.load_moving_image(0+offset)
        for tifi in range(10):
            for _ in range(10):
                try:
                    moving_image = self.load_moving_image(tifi+offset+1)
                    transformation = align_stacks_itk(moving_image,fixed_image,verbose=False,initial = True)
                    moving_resampled = self.apply_transformation(fixed_image,moving_image,transformation)
                    moving_resampled = self.itk_to_np(moving_resampled)
                    images.append(moving_resampled)
                    break
                except:
                    continue
        images = np.array(images)
        average_image = np.mean(images,axis=0)
        return average_image
    
    def save_average_image(self):
        average_image = self.create_average_image()
        save_path = os.path.join(self.average_image_path,f'average_images_0.tif')
        imsave(save_path,average_image)
    
    def align_tifi(self,tifi):
        for _ in range(10):
            try:
                moving = self.load_moving_image(tifi)
                fixed = self.load_fixed_image()
                transformation = align_stacks_itk(moving,fixed,verbose=False,initial=True)
                save_path = self.get_tifi_transformation_path(tifi)
                sitk.WriteTransform(transformation, save_path)
                break
            except:
                ...
    
    def create_inspection_tifi(self,i):
        tif_path = self.get_path_to_tifi(i)
        original_image = self.load_average_image()
        file_name = os.path.basename(tif_path)
        transformed_path = os.path.join(self.transformed_image_path,file_name)
        transformed_image = io.imread(transformed_path)[:,0,:,:]
        transformed_image[np.isnan(transformed_image)]==0
        fixed = transformed_image/(np.max(transformed_image)+0.0001)
        moving = original_image/np.max(original_image)
        image_z = np.floor(len(transformed_image)/2).astype(int)
        img = np.array([fixed[image_z],moving[image_z],np.zeros(fixed.shape[1:])])
        img = np.swapaxes(img, 0,2)
        downsampled_img = ndimage.interpolation.zoom(img,[.5,.5,1])
        save_path = os.path.join(self.inspection_path,file_name[:-4]+'.png')
        io.imsave(save_path,downsampled_img)

    def compare_images(self,imagea,imageb):
        imagea = self.itk_to_np(imagea)
        imageb = self.itk_to_np(imageb)
        create_figure_with_color(6, imagea, imageb)
    
    def apply_alignment_to_tifi(self,tifi):
        transformation = self.load_transformation(tifi)
        average_image = self.load_fixed_image()
        image_ch1 = self.load_moving_image(tifi,channel=1)
        image_ch2 = self.load_moving_image(tifi,channel=2)
        tranformed_ch1 = self.apply_transformation(average_image,image_ch1,transformation)
        tranformed_ch2 = self.apply_transformation(average_image,image_ch2,transformation)
        tranformed_ch1 = np.array(self.itk_to_np(tranformed_ch1))
        tranformed_ch2 = np.array(self.itk_to_np(tranformed_ch2))
        transformed = np.stack([tranformed_ch1,tranformed_ch2])
        transformed = np.swapaxes(transformed,0,1)
        filename = os.path.basename(self.get_path_to_tifi(tifi))
        save_path = os.path.join(self.transformed_image_path,filename)
        imsave(save_path,transformed)

def mutual_information(imagea,imageb):
    hgram, x_edges, y_edges = np.histogram2d(imagea.ravel(),imageb.ravel(),bins=100)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) 
    py = np.sum(pxy, axis=0) 
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0 
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def run_alignment(image_path):
    aligner = StackAligner(image_path)
    for i in range(aligner.ntifs):
        aligner = StackAligner(image_path)
        aligner.align_tifi(i)
        aligner = StackAligner(image_path)
        aligner.apply_alignment_to_tifi(i)
        aligner = StackAligner(image_path)
        aligner.create_inspection_tifi(i)