import numpy as np

def correlate_pixel(image1,image2):
    assert(len(image1) == len(image2))
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)

    sigma1 = np.sum(np.conjugate(image1-mean1)*(image1-mean1))
    sigma2 = np.sum(np.conjugate(image2-mean2)*(image2-mean2))
    # not positive what should be here...but if we set this to zero, then if the data is exactly
    # the mean, then 
    denom = np.sqrt(sigma1)*np.sqrt(sigma2)
    denom_zero = np.isclose(denom,0)
    # the pixels should be considered correlated if both images are close to the mean and the means are close to eachother
    if np.isclose(mean1,mean2) and denom_zero:
        return 1
    #FIXME
    elif denom_zero:
        return 1
    corr = np.sum(np.conjugate(image1-mean1)*(image2-mean2))/(denom)
    return np.linalg.norm(corr)

def generate_random_image(edge_length=3):
    num_pixels = edge_length**2
    pixels = (np.random.rand(num_pixels)+np.random.rand(num_pixels)*1j).reshape(edge_length,edge_length)
    return pixels


def compute_max_diff_loc(image1,image2):
    diff = (np.real(image1)-np.real(image2))**2+(np.imag(image1)-np.imag(image2))**2
    return np.argmax(diff)



def correlate_images(image1, image2,template_size=3, mask=True):
    assert(image1.shape == image2.shape)
    assert(template_size%2 == 1)
    correlated_image = np.zeros(image1.shape)
    edge_size = template_size//2
    for i in range(edge_size,image1.shape[0]-edge_size):
        for j in range(edge_size,image1.shape[1]-edge_size):
            subimage1 = image1[i-edge_size:i+edge_size+1,j-edge_size:j+edge_size+1].flatten()
            subimage2 = image2[i-edge_size:i+edge_size+1,j-edge_size:j+edge_size+1].flatten()
            if mask:
                max_diff_loc = compute_max_diff_loc(subimage1,subimage2)
                subimage1 = np.delete(subimage1, max_diff_loc)
                subimage2 = np.delete(subimage2, max_diff_loc)
            correlated_image[i,j] = correlate_pixel(subimage1,subimage2)
    return correlated_image


def blocked_mean(image, template_size=3):
    assert(template_size%2 == 1)
    if template_size == 1:
        return image
    mean_image = np.zeros(image.shape)
    edge_size = template_size//2
    for i in range(edge_size,image.shape[0]-edge_size):
        for j in range(edge_size,image.shape[1]-edge_size):
            subimage = image[i-edge_size:i+edge_size+1,j-edge_size:j+edge_size+1]
            mean_image[i,j] = np.mean(subimage)
    return mean_image

def connectivity_threshold(corr1, corr2, threshold=0.2, num_connected=9):
    diff_image = (corr2-corr1)
    threshold_image = diff_image < -1*threshold
    labeled_threshold_image = np.zeros_like(threshold_image)
    return diff_image, labeled_threshold_image

class Analysis(object):
    def __init__(self):
        self._frames = []
        self.correlation_images = None
        self.diff_images = None
        self.threshold_images = None

        self._threshold = None
        self._blocksize = None
        self._mean_blocksize = None
        self._mask_max = None
        self._num_connected = None

    def save(self, name):
        np.savez_compressed(name, frames = self._frames, correlation_images=self.correlation_images,
                           diff_images=self.diff_images, threshold_images=self.threshold_images,
                           threshold=self._threshold,blocksize=self._blocksize,
                           mean_blocksize=self._mean_blocksize,
                           mask_max=self._mask_max, num_connected=self._num_connected)

    def add_frame(self, angle,alignment, linear_retardation=False):
        assert(angle.ndim == 2 and alignment.ndim == 2)
        assert(angle.shape == alignment.shape)
        image = None
        if not linear_retardation:
            image = np.cos(2*angle)*np.sin(alignment)**2+1j*np.sin(2*angle)*np.sin(alignment)**2
        else:
            image = np.cos(2*angle)*alignment+1j*np.sin(2*angle)*alignment
        self._frames.append(image)

    def _correlate_images(self,blocksize,mean_blocksize,mask_max=True):
        self.correlation_images = np.zeros((len(self._frames)-1, self._frames[0].shape[0], self._frames[0].shape[1]))
        for i,(frame1,frame2) in enumerate(zip(self._frames[:-1],self._frames[1:])):
            self.correlation_images[i,:,:] = blocked_mean(correlate_images(frame1,frame2, blocksize,mask_max),mean_blocksize)

    def _threshold_images(self, threshold,num_connected):
        #def connectivity_threshold(corr1, corr2, threshold=0.2, num_connected=9):
        assert(self.correlation_images is not None)
        assert(self.correlation_images.ndim == 3)
        self.diff_images = np.zeros((len(self.correlation_images)-1, self.correlation_images.shape[1], self.correlation_images.shape[2]))
        self.threshold_images = np.zeros((len(self.correlation_images)-1, self.correlation_images.shape[1], self.correlation_images.shape[2]))
        for i,(corr1,corr2) in enumerate(zip(self.correlation_images[:-1],self.correlation_images[1:])):
            self.diff_images[i,:,:],self.threshold_images[i,:,:] = connectivity_threshold(corr1,corr2,threshold,num_connected)

    def run(self, threshold=0.2,blocksize=5,mean_blocksize=1,mask_max=True,num_connected=9):
        assert(len(self._frames)>1)
        self._correlate_images(blocksize,mean_blocksize,mask_max)
        self._threshold_images(threshold,num_connected)
        self._threshold = threshold
        self._blocksize = blocksize
        self._mean_blocksize = mean_blocksize
        self._mask_max = mask_max
        self._num_connected = num_connected


#analysis = VectorCorrelationAnalysis()
#analysis.run(num_connected=1,threshold=0.1,blocksize=3)
