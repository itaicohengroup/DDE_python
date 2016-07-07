"""
Direct Deformation Estimation (DDE) analysis of local image deformation

   Direct Deformation Estimation (DDE) is a variation of the Lucas-Kanade (LK)
   algorithm for estimating image displacements and deformation gradient tensor
   fields [1-2].

   LK estimates image displacements by optimizing a local warp of
   the template image; the deformation gradient is then calculated by
   differentiating the displacement field. DDE limits the LK algorithm to a 2D
   affine warp, which (after optimization) directly maps to the deformation
   gradient tensor (F), thus eliminating the noise inherent in numerical
   differentiation of the displacement field.

   To optimize the warp parameters, this code implements a variation of the
   Levenberg-Marquadt algorithm (LMA). This LMA variation was developed by
   Brian D. Leahy and collaborators [3].

   References:
   [1] Boyle, J.J., Kume, M., Wyczalkowski, M.A., Taber, L.A., Pless, R.B.,
       Xia, Y., Genin, G.M., Thomopoulos, S., 2014. Simple and accurate methods
       for quantifying deformation, disruption, and development in biological
       tissues. Journal of The Royal Society Interface 11, 20140685.
   [2] Baker, S., Matthews, I., 2004. Lucas-Kanade 20 Years On: A Unifying
       Framework. International Journal of Computer Vision 56, 221-255.
   [3] **include reference on PERI optimization here**

   Written by Lena R. Bartell
   June 24, 2016
"""

# Imports
from Tkinter import Tk
from tkFileDialog import askopenfilename
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator as interpolate
import peri.opt.optimize as opt
import matplotlib.pyplot as plt
import os

# globals
DEBUG = False
clear = lambda: os.system('cls')

# Define DDEStack class for DDE analysis of a tiff stack
class DDEStack(object):
    """
    parameters are specified in i,j (not x,y) format
    """

    # Initialize class
    def __init__(self, **kwargs):
        print 'Initializing DDEStack'
        self._set_options(**kwargs)
        self._set_region_centers()
        self._initialize_frames() # list of DDEframes objects
        self._optimize_warp()

    # Set processing parameters as instance variables
    def _set_options(self, filename=None, regionsize=(15, 15),
                    regionspacing=None, euler=True, LMkwargs=None):

        # open the tiff stack, ask for a new file if filename is invalid
        self.filename = filename
        try:
            self.tiffstack = Image.open(self.filename, mode='r')
        except IOError:
            Tk().withdraw() # keep the root window from appearing
            self.filename = askopenfilename()
            self.tiffstack = Image.open(self.filename, mode='r')

        # region size is a 2-element list of odd integers specifing the region
        # extent in pixels using the i,j (not x,y) format
        if len(regionsize)==1:
            regionsize = (regionsize, regionsize)
        else:
            regionsize = (regionsize[0:2])
        regionsize = [int(x) for x in regionsize] # convert to integer list
        regionsize = [x+1 if x%2==0 else x for x in regionsize] # force odd
        self.regionsize = regionsize

        # region spacing is a 2-element list of integers specifying the spacing
        # between region centers in pixels using the i,j format
        if regionspacing==None:
            regionspacing = self.regionsize
        else:
            if len(regionspacing)==1:
                regionspacing = (regionspacing, regionspacing)
            else:
                regionspacing = (regionspacing[0:2])
            regionspacing = [int(x) for x in regionspacing] # list of ints
        self.regionspacing = regionspacing
        
        # compare current image to first image (False, lagrngian) of previous image (True, euler)
        self.euler = euler

        # catch other parameters
        self.LMkwargs = LMkwargs

    # Define region centers
    def _set_region_centers(self):
        halfsize = [(x-1)/2 for x in self.regionsize]
        imsize = self.tiffstack.size
        regionspacing = self.regionspacing
        Yv, Xv = (np.arange(halfsize[k], imsize[k]-halfsize[k],
                            regionspacing[k]) for k in xrange(2))
        self.regions_Y0, self.regions_X0 = np.meshgrid(Yv, Xv, indexing='ij')
        self.num_regions = self.regions_X0.size

    # Initialize each frame in the stack
    def _initialize_frames(self):
        self.frame = list()
        self.num_frames = self.tiffstack.n_frames
        for ff in xrange(self.num_frames):
            self.frame.append(DDEImage(self.tiffstack,self.regions_X0,
                                       self.regions_Y0, self.regionsize, ff))

    # Get processing parameters
    def get_parameters(self):
        params = {
            'filename': self.filename,
            'regionsize': self.regionsize,
            'regionspacing': self.regionspacing,
            'num_frames': self.num_frames,
            'num_regions': self.num_regions,
            'euler': self.euler}
        return params

    def get_image_data(self, frameix, regionix, warped=False):
        if warped==True:
            X = self.frame[frameix].region[regionix].x
            Y = self.frame[frameix].region[regionix].y
        elif warped==False:
            X = self.frame[frameix].region[regionix].X
            Y = self.frame[frameix].region[regionix].Y

        pts = np.vstack((X, Y)).T
        return self.frame[frameix].interp(pts)

    def _get_warped_image_func(self, frameix, regionix):
        def func(p):
            self.frame[frameix].region[regionix].set_warped_coordinates(p)
            return self.get_image_data(frameix, regionix, warped=True)
        return func

    # Optimize warp for all regions in all images
    def _optimize_warp(self):

        num_frames = self.tiffstack.n_frames
        num_regions = self.regions_X0.size
        
        # default initial guess for warp parameters is all zeros
        p0 = np.array([0., 0., 0., 0., 0., 0.], dtype=float)
        
        # loop over each region in each frame
        for ff in xrange(num_frames-1):
            for rr in xrange(num_regions):
                
                # setup for optimization
                # - if comparing to the first image, use the previous time's
                #   optimized parameters as an initial guess for the warp
                if (not self.euler) and (ff>0):
                    p0 = self.frame[ff].region[rr].LMoptimization.param_vals
                
                # - template image data
                if self.euler:
                    data = self.get_image_data(ff, rr, warped=False)
                else:
                    data = self.get_image_data(0, rr, warped=False)
                
                # - function to get warped image data, given warp parameters
                func = self._get_warped_image_func(ff+1, rr)
                
                # use Brian's LM optimization procedure
                LMoptimization = opt.LMFunction(data, func, p0,
                                                **self.LMkwargs)
                LMoptimization.do_run_2()
                self.frame[ff+1].region[rr].LMoptimization = LMoptimization

                if DEBUG and (LMoptimization.get_termination_stats(
                        )['model_cosine'] > 0.5):
                    self.show_warp(ff, rr)

                # update progress
                clear()
                print ' Frame %d of %d, Region %d of %d'%(
                    ff+1, num_frames-1, rr+1, num_regions)

    # Plot region in template and warped image
    def show_warp(self, ff, rr):
        if self.euler:
            template = self.get_image_data(ff, rr, warped=False)
        else:
            template = self.get_image_data(1, rr, warped=False)
        warped = self.get_image_data(ff+1, rr, warped=True)
        plt.figure()
        plt.subplot(121)
        plt.imshow(template.reshape(self.regionsize))
        plt.title('template, frame %d region %d'%(ff+1,rr))
        plt.subplot(122)
        plt.imshow(warped.reshape(self.regionsize))
        plt.title('warped, frame %d region %d'%(ff+1,rr))
        plt.show()

    def show_displacements(self, basename=None, savekwargs=None):
        # initialize figure window
        plt.figure()
        if basename is not None:
            digits = len(str(self.num_frames-1))

        # show each frame one at a time
        for ff in xrange(1, self.num_frames):

            # get image data and points in current and next frame
            im = self.frame[ff].data
            if self.euler:
                x0 = [a.X0 for a in self.frame[ff-1].region]
                y0 = [a.Y0 for a in self.frame[ff-1].region]
            else:
                x0 = [a.X0 for a in self.frame[0].region]
                y0 = [a.Y0 for a in self.frame[0].region]
            x1 = [a.x0 for a in self.frame[ff].region]
            y1 = [a.y0 for a in self.frame[ff].region]
            num_points = len(x0)

            # show each image
            plt.imshow(im)
            plt.hold(True)
            for pp in xrange(num_points):
                plt.plot((x0[pp], x1[pp]), (y0[pp], y1[pp]), 'k.-')
            plt.title('Frame %d'%ff)
            plt.hold(False)
            plt.show()
            
            # save to tif file
            if basename is not None:
                filename = ('%s_%0' + str(digits) + 'd.tif')%(basename, ff)
                plt.savefig(filename, **savekwargs)
            else:
                plt.waitforbuttonpress()
        plt.close()

# Define DDEImage class to hold image data and interpolants
class DDEImage(object):

    # Initialize class
    def __init__(self, tiffstack, X0, Y0, regionsize, frameix):
        self.frameix = frameix
        self._get_image_data(tiffstack)
        self._create_image_interpolants(tiffstack.size)
        self._initialize_regions(X0, Y0, regionsize)

    # Get image data and numeric gradients
    def _get_image_data(self, tiffstack):
        tiffstack.seek(self.frameix)
        self.data = np.array(tiffstack, dtype='float')

    # Create interpolant for image
    def _create_image_interpolants(self, imsize):
        vert_y, vert_x = (np.arange(limit).astype(float) for limit in imsize)
        self.interp = interpolate((vert_x, vert_y), self.data,
                                  bounds_error=False, fill_value=0.)

    # Initialize each region in the image and store them all in a list
    def _initialize_regions(self, X0, Y0, regionsize):
        p0 = np.zeros((6,), dtype='float')
        self.region = list()
        for rr in xrange(X0.flat[:].size):
            self.region.append(DDERegion(X0.flat[rr], Y0.flat[rr],
                                         regionsize, p0))

# Define DDERegion class to create and hold region undeformed and warped
# coordinates of a region
class DDERegion(object):

    def __init__(self, X0, Y0, regsize, p):
        self._set_coordinates(X0, Y0, regsize)
        self.set_warped_coordinates(p)

    def _set_coordinates(self, X0, Y0, regsize):
        delta = [(x-1)/2 for x in regsize]
        Yv = np.arange(Y0-delta[0], Y0+delta[0]+1).astype(float)
        Xv = np.arange(X0-delta[1], X0+delta[1]+1).astype(float)
        self.X0, self.Y0 = X0, Y0
        Y, X = np.meshgrid(Yv, Xv, indexing='ij')
        self.X, self.Y= X.ravel(), Y.ravel()

    def set_warped_coordinates(self, p):
        # set warp paremeters and update the related quantities
        self.p = p
        self.warp = affine2d(self.p)
        self.F = F(self.p)
        self.displacement = displacement(self.p)

        # set warped centroid
        warped_data0 =np.dot(self.warp,
                             np.array((self.X0, self.Y0, 1)).reshape(3, 1))
        self.x0, self.y0 = warped_data0[0][0], warped_data0[1][0]

        # set warped coordinates
        self.x, self.y = (np.empty_like(a) for a in (self.X, self.Y))
        for ii in xrange(self.X.size):
            X = self.X[ii]
            Y = self.Y[ii]
            data = np.array((X, Y, 1.)).reshape((3, 1))
            warped_data = np.dot(self.warp, data)
            self.x.flat[ii] = warped_data[0][0]
            self.y.flat[ii] = warped_data[1][0]

# Function to creat 2d affine transformation matrix given parameters p
def affine2d(p):
    return np.array([
        [1+p[0],   p[2], p[4]],
        [  p[1], 1+p[3], p[5]],
        [     0,      0,    1.]])

# Extract deformation gradient tensor from 2d affine transformation parameters p
def F(p):
    return np.array([
        [1+p[0],   p[2]],
        [  p[1], 1+p[3]]])

# Extract displacement from 2d affine transformation parameters p
def displacement(p):
    return np.array([p[4], p[5]])



















