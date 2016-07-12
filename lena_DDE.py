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

   Lena R. Bartell
"""

# Imports
from Tkinter import Tk
from tkFileDialog import askopenfilename
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator as interpolate
import peri.opt.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os

# globals
DEBUG = False
clear = lambda: os.system('cls') # Works for windows, not mac?

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
        if self.euler:
            self._traceback_F()

    # Set processing parameters as instance variables
    def _set_options(self, filename=None, regionsize=(15, 15),
                    regionspacing=None, euler=True, LMkwargs={}):

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
        if isinstance(regionsize, (int, long)):
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
            if isinstance(regionspacing, (int, long)):
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
        imsize = (self.tiffstack.height, self.tiffstack.width)
        regionspacing = self.regionspacing
        Yv, Xv = (np.arange(halfsize[k], imsize[k]-halfsize[k],
                            regionspacing[k]) for k in xrange(2))
        self.regions_Y0, self.regions_X0 = np.meshgrid(Yv, Xv, indexing='ij')
        self.regions_Yv, self.regions_Xv = Yv, Xv
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

        pts = np.vstack((Y, X)).T
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
    
    # Traceback deformation gradient tensor components in time and interpolate
    # to get full deformation gradient tensor at each initial region. Only used
    # for 'euler' analysis where deformation is optimized between current and 
    # preceeding frame (rather than 1st frame comparison)
    def _traceback_F(self):
        
        # setup initial values
        f12_prev, f21_prev = [np.zeros(self.num_regions) for _ in xrange(2)]
        f11_prev, f22_prev = [np.ones(self.num_regions) for _ in xrange(2)]
        x_prev, y_prev = self.regions_X0, self.regions_Y0
        
        # for each time point
        for tt in xrange(1, self.num_frames):
            
            # get data for each component of displacement and F (from previous
            # to current time point)
            dxdata  = [a.displacement[0] for a in self.frame[tt].region]
            dydata  = [a.displacement[1] for a in self.frame[tt].region]
            f11data  = [a.F[0,0] for a in self.frame[tt].region]
            f12data  = [a.F[0,1] for a in self.frame[tt].region]
            f21data  = [a.F[1,0] for a in self.frame[tt].region]
            f22data  = [a.F[1,1] for a in self.frame[tt].region]
            
            shape = self.regions_X0.shape
            dxdata, dydata, f11data, f12data, f21data, f22data = \
                [np.array(a).reshape(shape) for a in 
                (dxdata, dydata, f11data, f12data, f21data, f22data)]
            
            # create scalar field interpolants for each component of 
            # displacement and F
            vert_y, vert_x = self.regions_Yv, self.regions_Xv
            dx, dy, f11, f12, f21, f22 = \
                [interpolate((vert_y, vert_x), a, bounds_error=False, fill_value=None) 
                for a in (dxdata, dydata, f11data, f12data, f21data, f22data)] #(y,x)!
            
            # for each region
            for rr in xrange(self.num_regions):
                
                # interpolate to get displacement to next time point
                x0, y0 = x_prev.flat[rr], y_prev.flat[rr]
                pts = np.vstack((y0, x0)).T
                x1 = x0 + dx(pts)
                y1 = y0 + dy(pts)
                
                # interpolate to get deformation gradient to next time point
                F0 = np.array([[f11_prev[rr], f12_prev[rr]], [f21_prev[rr], f22_prev[rr]]])
                Ftmp = np.array([[f11(pts)[0], f12(pts)[0]], [f21(pts)[0], f22(pts)[0]]])
                F1 = np.dot(Ftmp, F0)
                
                # store the results
                self.frame[tt].region[rr].F_traced = F1
                self.frame[tt].region[rr].displacement_traced = \
                    np.array([dx(pts), dy(pts)])
                
                # set for next iteration
                x_prev[rr], y_prev[rr] = x1, y1
                f11_prev[rr], f12_prev[rr], f21_prev[rr], f22_prev[rr] = \
                    F1.flat[:]
            
        # replace euler-description data with interpolated lagrangian 
        # descripion
        for tt in xrange(1, self.num_frames):
            for rr in xrange(self.num_regions):
                self.frame[tt].region[rr].displacement_euler = \
                    self.frame[tt].region[rr].displacement
                
                self.frame[tt].region[rr].displacement = \
                    self.frame[tt].region[rr].displacement_traced
                    
                del self.frame[tt].region[rr].displacement_traced
                
                self.frame[tt].region[rr].F_euler = \
                    self.frame[tt].region[rr].F
                
                self.frame[tt].region[rr].F = \
                    self.frame[tt].region[rr].F_traced
                
                del self.frame[tt].region[rr].F_traced
        

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

    def show_deformation(self, basename=None, savekwargs={},
                         showim=True, imcolormap='gray', showboxes=True,
                         showstrain=True, straincolormap='seismic',
                         strainlim=0.1, alpha=0.75, xlim=None, ylim=None,
                         showcolorbar=True):
        # initialize figure window
        plt.figure()
        if basename is not None:
            digits = len(str(self.num_frames-1))

        # setup for plotting box corners
        if showboxes or showstrain:
            Yoffset = (self.regionsize[0]-1)/2 * np.array((-1, 1, 1, -1, -1))
            Xoffset = (self.regionsize[1]-1)/2 * np.array((-1, -1, 1, 1, -1))

        # show each frame one at a time
        for ff in xrange(1, self.num_frames):

            # clear the figure
            plt.hold(False)
            plt.clf()
            plt.hold(True)

            # get and show image
            if showim:
                im = self.frame[ff].data
                plt.imshow(im, cmap=imcolormap)

            # get and show grid box deformation & strain for each region
            if showboxes or showstrain:

                patches = []
                strains = []

                # for each region
                for rr in xrange(self.num_regions):

                    # get undeformed box center
#                    if self.euler:
#                        X0 = self.frame[ff-1].region[rr].X0
#                        Y0 = self.frame[ff-1].region[rr].Y0
#                    else:
                    X0 = self.frame[0].region[rr].X0
                    Y0 = self.frame[0].region[rr].Y0

                    # calculate undeformed box corners
                    X, Y = X0 + Xoffset, Y0 + Yoffset

                    # calculate deformed box corners
                    warp = self.frame[ff].region[rr].warp
                    warped_corners = np.dot(warp, np.array((X, Y, 1)).reshape(3, 1))
                    x, y = warped_corners[0][0], warped_corners[1][0]

                    # plot deforned box corners
                    if showboxes:
                        plt.plot(x, y, 'r.-', lw=.5, ms=3)

                    # plot patch with fill color based on strain value
                    if showstrain:
                        F = self.frame[ff].region[rr].F
                        E = (np.dot(F.T, F) - np.eye(2)) / 2
                        strain = E[0,0]
#                        color = scalarMap.to_rgba(strain)
#                        poly = Polygon(np.vstack((x,y)).T, facecolor=color,
#                                       edgecolor='none', alpha=alpha)
                        poly = Polygon(np.vstack((x,y)).T)
                        patches.append(poly)
                        strains.append(strain)

                if showstrain:
                    p = PatchCollection(patches, cmap=straincolormap, alpha=alpha)
                    p.set_array(np.array(strains))
                    p.set_clim([-strainlim, strainlim])
                    plt.gca().add_collection(p)

                # Add a colorbar
                if showstrain and showcolorbar:
                    plt.colorbar(p, label='strain')

            # set axes limits
            if xlim is not None:
                plt.gca().set_xlim(xlim)
            if ylim is not None:
                plt.gca().set_ylim(xlim)

            # finish figure and, if requested, save to a tif file
            if basename is not None:
                plt.title('Frame %d'%ff)
                plt.show()
                filename = ('%s_%0' + str(digits) + 'd.tif')%(basename, ff)
                plt.savefig(filename, **savekwargs)
            else:
                plt.title('Frame %d (click to advance)'%ff)
                plt.show()
                plt.waitforbuttonpress()

        # close figure window
        plt.close()

# Define DDEImage class to hold image data and interpolants
class DDEImage(object):

    # Initialize class
    def __init__(self, tiffstack, X0, Y0, regionsize, frameix):
        self.frameix = frameix
        self._get_image_data(tiffstack)
        self._create_image_interpolants((tiffstack.height, tiffstack.width))
        self._initialize_regions(X0, Y0, regionsize)

    # Get image data and numeric gradients
    def _get_image_data(self, tiffstack):
        tiffstack.seek(self.frameix)
        self.data = np.array(tiffstack, dtype='float')

    # Create interpolant for image
    def _create_image_interpolants(self, imsize):
        vert_y, vert_x = (np.arange(limit).astype(float) for limit in imsize)
        self.interp = interpolate((vert_y, vert_x), self.data,
                                  bounds_error=False, fill_value=None) #(y,x)!

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



















