# DDE_python
by [Lena R. Bartell, Itai Cohen Group, Cornell Unviersity](#authors)

Direct Deformation Estimation (DDE) analysis of local image deformation. 

DDE is a variation of the Lucas-Kanade (LK) algorithm for estimating image displacements and deformation gradient tensor fields [1-2]. LK estimates image displacements by optimizing a local warp of the template image; the deformation gradient is then calculated by differentiating the displacement field. DDE limits the LK algorithm to a 2D affine warp, which (after optimization) directly maps to the deformation gradient tensor (F), thus eliminating the noise inherent in numerical differentiation of the displacement field. To optimize the warp parameters, this code implements a variation of the Levenberg-Marquadt algorithm (LMA). This LMA variation was implemented in python by Brian D. Leahy and collaborators [3].

## Bibliography
1. Boyle, J.J., Kume, M., Wyczalkowski, M.A., Taber, L.A., Pless, R.B., Xia, Y., Genin, G.M., Thomopoulos, S., 2014. Simple and accurate methods
for quantifying deformation, disruption, and development in biological tissues. Journal of The Royal Society Interface 11, 20140685.
2. Baker, S., Matthews, I., 2004. Lucas-Kanade 20 Years On: A Unifying Framework. International Journal of Computer Vision 56, 221-255.
3. _include reference on PERI optimization here_ (currently in preparation)

## More Information

### Authors

This code was written and/or modified by Lena Bartell during the completion of her PhD thesis in Prof. Itai Cohen's group at Cornell University. Lena was generously supported by NSF GRFP DGE-1144153 and NIH 1F31-AR069977. 

We welcome feedback. Please email Lena with any questions, comments, suggestions, or bugs. 

#### Programmer

Lena R. Bartell  
PhD Candidate  
Applied Physics  
Cornell University  
lrb89@cornell.edu  
[GitHub profile](https://github.com/lbartell)

#### Principal Investigator

Itai Cohen  
Associate Professor  
Department of Physics  
Cornell University  
ic67@cornell.edu  
[Cohen Group website](http://cohengroup.lassp.cornell.edu/)

### Copyright
Copyright &copy; 2016 - 2017 by Cornell University. All Rights Reserved.
 
Permission to use `DDE_python` (the “Work”) and its associated copyrights solely for educational, research and non-profit purposes, without fee is hereby granted, provided that the user agrees as follows:
 
Those desiring to incorporate the Work into commercial products or use Work and its associated copyrights for commercial purposes should contact the Center for Technology Licensing at Cornell University at 395 Pine Tree Road, Suite 310, Ithaca, NY 14850; email: ctl-connect@cornell.edu; Tel: 607-254-4698; FAX: 607-254-5454 for a commercial license.
 
IN NO EVENT SHALL CORNELL UNIVERSITY (“CORNELL”) BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THE WORK AND ITS ASSOCIATED COPYRIGHTS, EVEN IF CORNELL MAY HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
THE WORK PROVIDED HEREIN IS ON AN "AS IS" BASIS, AND CORNELL HAS NO OBLIGATION TO PROVIDE ANY MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.  CORNELL MAKES NO REPRESENTATIONS AND EXTENDS NO WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESS, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF WORK AND ITS ASSOCIATED COPYRIGHTS WILL NOT INFRINGE ANY PATENT, TRADEMARK OR OTHER RIGHTS.