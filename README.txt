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
   Levenberg-Marquadt algorithm (LMA). This LMA variation was implemented in
   python by Brian D. Leahy and collaborators [3].

   References:
   [1] Boyle, J.J., Kume, M., Wyczalkowski, M.A., Taber, L.A., Pless, R.B.,
       Xia, Y., Genin, G.M., Thomopoulos, S., 2014. Simple and accurate methods
       for quantifying deformation, disruption, and development in biological
       tissues. Journal of The Royal Society Interface 11, 20140685.
   [2] Baker, S., Matthews, I., 2004. Lucas-Kanade 20 Years On: A Unifying
       Framework. International Journal of Computer Vision 56, 221-255.
   [3] **include reference on PERI optimization here**

   Lena R. Bartell