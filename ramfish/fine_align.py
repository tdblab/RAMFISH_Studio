"""Step 3 processing: the registration code of ramfish_fine_aligner.py, unchanged."""

from __future__ import annotations

import cv2
import numpy as np
import SimpleITK as sitk


def read_image_bytes(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError("File could not be decoded. Ensure it is a valid 2D image.")
    return img


def match_fixed_shape_centered(moving_img: np.ndarray, fixed_shape: tuple) -> np.ndarray:
    fixed_h, fixed_w = fixed_shape
    moving_h, moving_w = moving_img.shape
    if (fixed_h, fixed_w) == (moving_h, moving_w):
        return moving_img
    canvas = np.zeros((fixed_h, fixed_w), dtype=moving_img.dtype)
    start_y_canvas = max(0, (fixed_h - moving_h) // 2)
    start_x_canvas = max(0, (fixed_w - moving_w) // 2)
    start_y_moving = max(0, (moving_h - fixed_h) // 2)
    start_x_moving = max(0, (moving_w - fixed_w) // 2)
    copy_h = min(fixed_h, moving_h)
    copy_w = min(fixed_w, moving_w)
    canvas[start_y_canvas:start_y_canvas + copy_h, start_x_canvas:start_x_canvas + copy_w] = \
        moving_img[start_y_moving:start_y_moving + copy_h, start_x_moving:start_x_moving + copy_w]
    return canvas


def compute_transforms(fixed_ref_np, moving_ref_np, bspline=True):
    """Affine then non-rigid B-Spline registration.

    ``bspline=False`` stops after the affine step (no local bending): used for tissue masks, whose flat interior
    gives the B-Spline nothing to hold on to, so it would smear the inside of the tissue.
    """
    fixed_image = sitk.GetImageFromArray(fixed_ref_np.astype(np.float32))
    moving_image = sitk.GetImageFromArray(moving_ref_np.astype(np.float32))

    fixed_image_reg = sitk.Normalize(fixed_image)
    moving_image_reg = sitk.Normalize(moving_image)

    blur_filter = sitk.DiscreteGaussianImageFilter()
    blur_filter.SetVariance(4.0)
    fixed_image_reg = blur_filter.Execute(fixed_image_reg)
    moving_image_reg = blur_filter.Execute(moving_image_reg)

    affine_reg = sitk.ImageRegistrationMethod()
    affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    affine_reg.SetMetricSamplingStrategy(affine_reg.RANDOM)
    affine_reg.SetMetricSamplingPercentage(0.1)
    affine_reg.SetInterpolator(sitk.sitkLinear)
    affine_reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    affine_reg.SetOptimizerScalesFromPhysicalShift()
    initial_affine = sitk.CenteredTransformInitializer(
        fixed_image_reg, moving_image_reg, sitk.AffineTransform(2),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    affine_reg.SetInitialTransform(initial_affine, inPlace=False)
    affine_transform = affine_reg.Execute(fixed_image_reg, moving_image_reg)

    affine_resampler = sitk.ResampleImageFilter()
    affine_resampler.SetReferenceImage(fixed_image_reg)
    affine_resampler.SetInterpolator(sitk.sitkLinear)
    affine_resampler.SetDefaultPixelValue(0)
    affine_resampler.SetTransform(affine_transform)
    moving_image_affine = affine_resampler.Execute(moving_image_reg)

    if not bspline:
        return affine_transform, sitk.Transform(2, sitk.sitkIdentity), fixed_image_reg

    bspline_reg = sitk.ImageRegistrationMethod()
    bspline_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    bspline_reg.SetMetricSamplingStrategy(bspline_reg.RANDOM)
    bspline_reg.SetMetricSamplingPercentage(0.1)
    bspline_reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=50)
    bspline_reg.SetInterpolator(sitk.sitkLinear)
    initial_bspline = sitk.BSplineTransformInitializer(fixed_image_reg, [8, 8])
    bspline_reg.SetInitialTransform(initial_bspline, inPlace=False)
    bspline_reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    bspline_reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    bspline_transform = bspline_reg.Execute(fixed_image_reg, moving_image_affine)

    return affine_transform, bspline_transform, fixed_image_reg


def apply_transforms(affine_transform, bspline_transform, fixed_image_ref, moving_target_np):
    moving_target = sitk.GetImageFromArray(moving_target_np.astype(np.float32))

    affine_resampler = sitk.ResampleImageFilter()
    affine_resampler.SetReferenceImage(fixed_image_ref)
    affine_resampler.SetInterpolator(sitk.sitkLinear)
    affine_resampler.SetDefaultPixelValue(0)
    affine_resampler.SetTransform(affine_transform)
    moving_target_affine = affine_resampler.Execute(moving_target)

    bspline_resampler = sitk.ResampleImageFilter()
    bspline_resampler.SetReferenceImage(fixed_image_ref)
    bspline_resampler.SetInterpolator(sitk.sitkLinear)
    bspline_resampler.SetDefaultPixelValue(0)
    bspline_resampler.SetTransform(bspline_transform)

    aligned_target_sitk = bspline_resampler.Execute(moving_target_affine)
    return sitk.GetArrayFromImage(aligned_target_sitk).astype(moving_target_np.dtype)
