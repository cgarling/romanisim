"""
Unit tests for PSF functions.
"""
import pytest

import numpy as np
from romanisim import psf
from romanisim.models.bandpass import galsim2roman_bandpass
import galsim
import galsim.roman


class FakeWCS():
    def __init__(self):
        pass

    def toWorld(self, pos):
        return galsim.CelestialCoord(pos.x * 0.1 * galsim.arcsec,
                                     pos.y * 0.1 * galsim.arcsec)

    def local(self, *args, **kwargs):
        return galsim.JacobianWCS(0.1, 0, 0, 0.1)


@pytest.mark.parametrize("args, kwargs, position", [
    ((1, 'F087'), {'psftype': 'stpsf', 'nlambda': 1}, None),
    ((2, 'F184'), {'psftype': 'stpsf', 'nlambda': 1}, None),
    ((3, 'F087'), {'psftype': 'epsf'}, None),
    ((4, 'F184'), {'psftype': 'galsim'}, None),
    ((5, 'H158'), {'psftype': 'galsim'}, None),
    ((6, 'H158'), {'psftype': 'galsim', 'chromatic': True}, None),
    ((7, 'F184'), {'pix': (1000, 1000), 'psftype': 'galsim'}, None),
    ((8, 'F184'), {'pix': (1000, 1000), 'psftype': 'stpsf', 'nlambda': 1}, None),
    ((9, 'F184'), {'pix': (1000, 1000), 'psftype': 'epsf'}, None),
    ((10, 'F129'), {'psftype': 'stpsf', 'wcs': FakeWCS(), 'nlambda': 1}, None),
    ((11, 'F087'), {'psftype': 'stpsf', 'variable': True, 'nlambda': 1}, (100, 100)),
    ((12, 'F129'), {'psftype': 'epsf', 'wcs': FakeWCS()}, None),
    ((13, 'F087'), {'psftype': 'epsf', 'variable': True}, (100, 100))])
def test_make_psf(args, kwargs, position):
    p = psf.make_psf(*args, **kwargs)
    if position is not None:
        p = p.at_position(*position)

    bandpass = galsim.roman.getBandpasses(AB_zeropoint=True)['H158']
    vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')

    if not kwargs.get('chromatic', False):
        method = 'auto'
        im = p.drawImage(method=method).array
    else:
        im = (p * vega_sed.withFlux(1, bandpass)).drawImage(bandpass).array
    totsum = np.sum(im)
    assert totsum < 1
    assert totsum > 0.9
    # assert that image catches no more than 100% and no less than 90%
    # of flux?
    assert np.min(im) > np.max(im) * (-1e-3)
    # ideally nothing negative


@pytest.mark.parametrize("sca", [1, 2])
def test_get_epsf_from_crds_detector_match(sca):
    """get_epsf_from_crds must request the SCA-specific CRDS detector
    (WFIxx); a wrong detector value (e.g. SCAxx) fails to match any
    SCA-specific epsf rmap entry and CRDS silently falls back to the
    same generic, non-SCA-specific reference file for every SCA."""
    filter_name = 'F087'
    model = psf.get_epsf_from_crds(sca, filter_name)
    assert model.meta.instrument.detector == f'WFI{sca:02d}'
    assert model.meta.instrument.optical_element == galsim2roman_bandpass[filter_name]


def test_get_epsf_from_crds_detector_varies_by_sca():
    """Different SCAs must resolve to different epsf reference files;
    if the CRDS detector header were wrong, every SCA would collapse
    onto the same fallback reference."""
    filter_name = 'F087'
    model1 = psf.get_epsf_from_crds(1, filter_name)
    model2 = psf.get_epsf_from_crds(2, filter_name)
    assert model1.meta.instrument.detector != model2.meta.instrument.detector


def test_get_gridded_psf_model_uses_noipc():
    """The gridded PSF model must be built from the IPC-free ``psf_noipc``
    array, not the IPC-convolved ``psf`` array.  romanisim.l1.make_l1 applies
    IPC to the resultants, so using ``psf`` here would convolve IPC twice."""
    focus, spectral_type = 0, 1
    model = psf.get_epsf_from_crds(3, 'F087')
    gridded = psf.get_gridded_psf_model(
        model, focus=focus, spectral_type=spectral_type)

    noipc = np.asarray(model.psf_noipc[focus, spectral_type])
    withipc = np.asarray(model.psf[focus, spectral_type])

    np.testing.assert_array_equal(gridded.data, noipc)
    # guard against the reference file shipping identical arrays, which would
    # make the check above pass vacuously
    assert not np.array_equal(noipc, withipc)


def test_epsf_noipc_plus_romanisim_ipc_matches_psf():
    """Convolving the reference ``psf_noipc`` array with romanisim's IPC kernel
    must reproduce the reference ``psf`` array to high precision.

    This checks that romanisim applies the IPC kernel in the same orientation
    that was used to build the CRDS ePSF reference (i.e. no transpose or axis
    flip between the implementations).  The correct orientation matches at
    ~5e-8 while the most favorable wrong orientation (``fliplr`` of the kernel)
    differs at ~1e-4.

    Notes
    -----
    The reference ``psf`` array was produced by stpsf using a kernel that
    matches romanisim's built-in default ``romanisim.models.ipc.ipc_kernel``
    (both descend from the same draft WFI simulation documentation).  A change
    to either kernel would legitimately break this test.  This exercises only
    the ``usecrds=False`` kernel path; with ``usecrds=True`` ``make_l1`` applies
    the CRDS ``ipc`` reference kernel instead, which differs from this one by
    ~15-25% in the wings.
    """
    from romanisim import l1
    from romanisim.models.ipc import ipc_kernel

    focus, spectral_type, grid_index = 0, 1, 4
    model = psf.get_epsf_from_crds(3, 'F087')

    noipc = np.asarray(model.psf_noipc[focus, spectral_type, grid_index],
                       dtype=np.float64)
    withipc = np.asarray(model.psf[focus, spectral_type, grid_index],
                         dtype=np.float64)

    # No kernel argument -> uses the default romanisim.models.ipc.ipc_kernel,
    # the same path make_l1 takes when ipc_model is None.
    convolved = l1.add_ipc(noipc[None, :, :])[0]

    # psf and psf_noipc in the reference file carry a constant relative
    # normalization (withipc.sum() / noipc.sum() =  0.994599), so
    # compare after normalizing both to unit sum.  The metric is the largest
    # absolute deviation as a fraction of the reference peak.
    ref = withipc / withipc.sum()

    def peak_resid(arr):
        arr = arr / arr.sum()
        return np.max(np.abs(arr - ref)) / np.max(ref)

    assert peak_resid(convolved) < 1e-6

    # The kernel is asymmetric enough that a flipped application is caught:
    # this proves the check above is actually orientation-sensitive.
    flipped = l1.add_ipc(noipc[None, :, :], np.asarray(ipc_kernel)[:, ::-1])[0]
    assert peak_resid(flipped) > 1e-5
