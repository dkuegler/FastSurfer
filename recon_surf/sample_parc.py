#!/usr/bin/env python3


# Copyright 2024 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
import argparse
import os
import sys
from concurrent.futures import Executor, ThreadPoolExecutor, Future
from pathlib import Path
from typing import Callable, TypeVar, Optional, Sequence
from logging import getLogger as _get_logger

import numpy as np
import nibabel.freesurfer.io as fs
import nibabel as nib
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from lapy import TriaMesh

from smooth_aparc import smooth_aparc, _ReadSurfaceWithMetadata

T_Label = TypeVar("T_Label", bound=np.integer)
T_VertexId = TypeVar("T_VertexId", bound=np.integer)
T_VertexCoord = TypeVar("T_VertexCoord", bound=np.floating)
TV = TypeVar("TV", bound=np.number)

HELPTEXT = """
Script to sample labels from image to surface and clean up. 

USAGE:
sample_parc --inseg <segimg> --insurf <surf> --incort <cortex.label>
            --seglut <seglut> --surflut <surflut> --outaparc <out_aparc>
            --projmm <float> --radius <float>


Dependencies:
    Python 3.8

    Numpy
    http://www.numpy.org

    Nibabel to read and write FreeSurfer surface meshes
    http://nipy.org/nibabel/


Original Author: Martin Reuter
Date: Dec-18-2023

"""

h_inseg = "path to input segmentation image"
h_incort = "path to input cortex label mask"
h_insurf = "path to input surface"
h_outaparc = "path to output aparc"
h_surflut = "FreeSurfer look-up-table for values on surface"
h_seglut = "Look-up-table for values in segmentation image (rows need to correspond to surflut)"
h_projmm = "Sample along normal at projmm distance (in mm), default 0"
h_radius = "Search around sample location at radius (in mm) for label if 'unknown', default None"

logger = _get_logger(__name__)


def options_parse() -> argparse.Namespace:
    """
    Create a command line interface and return command line options.

    Returns
    -------
    options : argparse.Namespace
        A Namespace object holding options.
    """
    parser = argparse.ArgumentParser(
        usage=HELPTEXT,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="$Id: smooth_aparc,v 1.0 2018/06/24 11:34:08 mreuter Exp $",
    )
    parser.add_argument("--inseg", dest="inseg", help=h_inseg)
    parser.add_argument("--insurf", dest="insurf", help=h_insurf)
    parser.add_argument("--incort", dest="incort", help=h_incort)
    parser.add_argument("--surflut", dest="surflut", help=h_surflut)
    parser.add_argument("--seglut", dest="seglut", help=h_seglut)
    parser.add_argument("--outaparc", dest="outaparc", help=h_outaparc)
    parser.add_argument(
        "--projmm",
        dest="projmm",
        help=h_projmm,
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--radius",
        dest="radius",
        help=h_radius,
        default=None,
        type=float,
    )
    args = parser.parse_args()

    if args.insurf is None or args.inseg is None or args.outaparc is None:
        sys.exit("ERROR: Please specify input surface, input image and output aparc!")

    if args.surflut is None or args.seglut is None:
        sys.exit("ERROR: Please specify surface and segmentatin image LUT!")

    # maybe later add functionality, to not have a cortex label, e.g. 
    # like FreeSurfer find largest connected component and fill only
    # the other unknown regions
    if args.incort is None:
        sys.exit("ERROR: Please specify surface cortex label!")

    return args


def construct_adj_cluster(
    tria: npt.NDArray[T_VertexId],
    annot: npt.NDArray[T_Label],
) -> sparse.csc_matrix:
    """
    Compute adjacency matrix of edges from same annotation label only.

    Operates only on triangles and removes edges that cross annotation
    label boundaries.

    Parameters
    ----------
    tria : npt.NDArray[int]
        Array of vertex edges.
    annot : npt.NDArray[int]
        Array of per vertex annotation labels.

    Returns
    -------
    sparse.csc_matrix
        The non-directed adjacency matrix
        will be symmetric. Each inner edge (i,j) will have
        the number of triangles that contain this edge.
        Inner edges usually 2, boundary edges 1. Higher
        numbers can occur when there are non-manifold triangles.
        The sparse matrix can be binarized via:
        adj.data = np.ones(adj.data.shape).
     """
    t0 = tria[:, 0]
    t1 = tria[:, 1]
    t2 = tria[:, 2]
    i = np.column_stack((t0, t1, t1, t2, t2, t0)).reshape(-1)
    j = np.column_stack((t1, t0, t2, t1, t0, t2)).reshape(-1)
    ia = annot[i]
    ja = annot[j]
    keep_edges = (ia == ja)
    i = i[keep_edges]
    j = j[keep_edges]
    dat = np.ones(i.shape)
    n = annot.shape[0]
    return sparse.csc_matrix((dat, (i, j)), shape=(n, n))


def find_all_islands(
    surf: tuple[npt.NDArray[T_VertexCoord], npt.NDArray[T_VertexId]],
    annot: npt.NDArray[T_Label],
) -> npt.NDArray[T_Label]:
    """
    Find vertices in disconnected islands for all labels in surface annotation.

    Parameters
    ----------
    surf : tuple[npt.NDArray[float], npt.NDArray[int]]
        Surface as returned by nibabel fs.read_geometry, where:
        surf[0] is the np.array of (n, 3) vertex coordinates and
        surf[1] is the np.array of (m, 3) triangle indices.
    annot : npt.NDArray[int]
        Annotation as an int array of (n,) with label ids for each vertex.
        This is for example the first element of the tupel returned by
        nibabel fs.read_annot.

    Returns
    -------
    vidx : npt.NDArray[int] (i,)
        Array listing vertex indices of island vertices, empty if no islands
        (components disconnected from the largest label region) are found.
    """
    from concurrent.futures import ThreadPoolExecutor, Future

    def get_island_indices_by_label(lid: int, labs: np.ndarray) -> np.ndarray:
        mask = annot == lid
        ll = labs[mask]
        lidx = np.arange(labs.size)[mask]
        lmax = np.bincount(ll).argmax()
        v = lidx[(ll != lmax)]
        if v.size > 0:
            logger.warning(
                f"Found disconnected islands ({v.size} vertices total) for label {lid}!"
            )
        return v

    with ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        # explicitly define callable to fix typing warning
        unique: Callable[[npt.NDArray[TV]], npt.NDArray[TV]] = np.unique
        # compute disconnected components
        lids: Future[npt.NDArray[T_Label]] = pool.submit(unique, annot)

        # construct adjacency matrix without edges across regions:
        adjacency_matrix = construct_adj_cluster(surf[1], annot)
        # for each label, get islands that are not connected to main component
        num_components, labels = connected_components(
            csgraph=adjacency_matrix,
            directed=False,
            return_labels=True,
        )
        from functools import partial
        func = partial(get_island_indices_by_label, labs=labels)
        return np.concatenate(
            list(pool.map(func, lids.result())),
            dtype=np.int32)


def sample_nearest_nonzero(
    img: nib.analyze.SpatialImage,
    vox_coords: npt.NDArray[float],
    radius: float = 3.0,
) -> npt.NDArray[int]:
    """
    Sample closest non-zero value in a ball of radius around vox_coords.

    Parameters
    ----------
    img : nibabel.analyze.SpatialImage
        Image to sample. Voxels need to be isotropic.
    vox_coords : npt.NDArray[float](n,3)
        Coordinates in voxel space around which to search.
    radius : float default 3.0
        Consider all voxels inside this radius to find a non-zero value.

    Returns
    -------
    samples : np.ndarray(n,)
        Sampled values (dtype of img), returns zero for vertices where values are zero
        in ball.
    """
    # check for isotropic voxels 
    voxsize = img.header.get_zooms()
    logger.info(f"Check isotropic vox sizes: {voxsize}")
    if not np.allclose(voxsize, voxsize[0], atol=0.001, rtol=0.0):
        raise AssertionError('Voxels not isotropic!')
    data = np.asarray(img.dataobj)

    # radius in voxels:
    rvox = radius * voxsize[0]

    # sample window around nearest voxel
    if np.issubdtype(vox_coords.dtype, np.integer):
        x_nn = vox_coords
    else:
        x_nn = np.rint(vox_coords).astype(dtype=int)
    # Reason: to always have the same number of voxels that we check
    # and to be consistent with FreeSurfer, we center the window at
    # the nearest neighbor voxel, instead of at the float vox coordinates

    # create box with 2*rvox+1 side length to fully contain ball
    # and get coordiante offsets with zero at center
    ri = np.floor(rvox).astype(int)
    ll = np.arange(-ri,ri+1)
    xv, yv, zv = np.meshgrid(ll, ll, ll)
    # modify distances slightly, to avoid randomness when
    # sorting with different radius values for voxels that otherwise
    # have the same distance to center
    xvd = xv + 0.001
    yvd = yv + 0.002
    zvd = zv + 0.003
    ddm = np.sqrt(xvd * xvd + yvd * yvd + zvd * zvd).flatten()
    # also compute correct distance for ball mask below
    dd = np.sqrt(xv * xv + yv * yv + zv * zv).flatten()
    ddball = dd <= rvox

    # flatten and keep only ball with radius
    xv = xv.flatten()[ddball]
    yv = yv.flatten()[ddball]
    zv = zv.flatten()[ddball]
    ddm = ddm[ddball]

    # stack to get offset vectors
    offsets = np.column_stack((xv, yv, zv))

    # sort offsets according to distance
    # Note: we keep the first zero voxel so we can later
    # determine if all voxels are zero with the argmax trick
    sortidx = np.argsort(ddm)
    offsets = offsets[sortidx, :]

    # reshape and tile to add to list of coords
    # move axes and broadcast (axis0 will broadcast to x_nn)
    toffsets = offsets.transpose()[np.newaxis]
    s_coords = x_nn[:, :, np.newaxis] + toffsets

    # get image data at the s_coords locations
    s_data = data[s_coords[:, 0], s_coords[:, 1], s_coords[:, 2]]

    # get first non-zero if possible
    nzidx = np.not_equal(s_data, 0).argmax(axis=1)
    # the above return index zero if all elements are zero which is OK for us
    # as we can then sample there and get a value of zero
    samples = s_data[np.arange(s_data.shape[0]), nzidx]
    return samples


def sample_img(
    surf: Path | str | tuple[npt.NDArray[T_VertexCoord], npt.NDArray[T_VertexId]],
    img: Path | str | nib.MGHImage,
    cortex: Optional[Path | str | npt.NDArray[T_Label]] = None,
    projmm: float = 0.0,
    radius: Optional[float] = None,
) -> np.ndarray:
    """
    Sample volume at a distance from the surface.

    Parameters
    ----------
    surf : tuple[np.ndarray, np.ndarray], Path, str
        Surface as returned by nibabel fs.read_geometry, where:
        surf[0] is the np.ndarray of (n, 3) vertex coordinates and
        surf[1] is the np.ndarray of (m, 3) triangle indices.
        If type is str, read surface from file.
    img : nibabel.MGHImage, Path, str
        Image to sample.
        If type is str, read image from file.
    cortex : np.ndarray, Path, str
        Filename of cortex label or np.ndarray with cortex indices.
    projmm : float, default=0.0
        Sample projmm mm along the surface vertex normals (default=0).
    radius : float, optional 
        If given and if the sample is equal to zero, then consider
        all voxels inside this radius to find a non-zero value.

    Returns
    -------
    samples : np.ndarray (n,)
        Sampled values from img (same dtype as img).
    """
    if isinstance(surf, (str, Path)):
        surf = fs.read_geometry(surf, read_metadata=True)
    if isinstance(img, (str, Path)):
        img = nib.load(img)
    if not isinstance(img, nib.MGHImage):
        # MGHImage is needed because get_vox2ras_tkr is only defined for MGHHeader
        raise TypeError(f"img must be an MGHImage, but is {type(img).__name__}!")
    if isinstance(cortex, (str, Path)):
        cortex = fs.read_label(cortex)
    nvert = surf[0].shape[0]
    # Compute Cortex Mask
    if cortex is not None:
        mask = np.zeros(nvert, dtype=bool)
        mask[cortex] = True
    else:
        mask = np.ones(nvert, dtype=bool)

    data = np.asarray(img.dataobj)
    # Use LaPy TriaMesh for vertex normal computation
    mesh = TriaMesh(surf[0], surf[1])

    # make sure the triangles are oriented (normals pointing to the same direction
    if not mesh.is_oriented():
        print("WARNING: Surface is not oriented, flipping corrupted normals.")
        mesh.orient_()
    # compute sample coordinates projmm mm along the surface normal
    # in surface RAS coordiante system:
    x = mesh.v + projmm * mesh.vertex_normals()
    # mask cortex
    xx = x[mask]

    # compute Transformation from surface RAS to voxel space:
    vox2ras_orig = img.header.get_vox2ras_tkr()
    ras2vox = np.linalg.inv(vox2ras_orig)
    x_vox = np.dot(xx, ras2vox[:3, :3].T) + ras2vox[:3, 3]
    # sample at nearest voxel
    x_nn = np.rint(x_vox).astype(int)
    samples_nn = data[x_nn[:, 0], x_nn[:, 1], x_nn[:, 2]]
    # no search for zeros, done:
    if not radius:
        samplesfull = np.zeros(nvert, dtype="int")
        samplesfull[mask] = samples_nn
        return samplesfull
    # search for zeros, but no zeros exist, done:
    zeros = np.asarray(samples_nn == 0).nonzero()[0]
    if zeros.size == 0:
        samplesfull = np.zeros(nvert, dtype="int")
        samplesfull[mask] = samples_nn
        return samplesfull
    # here we need to do the hard work of searching in a windows
    # for non-zero samples
    logger.info(f"sample_img: found {zeros.size} zero samples, searching radius ...")
    z_nn = x_nn[zeros]
    z_samples = sample_nearest_nonzero(img, z_nn, radius=radius)
    samples_nn[zeros] = z_samples
    samplesfull = np.zeros(nvert, dtype="int")
    samplesfull[mask] = samples_nn
    return samplesfull


def replace_labels(
    img_labels: npt.NDArray[T_Label],
    img_lut: Path | str,
    surf_lut: Path | str,
    *,
    executor: Optional[Executor] = None,
) -> tuple[npt.NDArray[T_Label], npt.NDArray[int], npt.NDArray[str]]:
    """
    Replace image labels with corresponding surface labels or unknown.

    Parameters
    ----------
    img_labels : np.ndarray(n,)
        Array with image label ids.
    img_lut : str, Path
        Filename for image label look up table.
    surf_lut : str, Path
        Filename for surface label look up table.
    executor : concurrent.futures.Executor, optional
        Executor object for parallel IO.
    Returns
    -------
    surf_labels : np.ndarray (n,)
        Array with surface label ids.
    surf_ctab : np.ndarray shape(m,4)
        Surface color table (RGBA).
    surf_names : np.ndarray[str] shape(m,)
        Names of label regions.
    """
    # Helper function for typing and parallel IO
    def _load_lut(_f: Path | str) -> Future[npt.NDArray[int]]:
        return np.loadtxt(_f, usecols=(0, 2, 3, 4, 5), dtype=int)

    def _load_names(_f: Path | str) -> Future[npt.NDArray[str]]:
        return np.loadtxt(_f, usecols=(1,), dtype=str)

    if executor is None:
        surflut = _load_lut(surf_lut)
        surf_names = _load_names(surf_lut)
        imglut = _load_lut(img_lut)
        img_names = _load_names(img_lut)
    else:
        # same as clause above, but in different threads
        # first: schedule tasks
        _surflut = executor.submit(_load_lut, surf_lut)
        _surf_names = executor.submit(_load_names, surf_lut)
        _imglut = executor.submit(_load_lut, img_lut)
        _img_names = executor.submit(_load_names, img_lut)
        # then: wait for and assign the results
        surflut = _surflut.result()
        surf_names = _surf_names.result()
        imglut = _imglut.result()
        img_names = _img_names.result()

    surf_ids = surflut[:, 0]
    surf_ctab = surflut[:, 1:5]
    surf_ctab =  surflut[:,1:5]
    img_ids = imglut[:,0]
    img_ids = imglut[:, 0]
    assert (np.all(img_names == surf_names)), "Label names in the LUTs do not agree!"
    lut = np.zeros((img_labels.max() + 1,), dtype=img_labels.dtype)
    lut[img_ids] = surf_ids
    surf_labels = lut[img_labels]
    return surf_labels, surf_ctab, surf_names


def sample_parc(
    surf: tuple[npt.NDArray[T_VertexCoord], npt.NDArray[T_VertexId]] | Path | str,
    seg: nib.analyze.SpatialImage | Path | str,
    imglut: Path | str,
    surflut: Path | str,
    outaparc: Path | str,
    cortex: Optional[np.ndarray | str | Path] = None,
    projmm: float = 0.0,
    radius: Optional[float] = None,
) -> int | str:
    """
    Sample cortical GM labels from image to surface and smooth.

    Parameters
    ----------
    surf : tuple, str, Path
        Surface as returned by nibabel fs.read_geometry, where:
        surf[0] is the np.array of (n, 3) vertex coordinates and
        surf[1] is the np.array of (m, 3) triangle indices.
        If type is str, read surface from file.
    seg : nibabel.image, str, Path
        Image to sample.
        If type is str, read image from file.
    imglut : str, Path
        Filename for image label look up table.
    surflut : str, Path
        Filename for surface label look up table.
    outaparc : str, Path
        Filename for output surface parcellation.
    cortex : np.ndarray, str, Path, optional
        Filename of cortex label or np.ndarray with cortex indices.
    projmm : float, default=0.0
        Sample projmm mm along the surface vertex normals (default=0).
    radius : float, optional
        If given and if the sample is equal to zero, then consider
        all voxels inside this radius to find a non-zero value.

    Returns
    -------
    int, str
        Zero, if successful, or an error message
    """
    # define aliases for common/more descriptive typing
    _AnnotType = tuple[np.ndarray, np.ndarray, list[str]]
    read_surf: _ReadSurfaceWithMetadata = fs.read_geometry
    read_cortex: Callable[[Path | str], np.ndarray] = fs.read_label
    with ThreadPoolExecutor(8) as pool:
        # read input files (this only submits these operations to be done in indendent
        # threads so IO can be parallel
        if isinstance(surf, (str, Path)):
            _surf: Future = pool.submit(read_surf, surf, read_metadata=True)
        if isinstance(cortex, (str, Path)):
            _cortex: Future[np.ndarray] = pool.submit(read_cortex, cortex)
        if isinstance(seg, (str, Path)):
            seg = nib.load(seg)
            seg_data = np.asarray(seg.dataobj)

        # get rid of unknown labels first and translate the rest (avoids too much
        # filling later as sampling will search around sample point if label is zero)
        surf_segdata, surf_ctab, surf_names = replace_labels(
            seg_data,
            imglut,
            surflut,
            executor=pool,
        )

        if isinstance(surf, (str, Path)):
            logger.info(f"Reading in surface: {surf} ...")
            if _surf.exception() is None:
                surf = _surf.result()
            else:
                logger.exception(_surf.exception())
                return "Failed reading the input surface file."
        if isinstance(cortex, (str, Path)):
            logger.info(f"Reading in cortex label: {cortex} ...")
            if _cortex.exception() is None:
                cortex = _cortex.result()
            else:
                logger.exception(_cortex.exception())
                return "Failed reading the cortex label file."

    try:
        # create img with new data (needed by sample img)
        seg2 = nib.MGHImage(surf_segdata, seg.affine, seg.header)
        # sample from image to surface (and search if zero label)
        surfsamples = sample_img(surf, seg2, cortex, projmm, radius)
        # find label islands
        vidx = find_all_islands(surf, surfsamples)
        # set islands to zero (to ensure they get smoothed away later)
        surfsamples[vidx] = 0
        # smooth boundaries and remove islands inside cortex region
        smooths = smooth_aparc(surf, surfsamples, cortex)
        # write annotation
        fs.write_annot(outaparc, smooths, ctab=surf_ctab, names=surf_names)
    except Exception as e:
        logger.exception(e)
        return e.args[0]
    return 0


if __name__ == "__main__":
    from logging import INFO, basicConfig
    basicConfig(
        level=INFO,
        format="%(filename)s[%(lineno)d] %(levelname)s: %(message)s",
    )

    # Command Line options are error checking done here
    options = options_parse()

    sys.exit(sample_parc(
        options.insurf,
        options.inseg,
        options.seglut,
        options.surflut,
        options.outaparc,
        options.incort,
        options.projmm,
        options.radius,
    ))
