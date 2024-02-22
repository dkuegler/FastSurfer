#!/usr/bin/env python3


# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
import optparse
import os
import sys
from pathlib import Path
from typing import Callable, Literal, Optional, Protocol, TypeVar
from logging import getLogger

import numpy as np
import nibabel.freesurfer.io as fs
from numpy import typing as npt
from scipy import sparse


HELPTEXT = """
Script to fill holes and smooth aparc labels. 

USAGE:
smooth_aparc  --insurf <surf> --inaparc <in_aparc> --incort <cortex.label> --outaparc <out_aparc>


Dependencies:
    Python 3.8+

    Numpy
    http://www.numpy.org

    Nibabel to read and write FreeSurfer surface meshes
    http://nipy.org/nibabel/


Original Author: Martin Reuter
Date: Jul-24-2018

"""

h_inaparc = "path to input aparc"
h_incort = "path to input cortex label"
h_insurf = "path to input surface"
h_outaparc = "path to output aparc"

logger = getLogger(__name__)

T_Label = TypeVar("T_Label", bound=np.integer)
T_VertexFaces = TypeVar("T_VertexFaces", bound=np.integer)
T_VertexCoord = TypeVar("T_VertexCoord", bound=np.floating)
_SurfWithMetaData = tuple[
    npt.NDArray[T_VertexCoord],
    npt.NDArray[T_VertexFaces],
    dict[str, np.ndarray],
]


class _ReadSurfaceWithMetadata(Protocol):
    def __call__(self, __f: Path | str, read_metadata: True) -> _SurfWithMetaData: ...


def options_parse():
    """
    Create a command line interface and return command line options.

    Returns
    -------
    options
        Namespace object holding options.
    """
    parser = optparse.OptionParser(
        version="$Id: smooth_aparc,v 1.0 2018/06/24 11:34:08 mreuter Exp $",
        usage=HELPTEXT,
    )
    parser.add_option("--insurf", dest="insurf", help=h_insurf)
    parser.add_option("--incort", dest="incort", help=h_incort)
    parser.add_option("--inaparc", dest="inaparc", help=h_inaparc)
    parser.add_option("--outaparc", dest="outaparc", help=h_outaparc)
    (options, args) = parser.parse_args()

    if options.insurf is None or options.inaparc is None or options.outaparc is None:
        sys.exit("ERROR: Please specify input surface, input and output aparc")

    return options


def get_adjM(trias: npt.NDArray[int], n: int):
    """
    Create symmetric sparse adjacency matrix of triangle mesh.

    Parameters
    ----------
    trias : npt.NDArray[int](m, 3)
        Triangle mesh matrix.
        
    n : int
        Shape of output (n,n) adjaceny matrix, where n>=m.

    Returns
    -------
    adjM : np.ndarray (bool) shape (n,n)
        Symmetric sparse CSR adjacency matrix, true corresponds to an edge.
    """
    T = trias
    J = T[:, [1, 2, 0]]
    # flatten
    T = T.flatten()
    J = J.flatten()
    adj = sparse.csr_matrix((np.ones(T.shape, dtype=bool), (T, J)), shape=(n, n))
    # if max adj is > 1 we have non manifold or mesh trias are not oriented
    # if matrix is not symmetric, we have a boundary
    # in case we have boundary, make sure this is a symmetric matrix
    adjM = (adj + adj.transpose()).astype(bool)
    return adjM


def bincount2D_vectorized(a: npt.NDArray) -> np.ndarray:
    """
    Count number of occurrences of each value in array of non-negative ints.

    Parameters
    ----------
    a : np.ndarray
        Input 2D array of non-negative ints.

    Returns
    -------
    np.ndarray
        Array of counted values.
    """
    N = a.max() + 1
    a_offs = a + np.arange(a.shape[0])[:, None] * N
    return np.bincount(a_offs.ravel(), minlength=a.shape[0] * N).reshape(-1, N)


def mode_filter(
        adjM: sparse.csr_matrix,
        labels: npt.NDArray[T_Label],
        fillonlylabel: Optional[T_Label] = None,
        novote: npt.ArrayLike = ()
) -> npt.NDArray[T_Label]:
    """
    Apply mode filter (smoothing) to integer labels on mesh vertices.

    Parameters
    ----------
    adjM : sparse.csr_matrix[bool]
        Symmetric adjacency matrix defining edges between vertices,
        this determines what edges can vote so usually one adds the
        identity to the adjacency matrix so that each vertex is included
        in its own vote.
    labels : npt.NDArray[int]
        List of integer labels at each vertex of the mesh.
    fillonlylabel : int, optional
        Label to fill exclusively. None (default): smooth all labels.
    novote : npt.ArrayLike, default=()
        Label ids that should not vote.

    Returns
    -------
    labels_new : npt.NDArray[int]
        New smoothed labels.
    """
    # make sure labels lengths equals adjM dimension
    num_labels = labels.shape[0]
    if num_labels != adjM.shape[0] or num_labels != adjM.shape[1]:
        raise RuntimeError(
            f"mode_filter: adjM size {adjM.shape} does not match label length "
            f"{labels.shape}."
        )
    # remove rows with only a single entry from adjM
    # if we removed some triangles, we may have isolated vertices
    # adding the eye to adjM will produce these entries
    # since they are neighbors to themselves, this adds
    # values to nlabels below that we don't want
    counts = np.diff(adjM.indptr)
    rows = np.where(counts == 1)
    pos = adjM.indptr[rows]
    adjM.data[pos] = 0
    adjM.eliminate_zeros()
    # for num rings exponentiate adjM and add adjM from step before
    # we currently do this outside of mode_filter
    # new labels will be the same as old almost everywhere
    labels_new = labels
    # find vertices to fill
    # if fillonlylabels empty, fill all
    if not fillonlylabel:
        ids = np.arange(0, num_labels)
    else:
        # select the ones with the labels
        ids = np.where(labels == fillonlylabel)[0]
        if ids.size == 0:
            logger.warning(
                f"WARNING: No ids found with idx {fillonlylabel}  ... continue"
            )
            return labels
    # of all ids to fill, find neighbors
    nbrs = adjM[ids, :]
    # get vertex ids (I, J ) of each edge in nbrs
    [II, JJ, VV] = sparse.find(nbrs)
    # check if we have neighbors with -1 or 0
    # this could produce problems in the loop below, so lets stop for now:
    nlabels = labels[JJ]
    if any(nlabels == -1) or any(nlabels == 0):
        raise RuntimeError("there are -1 or 0 labels in neighbors!")
    # create sparse matrix with labels at neighbors
    nlabels = sparse.csr_matrix((labels[JJ], (II, JJ)))
    # print("nlabels: {}".format(nlabels))
    from scipy.stats import mode

    if not isinstance(nlabels, sparse.csr_matrix):
        raise ValueError("Matrix must be CSR format.")
    # novote = [-1,0,fillonlylabel]
    # get rid of rows that have uniform vote (or are empty)
    # for this to work no negative numbers should exist
    # get row counts, max and sums
    rmax = nlabels.max(1).toarray().squeeze()
    sums = np.asarray(nlabels.sum(axis=1)).ravel()
    counts = np.diff(nlabels.indptr)
    # then keep rows where max*counts differs from sums
    rmax = np.multiply(rmax, counts)
    rows = np.where(rmax != sums)[0]
    logger.info(f"rows: {nlabels.shape[0]}  reduced to {rows.size}")
    # Only after fixing the rows above, we can
    # get rid of entries that should not vote
    # since we have only rows that were non-uniform, they should not become empty
    # rows may become unform: we still need to vote below to update this label
    if novote:
        rr = np.isin(nlabels.data, novote)
        nlabels.data[rr] = 0
        nlabels.eliminate_zeros()
    # run over all rows and compute mode (maybe vectorize later)
    rempty = 0
    for row in rows:
        rvals = nlabels.data[nlabels.indptr[row] : nlabels.indptr[row + 1]]
        if rvals.size == 0:
            rempty += 1
            continue
        # print(str(rvals))
        mvals = mode(rvals, keepdims=True)[0]
        # print(str(mvals))
        if mvals.size != 0:
            # print(str(row)+' '+str(ids[row])+' '+str(mvals[0]))
            labels_new[ids[row]] = mvals[0]
    if rempty > 0:
        # should not happen
        print("WARNING: row empty: " + str(rempty))

    # idptr = nlabels.indptr
    # _indices = (idptr[rows], idptr[rows + 1])
    # _counts = _indices[1] - _indices[0]
    #
    # #
    # max_group_size = 8
    # row_val_by_size = [
    #     [(r, nlabels.data[i0:i1])
    #         for r, i0, i1, c in zip(rows, *_indices, _counts) if j == c]
    #     for j in range(max_group_size)
    # ]
    # row_val_max_size = [
    #     (r, nlabels.data[i0:i1])
    #     for r, i0, i1, c in zip(rows, *_indices, _counts) if c >= max_group_size
    # ]
    #
    # rempty = len(row_val_by_size[0])
    # if rempty > 0:
    #     # sanity-check all rows exist / should not happen
    #     logger.warning(f"row empty: {rempty}")
    #
    # # if rows of group size 1 exist, the mode filter always returns the 1st value
    # if len(row_val_by_size[1]) > 0:
    #     rows, rvals = map(np.asarray, zip(*row_val_by_size[1]))
    #     labels_new[ids[rows]] = rvals[:, 0]
    #
    # for r_v in row_val_by_size[2:]:
    #     if len(r_v) > 0:
    #         _rows, rvals = map(np.asarray, zip(*r_v))
    #         mvals = mode(rvals, keepdims=True, axis=1)[0]
    #         if mvals.size != 0:
    #             labels_new[ids[_rows]] = mvals[:, 0]
    #
    # for row, mvals in map(lambda r: (r[0], mode(r[1], keepdims=True)[0]),
    #                       row_val_max_size):
    #     if mvals.size != 0:
    #         labels_new[ids[row]] = mvals[0]

    # nbrs=np.squeeze(np.asarray(nbrs.todense())) # sparse matrix to dense matrix to np.array
    # nlabels=labels[nbrs]
    # counts = np.bincount(nlabels)
    # vote=np.argmax(counts)
    return labels_new


def smooth_aparc(
    surf: tuple[npt.NDArray[T_VertexCoord], npt.NDArray[T_VertexFaces]],
    labels: npt.NDArray[T_Label],
    cortex: Optional[npt.NDArray[T_VertexFaces]] = None,
):
    """
    Smooth aparc label regions on the surface and fill holes.

    First all labels with 0 and -1 unside cortex are filled via repeated
    mode filtering, then all labels are smoothed first with a wider and
    then with smaller filters to produce smooth label boundaries. Labels
    outside cortex are set to -1 at the end.

    Parameters
    ----------
    surf : nibabel surface
        Suface filepath and name of source.
    labels : np.array[int]
        Labels at each vertex (int).
    cortex : np.array[int]
        Vertex ids inside cortex mask.

    Returns
    -------
    smoothed_labels : np.array[int]
        Smoothed labels.
    """
    faces = surf[1]
    nvert = labels.size
    if labels.size != surf[0].shape[0]:
        raise RuntimeError(
            f"ERROR smooth_aparc: vertec count {surf[0].shape[0]} does not match "
            f"label length {labels.size}."
        )

    # Compute Cortex Mask
    if cortex is not None:
        mask = np.zeros(labels.shape, dtype=bool)
        mask[cortex] = True
    else:
        mask = np.ones(labels.shape, dtype=bool)
    # check if we have places where non-cortex has some labels
    noncortnum = np.where(~mask & (labels != -1))
    print(
        "Non-cortex vertices with labels: " + str(noncortnum[0].size)
    )  # num of places where non cortex has some real labels
    # here we need to decide how to deal with them
    # either we set everything outside cortex to -1 (the FS way)
    # or we keep these real labels and allow them to vote, maybe even shrink cortex
    # label? Probably not.

    # get non-cortex ids (here we could subtract the ids that have a real label)
    # for now we remove everything outside cortex
    noncortids = np.where(~mask)

    # remove triangles where one vertex is non-cortex to avoid these edges to vote on
    # neighbors later
    rr = np.isin(faces, noncortids)
    rr = np.reshape(rr, faces.shape)
    rr = np.amax(rr, 1)
    faces = faces[~rr, :]

    # get Edge matrix (adjacency)
    adjM = get_adjM(faces, nvert)

    # add identity so that each vertex votes in the mode filter below
    adjM = adjM + sparse.eye(adjM.shape[0])

    # print("adj shape: {}".format(adjM.shape))
    # print("v shape: {}".format(surf[0].shape))
    # print("labels shape: {}".format(labels.size))
    # print("labels: {}".format(labels))
    # print("minlab: "+str(np.min(labels))+" maxlab: "+str(np.max(labels)))

    # set all labels inside cortex that are -1 or 0 to fill label
    labels = labels.copy()
    fillonlylabel = np.max(labels) + 1
    labels[mask & (labels == -1)] = fillonlylabel
    labels[mask & (labels == 0)] = fillonlylabel
    # now we do not have any -1 or 0 (except 0 outside of cortex)
    # FILL HOLES
    ids = np.where(labels == fillonlylabel)[0]
    counter = 1
    idssize = ids.size
    while idssize != 0:
        print("Fill Round: " + str(counter))
        labels_new = mode_filter(adjM, labels, fillonlylabel, np.array([fillonlylabel]))
        labels = labels_new
        ids = np.where(labels == fillonlylabel)[0]
        if ids.size == idssize:
            # no more improvement, strange could be an island in the cortex label that
            # cannot be filled
            print(
                "Warning: Cannot improve but still have holes. Maybe there is an "
                "island in the cortex label that cannot be filled with real labels."
            )
            fillids = np.where(labels == fillonlylabel)[0]
            labels[fillids] = 0
            rr = np.isin(faces, fillids)
            rr = np.reshape(rr, faces.shape)
            rr = np.amax(rr, 1)
            faces = faces[~rr, :]
            # get Edge matrix (adjacency)
            adjM = get_adjM(faces, nvert)
            # add identity so that each vertex votes in the mode filter below
            adjM = adjM + sparse.eye(adjM.shape[0])
            break
        idssize = ids.size
        counter += 1
    # SMOOTH other labels (first with wider kernel then again fine-tune):
    adjM2 = adjM * adjM
    adjM4 = adjM2 * adjM2
    labels = mode_filter(adjM4, labels)
    labels = mode_filter(adjM2, labels)
    labels = mode_filter(adjM, labels)
    # set labels outside cortex to -1
    labels[~mask] = -1
    return labels


def main(
    insurfname: str | Path,
    inaparcname: str | Path,
    incortexname: str | Path,
    outaparcname: str | Path,
) -> Literal[0] | str:
    """
    Read files, smooth the aparc labels on the surface and save the smoothed labels.

    Parameters
    ----------
    insurfname : str, Path
        Suface filepath and name of source.
    inaparcname : str, Path
        Annotation filepath and name of source.
    incortexname : str, Path
        Label filepath and name of source.
    outaparcname : str, Path
        Surface filepath and name of destination.

    Returns
    -------
    int, str
        Zero, if successfull, an error message otherwise
    """
    from concurrent.futures import ThreadPoolExecutor, Future
    # define aliases for common/more descriptive typing
    _AnnotType = tuple[np.ndarray, np.ndarray, list[str]]
    read_surf: _ReadSurfaceWithMetadata = fs.read_geometry
    read_annot: Callable[[Path | str], _AnnotType] = fs.read_annot
    read_cortex: Callable[[Path | str], np.ndarray] = fs.read_label
    with ThreadPoolExecutor(3) as pool:
        # read input files (this only submits these operations to be done in indendent
        # threads so IO can be parallel
        _surf: Future = pool.submit(read_surf, insurfname, read_metadata=True)
        _aparc: Future[_AnnotType] = pool.submit(read_annot, inaparcname)
        _cortex: Future[np.ndarray] = pool.submit(read_cortex, incortexname)

    logger.info(f"Reading in surface: {insurfname} ...")
    if _surf.exception() is None:
        surf = _surf.result()
    else:
        logger.exception(_surf.exception())
        return "Failed reading the input surface file."
    logger.info(f"Reading in annotation: {inaparcname} ...")
    if _aparc.exception() is None:
        aparc = _aparc.result()
    else:
        logger.exception(_aparc.exception())
        return "Failed reading the aparc annotation file."
    logger.info(f"Reading in cortex label: {incortexname} ...")
    if _cortex.exception() is None:
        cortex = _cortex.result()
    else:
        logger.exception(_cortex.exception())
        return "Failed reading the cortex label file."
    # set labels (n) and triangles (n x 3)
    labels = aparc[0]
    try:
        slabels = smooth_aparc(surf, labels, cortex)
        logger.info(f"Outputting fixed annot: {outaparcname}")
        fs.write_annot(outaparcname, slabels, aparc[1], aparc[2])
    except Exception as e:
        logger.exception(e)
        return "Failed smoothing!"
    return 0


if __name__ == "__main__":
    from logging import INFO, basicConfig
    basicConfig(
        level=INFO,
        format="%(filename)s[%(lineno)d] %(levelname)s: %(message)s",
    )
    # Command Line options are error checking done here
    options = options_parse()

    sys.exit(main(options.insurf, options.inaparc, options.incort, options.outaparc))
