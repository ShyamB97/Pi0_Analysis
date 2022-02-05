"""
Created on: 26/01/2022 15:45

Author: Shyam Bhuller

Description: contais vector operations, based of awkward array records.
"""

import awkward as ak
import numpy as np

def vector(x, y, z):
    """Creates a vector like record

    Args:
        x (Any): x component
        y (Any): y component
        z (Any): z component

    Returns:
        ak.Record: record structurd like a 3-vector
    """
    return ak.zip({"x" : x, "y" : y, "z" : z})


def magntiude(vec):
    """magnitude of 3-vector

    Args:
        vec (ak.Record created by vector): vector

    Returns:
        ak.Array: array of magnitudes
    """
    return (vec.x**2 + vec.y**2 + vec.z**2)**0.5


def normalize(vec):
    """Normalize a vector (get direction)

    Args:
        vec (ak.Record created by vector): a vector

    Returns:
        ak.Record created by vector: norm of vector
    """
    m = magntiude(vec)
    return vector( vec.x / m, vec.y / m, vec.z / m )


def dot(a, b):
    """dot product of 3-vector

    Args:
        a (ak.Record created by vector): first vector
        b (ak.Record created by vector): second vector

    Returns:
        ak.Array: array of dot products
    """
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z)


def prod(s, v):
    """product of scalar and vector

    Args:
        s (ak.Array or single number): scalar
        v (ak.Record created by vector): vector

    Returns:
        ak.Record created by vector: s * v
    """
    return vector(s * v.x, s * v.y, s * v.z)


def angle(a, b):
    """Compute angle between two vectors

    Args:
        a (ak.Record created by vector): a vector
        b (ak.Record created by vector): another vector

    Returns:
        [ak.Array]: angle between a and b
    """
    return np.arccos(dot(a, b) / (magntiude(a) * magntiude(b)))
