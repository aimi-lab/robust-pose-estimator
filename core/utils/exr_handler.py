import OpenEXR
import Imath
import numpy as np


def load_exr(file_path:str=None) -> OpenEXR.InputFile:
    """
    Load a EXR file. Returns an OpenEXR.InputFile
    :param file_path: file path to pgm file
    :return: OpenEXR.InputFile
    """
    return OpenEXR.InputFile(file_path)


def save_exr(
            file_path:str=None,
            data:bytes=None,
            exr_header:OpenEXR.Header=None
            ) -> OpenEXR.OutputFile:
    """
    Save a EXR file. Returns an OpenEXR.OutputFile
    :param file_path: file path to exr file
    :param data: bytes-encoded array
    :return: OpenEXR.OutputFile
    """

    exr = OpenEXR.OutputFile(file_path, exr_header)
    exr.writePixels({'Y': data})
    exr.close()

    return exr


def exr2gry(exr:OpenEXR.InputFile=None) -> np.ndarray:
    """
    Convert a EXR file into a Numpy array. Note that it will have
    a shape of H x W, not W x H.
    :param exr: OpenEXR.InputFile
    :return: single channel 2-D Numpy array
    """

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    yf = np.frombuffer(exr.channel('Y', pt), dtype=np.float32)
    dw = exr.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    yf.shape = (sz[1], sz[0])

    return yf


def gry2exr(arr:np.ndarray=None, file_path:str=None):
    """
    Convert a single channel 2-D Numpy array to a EXR object.
    :param arr: spatial map
    :param file_path: file path as string
    :return: boolean
    """

    data = arr.astype(np.float32).tobytes()
    exr_header = OpenEXR.Header(arr.shape[1], arr.shape[0])
    exr_header['channels'] = {'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    exr = save_exr(file_path, data, exr_header)

    return exr