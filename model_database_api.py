from os import getcwd, walk
from os.path import isdir, isfile, join, splitext

import cv2
import numpy as np
from DLBio.helpers import MyDataFrame, load_json, to_uint8_image
from genericpath import isfile

from run_create_model_database import (BLOCK_INFO_NAME, DATABASE_PATH,
                                       META_NAME, TRAIN_OPT_NAME,
                                       create_end_stopping_image_box,
                                       get_example_batch)


def get_dataframe(model_folders):
    """Get the database values for a specific model as a Pandas dataframe.
    Each row one feature-map of one block with different metrics, which can 
    be none if this metric does not apply to the neuron.

    Location can be done with the following keys:
    model_type:
        if there are more than one model in your df. Note that this is 
        the overall architecture type, without the seed or number of blocks.
    block_id:
        Which type of block
    position:
        The database is essentially a list of blocks. This is the block list
        index
    depth:
        Like the block list index, but it counts the number of convolution as 
        proposed in the ResNet architecture.
    filter_idx: each block has a number of filters which is always the maximum
        computed in the block. For example an FP-block with an output dimension
        32 and q=2 has 64 filters. However, not all filters have a specific
        value. A pyrblock on the other hand would have 32 values in this
        example.
    [metric]-[layer/operation]: typical metric naming convention. E.g., 
        "act_per_fmap-bn2" is the activation per feature map after the second
        batch normalization.



    Parameters
    ----------
    model_folders : list of str
        list of model folders that are located in DATABASE_PATH (no absolute
        path).

    Returns
    -------
    pd.DataFrame

    """
    print(f'loading from {DATABASE_PATH.split("/")[-1]}')
    df = MyDataFrame(verbose=0)

    if not isinstance(model_folders, list):
        model_folders = [model_folders]

    for folder in model_folders:
        df = _update_dataframe(df, folder)

    return df.get_df()


def get_model_options(model_type):
    path = join(
        DATABASE_PATH,
        model_type,
        TRAIN_OPT_NAME
    )
    out = load_json(path)
    assert out is not None
    return out


def get_subtable(df, key_val_conditions):
    where = np.ones(df.shape[0], dtype=bool)
    for key, val in key_val_conditions.items():
        tmp = df[key] == val
        where = np.logical_and(where, tmp)

    return df[where]


def get_activations(row, operation, get_all_activations=False):
    """Returns activation feature map for the specifc operation of the neuron 
    defined by row

    Parameters
    ----------
    row : pandas Sequence (or dict)
        row corresponding to a neuron
    operation : str
        each 'neuron' has a series of operations. Define which one
    get_all_activations : bool, optional
        Return all feature maps, by default False

    Returns
    -------
    np.array of shape (N, H, W) or (N, D, H, W)
        activations of N randomly drawn images (usually five).

    Raises
    ------
    ValueError
        if the activation file cannot be found.
    """
    path = join(
        DATABASE_PATH,
        row["db_model_folder"],
        f'{str(int(row["position"])).zfill(3)}-{row["block_id"]}',
        f'{operation}-activation_examples.npy',
    )
    if not isfile(path):
        raise ValueError(f'No file found: {path}. CWD: {getcwd()}')
    activations = np.load(path)
    if get_all_activations:
        return activations

    activations = activations[..., row["filter_idx"]]
    return activations


def get_deg_of_es_activation(row, operation, get_all_activations=False):
    """Returns the activation of the neuron specified by row for the specific
    operation for the degree of end-stopping input image.

    Parameters
    ----------
    row : pd.Sequence
        row of the database representing one neuron
    operation : str
        which operation of the neuron
    get_all_activations : bool, optional
        return all feature maps, by default False

    Returns
    -------
    np.array (2,H,W)
        returns the activations for the positive and negative image
    """
    path = join(
        DATABASE_PATH,
        row["db_model_folder"],
        f'{str(int(row["position"])).zfill(3)}-{row["block_id"]}',
        f'{operation}-deg_of_es_activations.npy',
    )
    # has shape (2, D, H, W). 2 because pos and neg image activation
    activations = np.load(path)

    if get_all_activations:
        return activations

    # possible issues with q=2
    if activations.shape[1] <= row["filter_idx"]:
        print('shapes do not fit')
        print(activations.shape[1], row["filter_idx"])
        return None

    activations = activations[:, row["filter_idx"], ...]
    return activations


def get_activation_examples_input(is_cifar):
    return get_example_batch(is_cifar, as_numpy=True)


def get_degree_of_es_input(sign, is_cifar):
    return create_end_stopping_image_box(sign, is_cifar=is_cifar)[0]


def get_filter_pair(row, get_all_filters=False):
    path = join(
        DATABASE_PATH,
        row["model_type"],
        f'{str(int(row["position"])).zfill(3)}-{row["block_id"]}',
        f'filters.npy',
    )

    activations = np.load(path)
    if get_all_filters:
        return activations

    activations = activations[:, row["filter_idx"], ...]
    return activations


def get_upper(row, get_all_filters=True):
    path = join(
        DATABASE_PATH,
        row["model_type"],
        f'{str(int(row["position"])).zfill(3)}-{row["block_id"]}',
        f'lin_comb.npy',
    )

    activations = np.load(path)
    if get_all_filters:
        return activations

    activations = activations[:, row["filter_idx"]]
    return activations


def get_bn(row, get_all_filters=True):
    path = join(
        DATABASE_PATH,
        row["model_type"],
        f'{str(int(row["position"])).zfill(3)}-{row["block_id"]}',
        f'bn.npy',
    )

    activations = np.load(path)
    if get_all_filters:
        return activations
    return activations


def get_or_vector(filt):
    k = filt.shape[-1]
    xx, yy = np.meshgrid(np.linspace(-1., 1., k), np.linspace(-1., 1., k))
    X = np.stack([xx, yy], 0)
    X = X / (np.linalg.norm(X, 2, 0) + 1e-6)
    dx = X[0, ...][np.newaxis, ...]
    dy = X[1, ...][np.newaxis, ...]

    # orientation vector = linear combination of weights and dir vectors
    x1 = (filt * dx).sum(-1).sum(-1)
    y1 = (filt * dy).sum(-1).sum(-1)
    return np.array([x1, y1])


def get_keys(model_type):
    parent_folder = join(DATABASE_PATH, model_type)

    npy_files = []
    meta_keys = []

    for root, _, files_ in walk(parent_folder):
        if not files_:
            continue

        npy_files += [
            splitext(x)[0] for x in files_ if splitext(x)[1] == '.npy'
        ]

        if META_NAME in files_:
            meta = load_json(join(root, META_NAME))
            meta_keys += list(meta.keys())

    npy_files = list(set(npy_files))
    meta_keys = list(set(meta_keys))

    return meta_keys, npy_files


def get_deg_of_es_index(x):
    x = x.replace('_', '')
    return ['pos', 'neg'].index(x)


def get_degree_of_es_analysis(df, op_keys):
    # TODO is this only used for the first block dataframe?

    assert isinstance(op_keys, list)

    EPS = .1  # avoid division by zero and too high values
    SILENT_THRES = .1  # class neuron as silent when iD-sum < as this value
    ZERO_D_THRES = .1  # class neuron as silent when 01-ratio < as this value
    DEG_VAL_SILENT = -1.  # where to put class in a bar-plot
    DEG_VAL_0D = -.9  # where to put class in a bar-plot
    RATIO_12_CLIP_MIN = -.5  # clip minimal value of this metric
    RATIO_01_CLIP_MIN = -1.  # clip minimal value of this metric

    def comp_ratio_12(df, key, post='', eps=EPS, change_other_classes=True):
        id1 = df[key + '-id1' + post]
        id2 = df[key + '-id2' + post]
        out = np.array(1. - (id1 / (id2 + eps))).clip(min=RATIO_12_CLIP_MIN)

        if change_other_classes:
            cls = classifiy(df, key, post=post)
            out[cls == 'silent'] = DEG_VAL_SILENT
            out[cls == '0D'] = DEG_VAL_0D
        return out

    def comp_ratio_01(df, key, post='', eps=EPS):
        id0 = df[key + '-id0' + post]
        id1 = df[key + '-id1' + post]

        return np.array(1. - (id0 / (id1 + eps))).clip(min=RATIO_01_CLIP_MIN)

    def classifiy(df, key, post='', id1_t=ZERO_D_THRES):
        id2 = df[key + '-id2' + post]
        id0 = df[key + '-id0' + post]
        id1 = df[key + '-id1' + post]

        _sum = id2 + id1 + id0
        ratio = comp_ratio_01(df, key, post=post)

        out = np.zeros(ratio.shape).astype('str')
        out[:] = '1D & 2D'
        out[_sum < SILENT_THRES] = 'silent'
        out[ratio < id1_t] = '0D'

        return out

    # NOTE: some operations, e.g., lower in FP-block when q=2 have a lot of
    # nan values
    for key in op_keys:
        # add activations for negative and positive image
        df[key + '-id0'] = df[key + '-id0_neg'] + df[key + '-id0_pos']
        df[key + '-id1'] = df[key + '-id1_neg'] + df[key + '-id1_pos']
        df[key + '-id2'] = df[key + '-id2_neg'] + df[key + '-id2_pos']

        # remove silent and 0D signals
        df[key + '-id2_0'] = classifiy(df, key, post='')
        df[key + '-id2_0_pos'] = classifiy(df, key, post='_pos')
        df[key + '-id2_0_neg'] = classifiy(df, key, post='_neg')

        # compute the ratio between 1D and 2D
        df[key + '-id_ratio'] = comp_ratio_12(df, key)
        df[key + '-id_ratio_pos'] = comp_ratio_12(df, key, post='_pos')
        df[key + '-id_ratio_neg'] = comp_ratio_12(df, key, post='_neg')

    return df

# ---------------------------------------------------------------------------
# ---------- FUNCTIONS THAT COMPUTE NEW METRICS -----------------------------
# ---------------------------------------------------------------------------


def _update_dataframe(df, model_folder):

    parent_folder = join(DATABASE_PATH, model_folder)
    assert isdir(parent_folder), parent_folder
    files_found = False
    for root, _, files_ in walk(parent_folder):
        if not files_:
            continue

        if META_NAME not in files_:
            continue

        files_found = True

        meta = load_json(join(root, META_NAME))

        block_info = load_json(join(root, BLOCK_INFO_NAME))
        assert block_info is not None

        # get longest number of values
        # number can change, because q=2 in FP-net.
        N = max([len(list(x)) for x in meta.values()])

        # create one row
        for i in range(N):
            tmp = dict()
            #
            for key, value in meta.items():
                # q-factor != 1 creates different entropy numbers
                if i < len(value):
                    v = value[i]
                else:
                    v = np.nan
                tmp.update({key: v})

            # add block info to each entry
            tmp.update(block_info)
            tmp.update({'filter_idx': i, 'db_model_folder': model_folder})

            df.update(tmp, add_missing_values=True, missing_val=np.nan)

    assert files_found, f'No file in folder {parent_folder}'

    return df


def create_example_image(row, f=None, g=None):
    if f is None and g is None:
        tmp = get_filter_pair(row)
        print(tmp.shape)
        f = tmp[0, ...]
        g = tmp[1, ...]

    image = []

    # filter with arrow
    def fcn(x): return get_filter_image(x, get_min_padding(f, g))
    filter_image = image.append(
        get_row_image(fcn(f), fcn(g), zero_is_gray=True)
    )

    cv2.imwrite('test_0.png', image[-1])

    # fft image
    def fcn(x): return get_fft_image(x, 'abs')
    filter_image = image.append(
        get_row_image(fcn(f), fcn(g), zero_is_gray=False)
    )

    cv2.imwrite('test_1.png', image[-1])

    # filter outputs
    def fcn(x): return get_filtered_image(x, normalize=True)
    filter_image = image.append(
        get_row_image(fcn(f), fcn(g), zero_is_gray=False)
    )
    cv2.imwrite('test_2.png', image[-1])

    input_image = get_default_plus()
    input_image /= input_image.max()
    mult_image = get_multiplication_image(f, g)
    mult_image /= mult_image.max()
    image.append(
        get_row_image(
            input_image,
            mult_image,
            zero_is_gray=True
        ))
    cv2.imwrite('test_3.png', image[-1])

    image = np.concatenate(image, 0)

    return image


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
PADDING = 8
H = W = 512


def get_row_image(im_left, im_right, p=PADDING, zero_is_gray=False):
    # check resize necessary
    def maybe_resize(im):
        if im.shape[0] != H or im.shape[1] != W:
            im = cv2.resize(im, (H, W), interpolation=cv2.INTER_LINEAR)
        return im
    im_left = maybe_resize(im_left)
    im_right = maybe_resize(im_right)

    # min_val = min(im_left.min(), im_right.min())
    min_val = 0.  # min(im_left.min(), im_right.min())
    im_left = pad(im_left, ((0, p), (0, p)), min_val)
    im_right = pad(im_right, ((0, p), (0, 0)), min_val)

    out = np.concatenate([im_left, im_right], axis=1)
    if zero_is_gray:
        out = to_zero_is_gray_uint8(out)
    else:
        out = to_uint8_image(out)

    # print(out.shape)
    return out


def get_filter_image(filt, min_padding):
    v_orientation = compute_orientation_vector(
        filt, do_normalize=False, normed_filter=True)

    k = int(np.sqrt(filt.size))
    filt = pad(filt, ((k, k), (k, k)), min_padding)
    filt = cv2.resize(filt, (H, W), interpolation=cv2.INTER_NEAREST)

    filt -= min_padding
    max_val = filt.max()
    filt = filt / max_val

    filt = (255. * filt).astype('uint8')

    norm = np.linalg.norm(v_orientation)
    filt = draw_arrow(filt, v_orientation, norm=norm)

    filt = filt / 255.
    filt = filt * max_val
    filt = filt + min_padding

    return filt


def get_min_padding(f, g):
    z = np.concatenate([f.flatten(), g.flatten()])
    offset = np.concatenate([f.flatten(), g.flatten()]).std() * .1

    max_val = np.abs(z).max()
    min_padding = -(max_val + offset)
    return min_padding


def get_fft_image(filt, type_):
    if type_ == 'abs':
        out = to_fft(filt, return_abs=True)
    elif type_ == 'abs_squared':
        out = to_fft(filt, return_abs=True)**2.
    elif type_ == 'real':
        fft = to_fft(filt, return_abs=False)
        out = np.real(fft)
    elif type_ == 'imag':
        fft = to_fft(filt, return_abs=False)
        out = np.imag(fft)
    elif type_ == 'angle':
        fft = to_fft(filt, return_abs=False)
        out = np.arctan2(np.real(fft), np.imag(fft))
        out = out * 180 / np.pi
        out += 180.
        out /= 360.

    return out


def to_fft(arr, h=H, w=W, return_abs=True):
    h0, w0 = arr.shape
    pad_top = (h - h0) // 2
    pad_bottom = (h - h0) // 2 + (h - h0) % 2
    pad_left = (w - w0) // 2
    pad_right = (w - w0) // 2 + (w - w0) % 2

    arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)))

    fft = np.fft.fft2(arr)
    fft = np.fft.fftshift(fft)
    if return_abs:
        fft = np.abs(fft)

    return fft


def get_filtered_image(filt, normalize=True):
    plus = get_default_plus(False)
    plus = (plus - plus.mean()) / (plus.std() + 1e-3)

    out = cv2.filter2D(plus, -1, filt, borderType=cv2.BORDER_CONSTANT)
    #kernel = 1. / 9. * np.ones((3, 3))
    #out = cv2.filter2D(out, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    if normalize:
        out = (out - out.mean()) / (out.std() + 1e-3)

    out = cv2.resize(out, (H, W), interpolation=cv2.INTER_LINEAR)
    return out


def get_multiplication_image(f, g, normalize=True, use_relu=False, no_resize=False):
    plus = get_default_plus(False)[..., 0]
    plus = (plus - plus.mean()) / (plus.std() + 1e-3)
    #star = get_default_star()
    #star = (star - star.mean()) / (star.std() + 1e-3)

    left = cv2.filter2D(plus, -1, f, borderType=cv2.BORDER_CONSTANT)
    right = cv2.filter2D(plus, -1, g, borderType=cv2.BORDER_CONSTANT)

    if normalize:
        left = (left - left.mean()) / (left.std() + 1e-3)
        right = (right - right.mean()) / (right.std() + 1e-3)

    if use_relu:
        left = left.clip(min=0)
        right = right.clip(min=0)

    out = left * right

    if not no_resize:
        out = cv2.resize(out, (H, W), interpolation=cv2.INTER_LINEAR)
    return out


def compute_orientation_vector(filt, do_normalize=True, normed_filter=False):
    k = filt.shape[0]
    xx, yy = np.meshgrid(np.linspace(-1., 1., k), np.linspace(-1., 1., k))
    X = np.stack([xx, yy], 0)
    X = X / (np.linalg.norm(X, 2, 0) + 1e-6)
    filt = filt[np.newaxis, ...]
    if normed_filter:
        filt = filt / np.linalg.norm(filt.flatten())

    resulting_vector = (X * filt).sum(-1).sum(-1)
    if do_normalize:
        resulting_vector = resulting_vector / np.linalg.norm(resulting_vector)

    return resulting_vector


def pad(arr, pad_width, c):
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    return np.pad(arr, pad_width, mode='constant', constant_values=c)


def draw_arrow(image, dir_vector, length=70, min_len=30, norm=1):
    # Start coordinate, here (0, 0)
    # represents the top left corner of image
    start_point = np.array([image.shape[0] // 2, image.shape[1] // 2])

    # End coordinate
    end_point = (
        start_point + (min_len + norm * length) * dir_vector
    ).astype('int')

    # Line thickness of 9 px
    thickness = 6

    # Using cv2.arrowedLine() method
    # Draw a diagonal green arrow line
    # with thickness of 9 px
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.arrowedLine(
        image,
        tuple(start_point), tuple(end_point),
        (255, 255, 255), thickness
    )
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image


# TODO: make this nicer
X0 = 10
X1 = 32 - 10
X1 = 32 - 10
Y0 = 10
Y1 = 32 - 10
Y1 = 32 - 10
H0 = 3
W0 = 3

X0 = 10 * 7
X1 = 224 - 10 * 7
X1 = 224 - 10 * 7
Y0 = 10 * 7
Y1 = 224 - 10 * 7
Y1 = 224 - 10 * 7
H0 = 21
W0 = 21


class Rectangle():
    def __init__(self, **kwargs):
        # (y0, x0) & (y1, x1)
        self.x0 = kwargs['x0']
        self.x1 = kwargs['x1']
        self.y0 = kwargs['y0']
        self.y1 = kwargs['y1']


def get_default_plus(inverted=False):
    x0, y0, x1, y1 = X0, Y0, X1, Y1
    h, w = H0, W0

    h = h / 2.
    y_m = (y1 - y0) / 2. + y0
    ym0 = int(round(y_m - h))
    ym1 = int(round(y_m + h))
    horz = Rectangle(x0=x0, x1=x1, y0=ym0, y1=ym1)

    x_m = (x1 - x0) / 2. + x0
    w = w / 2.
    xm0 = int(round(x_m - w))
    xm1 = int(round(x_m + w))

    vert = Rectangle(x0=xm0, x1=xm1, y0=y0, y1=y1)

    out = draw([horz, vert])
    if inverted:
        out *= -1.

    out -= out.min()
    out /= out.max()

    return out


def to_zero_is_gray_uint8(image):
    b = 127
    y = np.abs(image).max()
    if y == 0:
        y = 1.
    m = 127 / y

    image = m * image + b
    return image.astype('uint8')


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


def _debug():
    # get_keys('fp_resnet_50_layer_start_q1')
    # get_dataframe(['fp_resnet_50_layer_start_q1'])
    _debug_filter_image()
    # get_dataframe(['fp_cifar10'])


def _debug_filter_image():
    f = np.array([[-1., -1., -1.], [0., 0., 0.], [+1., +1., +1.]])
    g = f.T.copy()
    create_example_image(None, f=f, g=g)


if __name__ == '__main__':
    _debug()
