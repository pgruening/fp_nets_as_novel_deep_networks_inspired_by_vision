from os.path import join
import matplotlib
import matplotlib.pyplot as plt
from DLBio.helpers import load_json

# name :(num param, val error)
RESULTS = {'ResNet-50 (baseline)': (25.5, 23.61)}

MODEL_1 = 'E-ResNet-50-R (q=0.8)'
MODEL_2 = 'E-ResNet-50-R (q=1)'

FOLDERS = {
    'E-ResNet-50-R (q=1)': 'experiments/imagenet_resnet/resnet50_imagenet_layer_start_q1',
    'E-ResNet-50-R (q=0.8)': 'experiments/imagenet_resnet/resnet50_imagenet_layer_start_q08'
}

# (name of the json file, key for the value of interest)
VAL_ERROR_NAME = ('validation_acc.json', 'error')
NUM_PARAM_NAME = ('model_specs.json', 'num_params')
# use millions instead of full number
NP_FACTOR = 10**-6

COLORS = {
    'E-ResNet-50-R (q=1)': 'g',
    'E-ResNet-50-R (q=0.8)': 'g',
    'ResNet-50 (baseline)': 'k'
}

MARKER = {
    'E-ResNet-50-R (q=1)': 'o',
    'E-ResNet-50-R (q=0.8)': 'd',
    'ResNet-50 (baseline)': 'd'
}


X_AXIS_LABEL = 'Number of Parameters (M)'
Y_AXIS_LABEL = 'Error (%)'
LINE_COLOR = 'g'
FONT_SIZE = 22
MARKER_SIZE = 20
LINE_WIDTH = 5
LINE_STYLE = '--'
MARKER_EDGE_COLOR = 'k'
MARKER_EDGE_WIDTH = 2

OUT_FOLDER = 'experiments/new_JOV_result_plots'
OUT_NAME = 'imagenet_result_plot'
EXTENSIONS = ['.pdf', '.png']


def run():
    for name, folder in FOLDERS.items():
        val = load_json(join(folder, VAL_ERROR_NAME[0]))
        num_params = load_json(join(folder, NUM_PARAM_NAME[0]))
        assert val is not None
        assert num_params is not None
        RESULTS[name] = (
            NP_FACTOR * num_params[NUM_PARAM_NAME[1]], val[VAL_ERROR_NAME[1]]
        )

    # set font size
    matplotlib.rc('font', size=FONT_SIZE)
    plt.figure(figsize=(12, 12))
    # create line between
    plt.plot(
        [RESULTS[MODEL_1][0], RESULTS[MODEL_2][0]],
        [RESULTS[MODEL_1][1], RESULTS[MODEL_2][1]],
        LINE_COLOR, linewidth=LINE_WIDTH, linestyle=LINE_STYLE
    )

    for name, (x, y) in RESULTS.items():
        plt.plot(
            x, y, COLORS[name] + MARKER[name],
            label=name, markersize=MARKER_SIZE,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH
        )

    plt.grid()
    plt.legend()
    plt.xlabel(X_AXIS_LABEL)
    plt.ylabel(Y_AXIS_LABEL)

    for ext in EXTENSIONS:
        plt.savefig(join(OUT_FOLDER, OUT_NAME + ext))

    plt.close()


if __name__ == '__main__':
    run()
