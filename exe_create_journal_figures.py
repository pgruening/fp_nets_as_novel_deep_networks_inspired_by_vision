import re
from os.path import join, splitext

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from DLBio.helpers import search_in_all_subfolders
from pdf2image import convert_from_path

OUT_PATH = 'journal_figures'

FIGURES = [{
    'name': 'cifar_test',
    'figures': ['basic_block_min_val_er.pdf',
                'pyr_block_min_val_er.pdf'],
    'subplot': (1, 2)
},
    {
    'name': 'imagenet_result',
    'figures': ['imagenet_result_plot.pdf']
},
    {
    'name': 'gamma_bin',
    'figures': ['gamma0.png',
                'gamma20.png',
                'gamma45.png',
                'gamma90.png',
                'gamma135.png',
                'gamma160.png',
                ],
    'subplot': (2, 3)
},
    {
    'name': 'entropy',
    'figures': ['resnet50_imagenet_layer_start_q1_entropy_vs_angle.pdf',
                'CifarJOVFPNet_N9_s744_entropy_vs_angle.pdf'],
    'subplot': (1, 2)
},
    {
    'name': 'degofes',
    'figures': ['CifarPyrResNet_N9_s744_degree_of_es_relu.pdf',
                'CifarJOVFPNet_N9_s744_degree_of_es_mult.pdf',
                'resnet50_imagenet_layer_start_q1_degree_of_es_mult.pdf', ],
    'subplot': (1, 3)
},
    {
    'name': 'angles',
    'figures': ['resnet50_imagenet_layer_start_q1_angle_dist.pdf',
                'CifarJOVFPNet_N9_s744_angle_dist.pdf'],
    'subplot': (1, 2)
},
    {  # handmade ??
    'name': 'iso_response',
    'figures': ['ln.png', 'fp.png'],
    'subplot': (1, 2)
},
    {
    'name': 'advex_result',
    'figures': ['basic_block_num_changes_1.pdf',
                'pyr_block_num_changes_1.pdf'],
    'subplot': (1, 2)
},
    {
    'name': 'jpeg_result',
    'figures': ['basic_block_nc_1.pdf',
                'pyr_block_nc_1.pdf'],
    'subplot': (1, 2)
},
    {
    'name': 'perfect_filter',
    'figures': ['perfect_filter_example.png']
},
    {
    'name': 'degofe_weight_map',
    'figures': ['degofes_signal.png',
                'degofes_overlay.png'],
    'subplot': (1, 2)
},
    {
    'name': 'cifar_advex_example',
    'figures': ['eps_1.png',
                'eps_2.png',
                'eps_4.png',
                'eps_8.png',
                ],
    'subplot': (4, 1)
},
    {
    'name': 'cifar_compression_example',
    'figures': ['jpeg_example_0.png', 'jpeg_example_1.png'],
    'subplot': (2, 1)
},
]


def run():
    for fig in FIGURES:
        print(fig['name'])
        paths_ = []
        for image in fig['figures']:
            # search for the image in the entire repository
            path = search_in_all_subfolders(re.compile(image), '.')
            assert len(path) == 1, image
            paths_.append(path[0])

        sub_plot = fig.get('subplot', (1, 1))
        _, ax = plt.subplots(*sub_plot)
        _, width = sub_plot
        for index, path in enumerate(paths_):
            if splitext(path)[-1] == '.pdf':
                image = convert_from_path(path, 500)
                image = np.array(image[0])
            else:
                image = mpimg.imread(path)
            if len(ax.shape) == 1:
                ax[index].imshow(image)
                ax[index].axis('off')
            else:
                y, x = get_yx(index, width)
                ax[y, x].imshow(image)
                ax[y, x].axis('off')

        plt.savefig(join(OUT_PATH, fig['name'] + '.pdf'), bbox_inches='tight')
        plt.close()


def get_yx(index, width):
    return index // width, index % width


if __name__ == '__main__':
    run()
