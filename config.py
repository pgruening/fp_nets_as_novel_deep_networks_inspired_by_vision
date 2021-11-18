"""
A file that we can use to put our individual paths into to avoid having them constantly pop us as changes in the files
The file should not be trackend constantly by git. To do this see:

https://stackoverflow.com/questions/13630849/git-difference-between-assume-unchanged-and-skip-worktree

git update-index --skip-worktree [FILE]

to change back: --no-skip-worktree

--skip-worktree is useful when you instruct git not to touch a specific file ever because developers 
should change it. For example, if the main repository upstream hosts some production-ready configuration
 files and you don’t want to accidentally commit changes to those files, --skip-worktree is exactly what you want.
"""

USER = 'philipp'

if USER == 'your_name':
    DATA_FOLDER = 'data'
    # see imagenet_sequential for more information
    IMAGENET_FOLDER_WITH_LMDB = '/imagenet_lmdb'
    IM_NET_TRAIN = {'link': '/imagenet_images_train'}
    IM_NET_VAL = {'link': '/imagenet_images_val'}
    IM_NET_ENTROPY_SUBSET = 'data/im_net_entropy/npy'
    PRINT_FREQUENCY = 50
    CIFAR_10_ENTROPY_SPLIT_FILE = 'data/reduced_valset.json'

elif USER == 'philipp':
    DATA_FOLDER = 'data'

    # Path to folder containing the files ILSVRC-train.lmdb & ILSVRC-val.lmdb
    IMAGENET_FOLDER_WITH_LMDB = '/nfshome/gruening/my_code'

    # Path to folder containing the ImageNet subolders with images
    IM_NET_TRAIN = {
        'original': '/nfshome/gruening/my_code/mount_II/hertel/imagenet/jpeg/train',
        'pc48': '/data_ssd1/hertel/imagenet/jpeg/train',
        'link': '/imagenet_images_train'
    }
    # from pc48
    IM_NET_VAL = {
        'original': '/nfshome/gruening/my_code/mount_II/hertel/imagenet/jpeg/validation',
        'pc48': '/data_ssd1/hertel/imagenet/jpeg/validation',
        'link': '/imagenet_images_val'
    }

    IMNET_LMDB_LINK = '/imagenet_lmdb'

    IM_NET_ENTROPY_SUBSET = 'data/im_net_entropy/npy'
    PRINT_FREQUENCY = 50
    CIFAR_10_ENTROPY_SPLIT_FILE = 'data/reduced_valset.json'


DO_NOT_COPY = ['test_cases']
