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
    IMAGENET_LMDB = 'path_to_lmdb'
    IM_NET_TRAIN = {'pc48': 'your/path'}
    IM_NET_VAL = {'pc48': 'your/path'}
    IM_NET_ENTROPY_SUBSET = 'data/im_net_entropy/npy'
    PRINT_FREQUENCY = 50
    CIFAR_10_ENTROPY_SPLIT_FILE = 'data/reduced_valset.json'

elif USER == 'philipp':
    DATA_FOLDER = 'data'
    # see imagenet_sequential for more information
    IMAGENET_LMDB = 'path_to_lmdb'
    IM_NET_TRAIN = {'pc48': 'your/path'}
    IM_NET_VAL = {'pc48': 'your/path'}
    IM_NET_ENTROPY_SUBSET = 'data/im_net_entropy/npy'
    PRINT_FREQUENCY = 50
    CIFAR_10_ENTROPY_SPLIT_FILE = 'data/reduced_valset.json'


DO_NOT_COPY = ['test_cases']
