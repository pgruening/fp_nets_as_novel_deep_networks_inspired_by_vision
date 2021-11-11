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

if USER == 'philipp':
    DATA_FOLDER = '/data_ssd0/gruening/pytorch_datasets'
    PRINT_FREQUENCY = 50
    REPO_DIR = '/nfshome/gruening/my_code/DLBio_repos/fp_net_after_jov'

DO_NOT_COPY = [
    '/nfshome/gruening/my_code/DLBio_repos/fp_net_after_jov/test_cases']

if USER == 'david':
    DATA_FOLDER = '/data'  # mapped from '/nfshome/gmelin/Documents/Data' in docker_run.sh
    PRINT_FREQUENCY = 50
    # mapped from '/nfshome/gmelin/Documents/repos/fp_net_after_jov'
    REPO_DIR = '/workingdir'
    DO_NOT_COPY = [
        '/workingdir/test_cases']
