# Execute this script to access the Docker-Container.
# use another line of -v to add other volumes.

# read only: path to imagenet lmdb files
# -v /nfshome/gruening/my_code/imagenet_lmdb:/imagenet_lmdb:ro \

# read only: path to imagenet image folder train and validation
# -v /nfshome/gruening/my_code/mount_II/hertel/imagenet/jpeg/train:/imagenet_images_train:ro \
# -v /nfshome/gruening/my_code/mount_II/hertel/imagenet/jpeg/validation:/imagenet_images_val:ro \
docker run -it \
    --gpus all \
    --name dl \
    --rm -v $(pwd):/workingdir \
    -v /nfshome/gruening/my_code/imagenet_lmdb:/imagenet_lmdb:ro \
    --user $(id -u):$(id -g) \
    dl_workingdir bash
