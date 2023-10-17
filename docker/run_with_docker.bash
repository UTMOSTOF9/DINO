docker run --gpus all \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --env="DISPLAY"  \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --ipc=host --net=host \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v $PWD:/code \
    -it --rm bash