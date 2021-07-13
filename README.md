
# docker-compare-sfr

Get docker at [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

## Build the docker image
```
docker build -f .\Dockerfile . -t compare-sfr-image
```

## Create and run a detached container

```
docker run -d -it --name compare-sfr-container \

--mount type=bind,source="$(pwd)"/pipe3d,target=/home/sfr/pipe3d \

--mount type=bind,source="$(pwd)"/prospector,target=/home/sfr/prospector \

--mount type=bind,source="$(pwd)"/ppxf,target=/home/sfr/ppxf \

compare-sfr-image /bin/bash
```

## Enter the container with
```
docker exec -it compare-sfr-container /bin/bash
```

## Execute all the examples
```
./run_all_examples.sh
```

## To stop and remove the container
```
docker stop compare-sfr-container

docker rm compare-sfr-container
```

## To remove the image
```
docker rmi compare-sfr-image
```
