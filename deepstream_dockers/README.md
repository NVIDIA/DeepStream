# DeepStream 9.1 Open Source Dockerfiles Guide

The documentation here is intended to help customers build the Open Source DeepStream Dockerfiles.

This information is useful for both x86 systems with dGPU setup and NVIDIA Jetson Thor/Orin only devices.

Improvements from previous releases.  
(A) Building Jetson Thor/Orin dockers on x86 Linux PCs (cross-compile on x86).  
(B) Build setup files to put all of the files in correct location for the various build x86 and x86 crosss-compile (for Jetson).  

NOTE: Jetson Thor/Orin support only. Both x86 and Jetson Thor/Orin uses CUDA 13.2.


## 1 Additional Installation to use all DeepStreamSDK Features within the docker container.

Since DS 6.3, deepStream docker containers do NOT package libraries necessary for certain multimedia operations like audio data parsing, CPU decode, and CPU encode. This change could affect processing certain video streams/files like mp4 that include audio tracks.

Please run the below script inside the docker images to install additional packages that might be necessary to use all of the DeepStreamSDK features :

```
/opt/nvidia/deepstream/deepstream/user_additional_install.sh
```

## 1.1 Triton samples additional libraries required.

For Triton samples, while running /opt/nvidia/deepstream/deepstream-9.1/samples/prepare_classification_test_video.sh, FFMPEG package along with additional dependent libs need to be installed using command below. For additional information please refer to section 1.4 (for codecs: DIFFERENCES WITH  DEEPSTREAM 6.1 AND ABOVE) & section 1.5  for BREAKING CHANGES in Release notes.


```
apt-get install --reinstall libflac8 libmp3lame0 libxvidcore4 ffmpeg
```


## 1.2 Libraries and Packages needed for docker builds.

There is a need for two directories to handle these libraries and packages for x86 and Jetson.

export ADDVAR99=\<your x86 and Jetson content directory\>

```
mkdir -p $ADDVAR99/x86/gst
mkdir -p $ADDVAR99/x86/optel
mkdir -p $ADDVAR99/jetson/gst
mkdir -p $ADDVAR99/jetson/optel
```


## 1.2.1 Prebuilt libgstrtpmanager.so, libgstrtsp.so and libgstvideoparsersbad.so found in the NGC dockers.

These files are found in the DS 9.1 dockers found on NGC.

Look at nvcr.io/nvidia/deepstream:9.1-triton-multiarch

To include that in these local builds you will need to download these files from the DS 9.1 NGC dockers. On both Jetson and x86_64 dockers they are found in the following location.
/tmp99/libgstrtpmanager.so
/tmp99/libgstrtsp.so
/tmp99/libgstvideoparsersbad.so

The docker cp is one method to get those libraries.

For x86/Jetson dockers either one will work (nvcr.io/nvidia/deepstream:9.1-triton-multiarch or nvcr.io/nvidia/deepstream:9.1-samples-multiarch ). The library files are stored in the same path for either docker. 

NOTE: See the docker cp command documentation for more details.

sudo docker cp container_name:/path/inside/container/file.txt /path/to/host/

For example ...

With the x86 docker running ...

sudo docker cp {container_name_x86}:/tmp99/libgstrtpmanager.so $ADDVAR99/x86/gst

With the Jetson docker running ...

sudo docker cp {container_name_jetson}:/tmp99/libgstrtpmanager.so $ADDVAR99/jetson/gst



This will need to be added to open the Dockerfiles if you wish to include it.

Alternatively you can use the ``/opt/nvidia/deepstream/deepstream/update_rtpmanager.sh`` script (included in the deepstream sdk) and build the library directly.

So when users run ``/opt/nvidia/deepstream/deepstream/user_additional_install.sh`` script (on the docker) libgstrtpmanager.so will be copied to the correct location.

For x86 copy x86 gst libraries into $ADDVAR99/x86/gst

For Jetson Thor copy jetson gst libraries into $ADDVAR99/jetson/gst

### 1.2.2 Packages for Open Telemetry

This may require updates to the particular packages if there are changes in location or versions. This is a helper script to download the packages for reference.

``get_optel_pkgs.sh`` is a script to help download these packages. These packages will be downloaded for both x86 and Jetson ($ADDVAR99/x86/optel and $ADDVAR99/jetson/optel).


## 1.3 Jetson Dockers (adding to the released DS 9.1 NGC Jetson dockers)

For the Jetson NGC dockers you will need to add the following lines if you are making modifications to those prebuilt NGC DS 9.1 Jetson dockers.

```
RUN apt-key adv --fetch-keys https://repo.download.nvidia.com/jetson/jetson-ota-public.asc   
RUN echo "deb https://repo.download.nvidia.com/jetson/common r39.2 main" >> /etc/apt/sources.list      
```

## 1.4 cuda-keyring_1.1-1_all.deb

This might be required to be installed if not already installed on the machine.

See [Cuda Install Guide for Linux](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-installation-guide-linux)  


## 2 Building x86 DS docker images

## 2.1 x86 Docker Pre-requisites

Please refer to the Prerequisites section at DeepStream NGC page [NVIDIA NGC](https://ngc.nvidia.com/catalog/containers/nvidia:deepstream) to build and run deepstream containers.


### 2.1.1 Prerequisites; Mandatory; (DeepStreamSDK package and terminology)

1) Please download the [DeepStreamSDK release](https://github.com/NVIDIA/DeepStream/releases/tag/v9.1.0) x86 tarball and place it locally
in the ``$ROOT/`` folder of this repository.

``cp deepstream_sdk_v9.1.0_x86_64.tbz2 ./x86_dockerfiles/``
 

#### 2.1.2 CuDNN 9.20.0 install 

For x86-samples docker only

#### 2.1.2.1  x86 docker pre-requisites

Requires Docker version 29 or later.

#### 2.1.2.1.1  x86-Triton

Nothing to be added other than deepstream package.

#### 2.1.2.2 For x86 samples docker the TensorRT 10.16.0 and cuDNN 9.20.0 install is required for the Docker build

Download file link: [nv-tensorrt-local-repo-ubuntu2404-10.16.0-cuda-13.2_1.0-1_amd64.deb](https://developer.download.nvidia.com/compute/tensorrt/10.16.0/local_installers/nv-tensorrt-local-repo-ubuntu2404-10.16.0-cuda-13.2_1.0-1_amd64.deb) from TensorRT download page.
Note: You may have to login to [developer.nvidia.com](https://developer.nvidia.com/) to download the file.  
Quick Steps:  
$ROOT is the root directory of this git repo.    
``cd $ROOT/``

``cp nv-tensorrt-local-repo-ubuntu2404-10.16.0-cuda-13.2_1.0-1_amd64.deb ./x86_dockerfiles/``

Also the CuDNN file.

Download file link: [cudnn-local-repo-ubuntu2404-9.20.0_1.0-1_amd64.deb](https://developer.download.nvidia.com/compute/cudnn/9.20.0/local_installers/cudnn-local-repo-ubuntu2404-9.20.0_1.0-1_amd64.deb) from cuDNN download page.
Note: You may have to login to [developer.nvidia.com](https://developer.nvidia.com/) to download the file.
Quick Steps:
$ROOT is the root directory of this git repo.
``cd $ROOT/``

``cp cudnn-local-repo-ubuntu2404-9.20.0_1.0-1_amd64.deb ./x86_dockerfiles/``

#### 2.1.2.3 Important Notes on docker image size optimization

Hosting files on server (e.g. https://\<host server\>) is another alternative.

### 2.1.3 x86 Build setup Command

```
cd $ROOT/
./setup_x86_build.sh

```

## 2.2 Building specific x86 docker types

### 2.2.1 Instructions for Building x86 DS Triton docker image

NOTE: Make sure you run the x86 Build setup command first.

```
cd $ROOT/x86_dockerfiles
sudo docker build --network host --progress=plain --build-arg DS_DIR=/opt/nvidia/deepstream/deepstream-9.1 -t deepstream:9.1.0-triton-local -f Dockerfile_triton_x86 ..

```
NOTE: There is an example build script called $ROOT/buildx86.sh with the same contents.
  
### 2.2.2 Instructions for Building x86 DS samples docker image

NOTE: Make sure you run the x86 Build setup command first.

```
cd $ROOT/x86_dockerfiles
sudo docker build --network host --progress=plain -t deepstream:9.1.0-samples-local -f Dockerfile_samples_x86 ..

```


## 3 Building Jetson DS docker images


### 3.1 Pre-requisites

Must be built on a x86 Linux machine.

Please refer to the Prerequisites section at DeepStream NGC page [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream) to run deepstream containers.

Download DeepStreamSDK tarball from [DeepStreamSDK release](https://github.com/NVIDIA/DeepStream/releases/tag/v9.1.0) Jetson tarball and place it locally
in the ``$ROOT/`` folder of this repository.

``cp deepstream_sdk_v9.1.0_jetson.tbz2 ./jetson_dockerfiles/ ``

## 3.1.1 Jetson requires JP 7.2 to run the dockers on Jetson

More information found here [JetPack 7.2 GA](https://developer.nvidia.com/embedded/jetpack).

## 3.1.2 Jetson build setup for x86 cross compile

```
cd $ROOT/
ls $ADDVAR99/jetson/ #To verify $ADDVAR99 env variable is set correctly & required files are present.
./setup_x86_cross_compile_jetson.sh  
./setup_jetson_build.sh
```

> **NOTE:** `setup_jetson_build.sh` copies the GST `.so` files and OpenTelemetry packages from
> `$ADDVAR99/jetson/gst/` and `$ADDVAR99/jetson/optel/` into `jetson_dockerfiles/` — the Docker
> build context. The Dockerfiles use `--mount=type=bind,src=jetson_dockerfiles,target=/tmp/docker`,
> so **this step is mandatory and must be run before `docker build`**. Skipping it causes:
> ```text
> /usr/bin/cp: cannot stat '/tmp/docker/libgstrtpmanager.so': No such file or directory
> ```

## 3.2 Building specific Jetson docker types

### 3.2.1 Instructions for building Jetson DS triton docker image

NOTE: Make sure you run the Jetson setup (x86 cross-compile) and Build setup command first.

```
cd $ROOT/jetson_dockerfiles  
sudo docker build --platform linux/arm64 --network host --progress=plain -t deepstream-l4t:9.1.0-triton-local -f Dockerfile_Jetson_Devel ..

```

NOTE: There is an issue with the Jetson triton docker where for this example deepstream-infer-tensor-meta-test. The issue is related to libopencv package not being installed with libopencv-dev.  The solution is to install libopencv package directly (e.g. apt install libopencv) in addition to all of the other packages listed in the deepstream-infer-tensor-meta-test README file.


### 3.2.2 Instructions for building Jetson DS samples docker image

NOTE: Make sure you run the Jetson setup (x86 cross-compile) and Build setup command first.


#### 3.2.2.1 For Jetson samples docker the TensorRT 10.16.0 install is required for the Docker build

Download file link: [nv-tensorrt-local-repo-ubuntu2404-10.16.0-cuda-13.2_1.0-1_arm64.deb](https://developer.download.nvidia.com/compute/tensorrt/10.16.0/local_installers/nv-tensorrt-local-repo-ubuntu2404-10.16.0-cuda-13.2_1.0-1_arm64.deb) from TensorRT download page.
Note: You may have to login to [developer.nvidia.com](https://developer.nvidia.com/) to download the file.

Quick Steps:

$ROOT is the root directory of this git repo.
``cd $ROOT/``

``cp nv-tensorrt-local-repo-ubuntu2404-10.16.0-cuda-13.2_1.0-1_arm64.deb ./jetson_dockerfiles/``



```
cd $ROOT/jetson_dockerfiles  
sudo docker build --platform linux/arm64 --network host --progress=plain -t deepstream-l4t:9.1.0-samples-local -f Dockerfile_Jetson_Run ..

```
## 4 Triton Migration Guide

N/A for DS 9.1
