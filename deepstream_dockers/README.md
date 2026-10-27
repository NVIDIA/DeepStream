# DeepStream 9.2.0 Open Source Dockerfiles Guide

# =========================
THIS IS A PR TEST CASE - update 2
# =========================


The documentation here is intended to help customers build the Open Source DeepStream Dockerfiles.

This information is useful for x86 systems with dGPU setup, NVIDIA Jetson Thor/Orin devices, and SBSA
(ARM server with dGPU) systems. The SBSA image also covers DGX Spark.

Improvements from previous releases.  
(A) Building Jetson Thor/Orin dockers on x86 Linux PCs (cross-compile on x86).  
(B) Build setup files to put all of the files in correct location for the various build x86 and x86 crosss-compile (for Jetson).  
(C) SBSA / DGX Spark Dockerfile and build setup (see section 5).  

NOTE: Jetson Thor/Orin support only. x86, Jetson Thor/Orin and SBSA all use CUDA 13.2.


## 1 Additional Installation to use all DeepStreamSDK Features within the docker container.

Since DS 6.3, deepStream docker containers do NOT package libraries necessary for certain multimedia operations like audio data parsing, CPU decode, and CPU encode. This change could affect processing certain video streams/files like mp4 that include audio tracks.

Please run the below script inside the docker images to install additional packages that might be necessary to use all of the DeepStreamSDK features :

```
/opt/nvidia/deepstream/deepstream/user_additional_install.sh
```

## 1.1 Triton samples additional libraries required.

For Triton samples, while running /opt/nvidia/deepstream/deepstream-9.2/samples/prepare_classification_test_video.sh, FFMPEG package along with additional dependent libs need to be installed using command below. For additional information please refer to section 1.4 (for codecs: DIFFERENCES WITH  DEEPSTREAM 6.1 AND ABOVE) & section 1.5  for BREAKING CHANGES in Release notes.


```
apt-get install --reinstall libflac8 libmp3lame0 libxvidcore4 ffmpeg
```


## 1.2 Libraries and Packages needed for docker builds.

There is a need for one directory per platform to handle these libraries and packages.

export ADDVAR99=\<your x86, Jetson and SBSA content directory\>

```
mkdir -p $ADDVAR99/x86/gst
mkdir -p $ADDVAR99/jetson/gst
mkdir -p $ADDVAR99/sbsa/gst
mkdir -p $ADDVAR99/sbsa/bsp
```

``$ADDVAR99/sbsa/bsp`` is only used by the SBSA build; see section 5.1.2.


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

With the SBSA docker running (nvcr.io/nvidia/deepstream:9.1-triton-sbsa-dgx-spark) ...

sudo docker cp {container_name_sbsa}:/tmp99/libgstrtpmanager.so $ADDVAR99/sbsa/gst



This will need to be added to open the Dockerfiles if you wish to include it.

Alternatively you can use the ``/opt/nvidia/deepstream/deepstream/update_rtpmanager.sh`` script (included in the deepstream sdk) and build the library directly.

So when users run ``/opt/nvidia/deepstream/deepstream/user_additional_install.sh`` script (on the docker) libgstrtpmanager.so will be copied to the correct location.

For x86 copy x86 gst libraries into $ADDVAR99/x86/gst

For Jetson Thor copy jetson gst libraries into $ADDVAR99/jetson/gst

For SBSA copy the aarch64 gst libraries into $ADDVAR99/sbsa/gst

### 1.2.2 Open Telemetry

Nothing to download. The Dockerfiles build opentelemetry-cpp from source at the tag
pinned by the ``OPENTELEMETRY_CPP_VERSION`` build arg, so a network connection is
required during the build (the documented ``docker build`` commands already pass
``--network host``).

Previously these packages were installed from prebuilt Debian ``.deb`` files staged in
``$ADDVAR99/{x86,jetson}/optel``; that is no longer needed, as those files are
periodically removed from the Debian archive pool.

### 1.2.3 OpenTelemetry build parallelism

The Dockerfiles build OpenTelemetry with all available cores by default. To limit
the number of parallel compile jobs, pass ``OPENTELEMETRY_BUILD_JOBS`` as a Docker
build argument. This can help when building an ARM64 image through emulation on an
x86 host:

```
sudo docker build --build-arg OPENTELEMETRY_BUILD_JOBS=2 ...
```

## 1.3 Jetson Dockers (adding to the released DS 9.2 NGC Jetson dockers)

For the Jetson NGC dockers you will need to add the following lines if you are making modifications to those prebuilt NGC DS 9.2 Jetson dockers.

```
RUN apt-key adv --fetch-keys https://repo.download.nvidia.com/jetson/jetson-ota-public.asc   
RUN echo "deb https://repo.download.nvidia.com/jetson/common r39.2 main" >> /etc/apt/sources.list      
```

## 1.4 cuda-keyring_1.1-1_all.deb

This might be required to be installed if not already installed on the machine.

See [Cuda Install Guide for Linux](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-installation-guide-linux)  


## 2 Building x86 DS docker images

## 2.1 x86 Docker Pre-requisites

Please refer to the [Prerequisites section of the DeepStream Docker Containers documentation](https://docs.nvidia.com/deepstream/9.2/text/DS_docker_containers#prerequisites) to build and run deepstream containers. Images are available on [GHCR](https://github.com/orgs/NVIDIA/packages/container/package/deepstream).


### 2.1.1 Prerequisites; Mandatory; (DeepStreamSDK package and terminology)

1) Please download the [DeepStreamSDK release](https://github.com/NVIDIA/DeepStream/releases/tag/v9.2.0) x86 tarball and place it locally
in the ``$ROOT/`` folder of this repository.

``cp deepstream_sdk_v9.2.0_x86_64.tbz2 ./x86_dockerfiles/``
 

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
sudo docker build --network host --progress=plain --build-arg DS_DIR=/opt/nvidia/deepstream/deepstream-9.2 -t deepstream:9.2.0-triton-local -f Dockerfile_triton_x86 ..

```
NOTE: There is an example build script called $ROOT/buildx86.sh with the same contents.
  
### 2.2.2 Instructions for Building x86 DS samples docker image

NOTE: Make sure you run the x86 Build setup command first.

```
cd $ROOT/x86_dockerfiles
sudo docker build --network host --progress=plain -t deepstream:9.2.0-samples-local -f Dockerfile_samples_x86 ..

```


## 3 Building Jetson DS docker images


### 3.1 Pre-requisites

Must be built on a x86 Linux machine.

Please refer to the [Prerequisites section of the DeepStream Docker Containers documentation](https://docs.nvidia.com/deepstream/9.2/text/DS_docker_containers#prerequisites) to run deepstream containers. Images are available on [GHCR](https://github.com/orgs/NVIDIA/packages/container/package/deepstream).

Download DeepStreamSDK tarball from [DeepStreamSDK release](https://github.com/NVIDIA/DeepStream/releases/tag/v9.2.0) Jetson tarball and place it locally
in the ``$ROOT/`` folder of this repository.

``cp deepstream_sdk_v9.2.0_jetson.tbz2 ./jetson_dockerfiles/ ``

## 3.1.1 Jetson requires JP 7.2.1 to run the dockers on Jetson

More information found here [JetPack 7.2.1](https://developer.nvidia.com/embedded/jetpack).

## 3.1.2 Jetson build setup for x86 cross compile

```
cd $ROOT/
ls $ADDVAR99/jetson/ #To verify $ADDVAR99 env variable is set correctly & required files are present.
./setup_x86_cross_compile_jetson.sh  
./setup_jetson_build.sh
```

> **NOTE:** `setup_jetson_build.sh` copies the GST `.so` files from
> `$ADDVAR99/jetson/gst/` into `jetson_dockerfiles/` — the Docker
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
sudo docker build --platform linux/arm64 --network host --progress=plain -t deepstream-l4t:9.2.0-triton-local -f Dockerfile_Jetson_Devel ..

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
sudo docker build --platform linux/arm64 --network host --progress=plain -t deepstream-l4t:9.2.0-samples-local -f Dockerfile_Jetson_Run ..

```
## 4 Triton Migration Guide

N/A for DS 9.2.0


## 5 Building SBSA DS docker images

SBSA covers ARM servers with a discrete GPU (GB200, Grace) and DGX Spark — the same image runs on all of them.

There is one SBSA image type, the Triton devel image, built from ``sbsa_dockerfiles/Dockerfile_SBSA_Triton``.

On Jetson the Tegra multimedia libraries are supplied by the host at run time through the NVIDIA container
runtime. An SBSA host has no Jetson BSP, so those libraries have to be **baked into the image**. The build
therefore needs the NVIDIA Jetson Linux Driver Package (BSP) in addition to the DeepStream SDK tarball.

### 5.1 Pre-requisites

Please refer to the [Prerequisites section of the DeepStream Docker Containers documentation](https://docs.nvidia.com/deepstream/9.2/text/DS_docker_containers#prerequisites) to run deepstream containers. Images are available on [GHCR](https://github.com/orgs/NVIDIA/packages/container/package/deepstream).

Requires Docker version 29 or later.

A native ARM64 host (SBSA / Grace / DGX Spark) is strongly recommended. Building on an x86 host works but
is slow, because opentelemetry-cpp and gRPC are compiled from source under emulation — see section 5.3.

$ROOT is the ``deepstream_dockers`` directory of this repository. Run all commands below from it.

The build needs three inputs staged by hand. Everything else (Triton base images, VPI, the NVIDIA
GStreamer plugins, Ubuntu packages, opentelemetry-cpp, gRPC) is downloaded during the build, so the build
host needs network access.

| Input | Where to get it | Where it goes |
| --- | --- | --- |
| Jetson Linux R39.2 Driver Package (BSP) | [Jetson Linux](https://developer.nvidia.com/embedded/jetson-linux) (section 5.1.2) | ``$ADDVAR99/sbsa/bsp/`` |
| GStreamer prebuilts (3 files) | the DS 9.1 SBSA DeepStream NGC docker (section 5.1.3) | ``$ADDVAR99/sbsa/gst/`` |
| ``deepstream_sdk_v9.2.0_sbsa.tbz2`` | [DeepStreamSDK release](https://github.com/NVIDIA/DeepStream/releases/tag/v9.2.0) (section 5.1.4) | ``$ROOT/sbsa_dockerfiles/`` |

### 5.1.1 Create the SBSA content directories

```
cd $ROOT/
export ADDVAR99=<your x86, Jetson and SBSA content directory>
mkdir -p $ADDVAR99/sbsa/gst $ADDVAR99/sbsa/bsp
```

### 5.1.2 Jetson Linux Driver Package (BSP)

Download the **Driver Package (BSP)** for Jetson Linux 39.2 from
[Jetson Linux](https://developer.nvidia.com/embedded/jetson-linux) into ``$ADDVAR99/sbsa/bsp/``:

```
wget -P $ADDVAR99/sbsa/bsp/ https://developer.nvidia.com/downloads/embedded/L4T/r39_Release_v2.0/release/Jetson_Linux_R39.2.0_aarch64.tbz2
```

Keep exactly one ``Jetson_Linux_R*_aarch64.tbz2`` in that directory; the setup script and the Dockerfile
both stop if they find none or more than one.

``setup_sbsa_build.sh`` copies the BSP into the build context, and the Dockerfile runs
``extract_packages.sh`` inside the container to unpack the Tegra libraries from it. The NVIDIA
GStreamer plugins are not part of the public BSP, so the Dockerfile downloads them from the public
Jetson apt repository, pinned to the BSP's package version:

| Source | Installed into the image |
| --- | --- |
| BSP ``nv_tegra/nvidia_drivers.tbz2`` | ``/usr/lib/aarch64-linux-gnu/tegra/`` |
| BSP ``nv_tegra/nvidia_l4t_multimedia_openrm.tbz2`` | ``/usr/lib/aarch64-linux-gnu/tegra/`` |
| apt ``nvidia-l4t-gstreamer`` (same version as the BSP) | ``/usr/lib/aarch64-linux-gnu/gstreamer-1.0/`` |

The OpenRM multimedia variant is the one that works against a discrete GPU. Keep the ``L4T_RELEASE``
build arg in the Dockerfile (default ``r39.2``) in step with the BSP you download, since the same
Jetson Linux release also supplies the GStreamer package.

### 5.1.3 GStreamer prebuilts

Copy ``libgstrtpmanager.so``, ``libgstrtsp.so`` and ``libgstvideoparsersbad.so`` out of the DS 9.1 SBSA
DeepStream NGC docker (see section 1.2.1 for what they are):

```
docker create --platform linux/arm64 --name ds_sbsa_gst nvcr.io/nvidia/deepstream:9.1-triton-sbsa-dgx-spark
for f in libgstrtpmanager.so libgstrtsp.so libgstvideoparsersbad.so; do
  docker cp ds_sbsa_gst:/tmp99/$f $ADDVAR99/sbsa/gst/
done
docker rm ds_sbsa_gst
file $ADDVAR99/sbsa/gst/*.so    # all three must report "ARM aarch64"
```

> **NOTE:** There is no DS 9.2 docker on NGC. The DS 9.1 prebuilts are Ubuntu 24.04 / GStreamer 1.24 builds,
> the same base as this image. Alternatively build them with ``/opt/nvidia/deepstream/deepstream/update_rtpmanager.sh``.

### 5.1.4 DeepStreamSDK package

Download the SBSA tarball from the [DeepStreamSDK release](https://github.com/NVIDIA/DeepStream/releases/tag/v9.2.0)
and place it in the build context:

```
cp deepstream_sdk_v9.2.0_sbsa.tbz2 ./sbsa_dockerfiles/
```

### 5.1.5 SBSA build setup command

Run without ``sudo``, so that ``ADDVAR99`` is kept:

```
cd $ROOT/
ls $ADDVAR99/sbsa/gst $ADDVAR99/sbsa/bsp #To verify $ADDVAR99 env variable is set correctly & required files are present.
./setup_sbsa_build.sh
ls sbsa_dockerfiles/
```

The script ends with ``SBSA build context staged in ./sbsa_dockerfiles/``, and ``sbsa_dockerfiles/`` then holds:

- ``Dockerfile_SBSA_Triton``, ``deepstream_sdk_v9.2.0_sbsa.tbz2`` and ``Jetson_Linux_R39.2.0_aarch64.tbz2``
- ``libgstrtpmanager.so``, ``libgstrtsp.so`` and ``libgstvideoparsersbad.so``
- ``extract_packages.sh``, ``gRPC_installation.sh``, ``entrypoint.sh`` and ``rsyslog``
- ``user_additional_install_devel.sh`` and ``user_deepstream_python_apps_install.sh``
- ``LicenseAgreement.pdf`` and ``LicenseAgreementContainer.pdf``

> **NOTE:** ``setup_sbsa_build.sh`` stages the specific-files, licenses, GStreamer prebuilts and the BSP
> tarball into ``sbsa_dockerfiles/`` — the Docker build context.
> The Dockerfile reads them through ``--mount=type=bind,src=sbsa_dockerfiles``, so **this step is
> mandatory and must be run before ``docker build``**.

### 5.2 Instructions for building the SBSA DS Triton docker image

NOTE: Make sure you run the SBSA Build setup command first.

```
cd $ROOT/sbsa_dockerfiles
sudo docker build --platform linux/arm64 --network host --progress=plain -t deepstream:9.2.0-triton-arm-sbsa-local -f Dockerfile_SBSA_Triton ..

```
NOTE: There is an example build script called $ROOT/buildsbsa.sh with the same contents.

Check that the image exists:

```
docker images deepstream:9.2.0-triton-arm-sbsa-local
```

### 5.3 Building the SBSA image on an x86 host

Register QEMU for arm64 and create a buildx builder, then stage the build context:

```
cd $ROOT/
./setup_x86_cross_compile_sbsa.sh
./setup_sbsa_build.sh
```

Then run the same ``docker build`` command as section 5.2. Expect a long build (several hours):
opentelemetry-cpp and gRPC are compiled from source and run under QEMU emulation.
``OPENTELEMETRY_BUILD_JOBS`` (section 1.2.3) is useful here to limit memory pressure.

> **NOTE:** If the aarch64 handler is already registered (``/proc/sys/fs/binfmt_misc/qemu-aarch64`` exists),
> ``setup_x86_cross_compile_sbsa.sh`` can be skipped. If you do run it, it selects the ``mysbsabuilder``
> buildx builder, whose results stay in the build cache: add ``--load`` to the ``docker build`` command,
> or switch back with ``docker buildx use default``, so the image shows up in ``docker images``.

### 5.4 Verifying the SBSA image

These checks need no GPU, so they also run on an x86 build host:

```
docker run --rm --platform linux/arm64 --entrypoint bash deepstream:9.2.0-triton-arm-sbsa-local -c '
  readlink -f lib/libnvbufsurface.so lib/libnvbufsurftransform.so
  dpkg -l libnvvpi4 vpi4-dev | tail -2
  ls /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvideo4linux2.so
  ldd lib/gst-plugins/libnvdsgst_infer.so | grep "not found"
  echo "AARCH64_DGPU=$AARCH64_DGPU"'
```

Expected:

- Both ``readlink`` lines resolve into ``/usr/lib/aarch64-linux-gnu/tegra/`` (``.so.1.0.0``), the SBSA builds.
- ``libnvvpi4`` and ``vpi4-dev`` are version ``4.1.3``.
- ``libgstnvvideo4linux2.so`` (``nvv4l2decoder`` / ``nvv4l2encoder``) is present.
- The only ``not found`` library is ``libcuda.so.1``, which the NVIDIA container runtime mounts from the
  host driver at run time.
- ``AARCH64_DGPU=1``.

### 5.5 Running the SBSA image

On the SBSA host (GB200 / Grace / DGX Spark) with the NVIDIA Container Toolkit installed:

```
sudo docker run -it --rm --runtime=nvidia --network=host -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics --gpus all --privileged -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix -v /etc/X11:/etc/X11 deepstream:9.2.0-triton-arm-sbsa-local
```

Inside the container, a quick sanity check:

```
deepstream-app --version-all
gst-inspect-1.0 nvv4l2decoder
```

To move an image built elsewhere onto the SBSA host, use ``docker save deepstream:9.2.0-triton-arm-sbsa-local | gzip > ds-sbsa.tar.gz``
and ``docker load < ds-sbsa.tar.gz``.

### 5.6 Notes on the SBSA image contents

The following differences from the x86 and Jetson images are deliberate:

- ``AARCH64_DGPU=1`` is set, which is how DeepStream distinguishes SBSA from Jetson at run time.
- The SBSA-aware ``libnvbufsurface``, ``libnvbufsurftransform`` and ``libnvcuvidv4l2`` shipped in the
  DeepStream SBSA SDK tarball replace the BSP copies, which do not understand SBSA context.
- VPI 4 is the discrete-GPU SBSA build that Isaac ROS rehosts publicly (``VPI_BASE_URL`` build arg,
  version and checksums pinned in the Dockerfile). The Jetson VPI packages link the Tegra multimedia
  stack and break the tracker on SBSA, so the build fails if ``libnvvpi.so.4`` links anything beyond libc.
- ``nvidia-l4t-gstreamer`` is unpacked from the public Jetson apt repository rather than installed, so
  that apt does not pull in the Tegra platform packages an SBSA container has no use for.
- ``libnvdsgst_udp.so`` (needs Rivermax) and the Tegra capture plugins ``libgstnvsiplsrc.so`` and
  ``libgstnvarguscamerasrc.so`` are removed, as they cannot work in an SBSA dGPU container.
