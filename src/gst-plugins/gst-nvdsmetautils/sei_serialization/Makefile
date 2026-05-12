# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX:= g++

SRCS:= sei_serialization.cpp

INCS:= $(wildcard *.h)
LIB:=libnvds_sei_serialization.so

NVDS_VERSION?=9.0

CFLAGS+= -fPIC -DDS_VERSION=\"9.0.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../../includes

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/



LIBS += -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl -lpthread \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
	-lgstnvdsseimeta -Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.cpp=.o)
PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $(OBJ_DIR)/$(LIB) $(OBJS) $(LIBS)
	cp $(OBJ_DIR)/$(LIB) $@

install: $(LIB)
	ldconfig

clean:
	rm -rf $(OBJS) $(LIB)
