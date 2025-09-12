# Copyright (c) 2014-2015 The Regents of the University of California.
# All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS

# CMake commands to download and build VTK as an internal package
# that is part of the svMultiPhysics build.
#
# VTK is built without graphics.
#
# The GIT_TAG field specifies a specific VTK commit: 35fb2ed -> 9.40

include(ExternalProject)

ExternalProject_Add(
    vtk-internal-build
    GIT_REPOSITORY https://github.com/Kitware/VTK.git
    GIT_TAG 35fb2ed
    SOURCE_DIR "${CMAKE_BINARY_DIR}/vtk-source"
    BINARY_DIR "${CMAKE_BINARY_DIR}/vtk-build"
    INSTALL_DIR "${CMAKE_BINARY_DIR}/vtk-install"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/vtk-install
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DCMAKE_BUILD_TYPE:STRING=RELEASE
        -DBUILD_EXAMPLES=OFF
        -DBUILD_TESTING=OFF
        -DVTK_USE_SYSTEM_EXPAT:BOOL=OFF
        -DVTK_USE_SYSTEM_ZLIB:BOOL=OFF
        -DVTK_LEGACY_REMOVE=ON
        -DVTK_Group_Rendering=OFF
        -DVTK_Group_StandAlone=OFF
        -DVTK_RENDERING_BACKEND=None
        -DVTK_WRAP_PYTHON=OFF
        -DModule_vtkChartsCore=ON
        -DModule_vtkCommonCore=ON
        -DModule_vtkCommonDataModel=ON
        -DModule_vtkCommonExecutionModel=ON
        -DModule_vtkFiltersCore=ON
        -DModule_vtkFiltersFlowPaths=ON
        -DModule_vtkFiltersModeling=ON
        -DModule_vtkIOLegacy=ON
        -DModule_vtkIOXML=ON
        -DVTK_GROUP_ENABLE_Views=NO
        -DVTK_GROUP_ENABLE_Web=NO
        -DVTK_GROUP_ENABLE_Imaging=NO
        -DVTK_GROUP_ENABLE_Qt=DONT_WANT
        -DVTK_GROUP_ENABLE_Rendering=DONT_WANT
    UPDATE_COMMAND ""
    BUILD_COMMAND make -j6
)

ExternalProject_Get_Property(vtk-internal-build INSTALL_DIR)

set(VTK_INCLUDE "${INSTALL_DIR}/include/vtk-9.4")
set(VTK_BUILD_INSTALL_LIB "${INSTALL_DIR}/lib")

# On Centos the installed lib directory is called lib64.
if(NOT EXISTS "${VTK_BUILD_INSTALL_LIB}")
  set(VTK_BUILD_INSTALL_LIB "${INSTALL_DIR}/lib64")
endif()

# This sets the VTK_LIBRARIES variable.
find_package(VTK CONFIG PATHS ${INSTALL_DIR})

