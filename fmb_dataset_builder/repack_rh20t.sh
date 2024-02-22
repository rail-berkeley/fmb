#!/bin/bash

# Path to tar file
TAR_FILE=$1

# Extract archive
echo tar -xf ${TAR_FILE}.tar.gz

# delete scenes we don't want
echo rm -rf ${TAR_FILE}/*_scene_0006*
echo rm -rf ${TAR_FILE}/*_scene_0007*
echo rm -rf ${TAR_FILE}/*_scene_0008*
echo rm -rf ${TAR_FILE}/*_scene_0009*
echo rm -rf ${TAR_FILE}/*_scene_0010*

# Path to output tar file
OUTPUT_TAR=${TAR_FILE}.cleaned.tar

# Re-compress folder
echo tar -cvf $OUTPUT_TAR $TAR_FILE