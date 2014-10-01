#!/usr/bin/env bash

PTB_DATA_URL=\
http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
OUTPUT_FILE=\
./simple-examples.tgz
OUTPUT_DIR=\
./simple-examples
DATA_SOURCE_DIR=\
${OUTPUT_DIR}/data
DATA_DEST_DIR=\
./ptb_data

if [ -d "${DATA_DEST_DIR}" ]; then
  echo "Skipping download and extraction; output directory already exists: ${DATA_DEST_DIR}"
  exit 0
fi

if [ -d "${DATA_SOURCE_DIR}" ]; then
  echo "Skipping download; data source dir already exists: ${DATA_SOURCE_DIR}"
else
  if [ -e "${OUTPUT_FILE}" ]; then
    echo "Skipping download; output file already exists: ${OUTPUT_FILE}"
  else
    OUTPUT_TMP_FILE="${OUTPUT_FILE}.tmp"
    wget ${PTB_DATA_URL} -O ${OUTPUT_TMP_FILE}
    if [ -e "${OUTPUT_TMP_FILE}" ]; then
      mv ${OUTPUT_TMP_FILE} ${OUTPUT_FILE}
      echo "Successfully downloaded data to: ${OUTPUT_FILE}"
    else
      echo "Download failed; exiting. :("
      exit 1
    fi
  fi

  tar -xzvf ${OUTPUT_FILE} || ( echo "tar extraction failed :("; exit 2 )
fi

if [ -d "${DATA_SOURCE_DIR}" ]; then
  echo "Successfully extracted ${OUTPUT_FILE} to: ${OUTPUT_DIR}"
  echo "Found data at: ${DATA_SOURCE_DIR}"
  echo "Moving data to: ${DATA_DEST_DIR}"
  mkdir -p ${DATA_DEST_DIR}
  mv ${DATA_SOURCE_DIR}/* ${DATA_DEST_DIR}/
  RM_FILES="${OUTPUT_FILE} ${OUTPUT_DIR}/"
  echo "Deleting downloaded data: ${RM_FILES}"
  rm -rf ${RM_FILES}
else
  echo "tar extraction succeeded, but data directory wasn't created: ${DATA_SOURCE_DIR}"
  exit 3
fi
