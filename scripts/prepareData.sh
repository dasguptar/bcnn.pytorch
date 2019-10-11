#!/bin/bash

# move to data directory
ORIGIN=$(pwd)
mkdir -p data/; cd data/

if [ ! -d CUB_200_2011 ]; then
    if [ ! -f CUB_200_2011.tgz ]; then
        echo "==> DOWNLOADING THE DATASET ..."
        wget -c http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
    fi
    echo "==> EXTRACTING THE DATASET ..."
    tar -xzf CUB_200_2011.tgz
    if [ -f attributes.txt ]; then
        mv attributes.txt CUB_200_2011/
    fi
    # rm CUB_200_2011.tgz
fi

cd CUB_200_2011/
ROOT=$(pwd)

echo "==> GETTING DIRECTORY STRUCTURE ..."
cd images/; find . -type d > $ROOT/dirs.txt; cd $ROOT;

# join to get index, filename, class, split, bounding boxes
join images.txt image_class_labels.txt | join - train_test_split.txt | join - bounding_boxes.txt > $ROOT/combined_details.txt

function initDirectoryStructure(){
    mkdir -p $1; cd $1; echo $(pwd)
    xargs mkdir -p < $ROOT/dirs.txt;
    cd ../;
}

echo "==> COPYING ORIGINAL FILES OVER ..."
initDirectoryStructure original
cd images/;
find . -name "*jpg" | parallel -j0 --eta --bar cp {} ../original/{}
cd ../

# print out filename with bbox dimensions with convert commands
echo "==> CROPPING IMAGES TO BOUNDING BOXES ..."
initDirectoryStructure cropped
awk -F' ' '{print "convert images/"$2" -crop "$5+$7"x"$6+$8"+"$5"+"$6" +repage cropped/"$2}' $ROOT/combined_details.txt | parallel -j0 --eta --bar

# resize images so that smaller dimension becomes 512 px, maintaining aspect ratio
echo "==> RESIZING CROPPED IMAGES ..."
initDirectoryStructure resized_cropped
find cropped/ -name "*jpg" | parallel -j0 --eta --bar convert -resize "512^" {} resized_{}

# resize images so that smaller dimension becomes 512 px, maintaining aspect ratio
echo "==> RESIZING UNCROPPED IMAGES ..."
initDirectoryStructure resized_original
find original/ -name "*jpg" | parallel -j0 --eta --bar convert -resize "512^" {} resized_{}

echo "==> REMOVING UNRESIZED IMAGES"
rm -rf $ROOT/original/*
rm -rf $ROOT/cropped/*

function splitDirectory(){
    src=$1
    tgt=$2
    mkdir -p $tgt; cd $tgt
    initDirectoryStructure train
    initDirectoryStructure test
    cd ../
    awk -v s="$src" -v t="$tgt" -F' ' '$4 == 1 {print "cp "s"/"$2" "t"/train/"$2}' $ROOT/combined_details.txt | parallel -j0 --eta --bar #--dryrun
    awk -v s="$src" -v t="$tgt" -F' ' '$4 == 0 {print "cp "s"/"$2" "t"/test/"$2}' $ROOT/combined_details.txt | parallel -j0 --eta --bar #--dryrun
}

echo "==> SPLITTING RESIZED AND UNCROPPED IMAGES INTO TRAIN/TEST FOLDERS"
splitDirectory resized_original original
echo "==> SPLITTING RESIZED AND CROPPED IMAGES INTO TRAIN/TEST FOLDERS"
splitDirectory resized_cropped cropped

echo "==> Cleaning up intermediate stuff"
cd $ROOT
rm -rf dirs.txt combined_details.txt resized_cropped/ resized_original/

# move back to original directory
cd $ORIGIN
