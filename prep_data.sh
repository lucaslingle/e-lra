set -e;
mkdir lra_data;
cd lra_data;

# based on https://github.com/state-spaces/s4/blob/main/src/dataloaders/README.md
curl https://storage.googleapis.com/long-range-arena/lra_release.gz \
    --output lra_release.tar.gz;
tar xvf lra_release.tar.gz;
mv lra_release/lra_release/listops-1000 listops;
mv lra_release/lra_release/tsv_data retrieval;
mkdir pathfinder;
mv lra_release/lra_release/pathfinder* pathfinder;
rm -rf lra_release;

# based on https://github.com/google-research/long-range-arena/pull/47/files
curl https://storage.googleapis.com/long-range-arena/pathfinder_tfds.gz \
    --output pathfinder_tfds.tar.gz
tar xvf pathfinder_tfds.tar.gz;
mv TFDS pathfinder_tfds;

cd ..;
