python -m penn.data.preprocess
python -m penn.train \
    --config config/crepe++-mdb.py \
    --datasets mdb \
    --gpu $1
python -m penn.train \
--config config/crepe++-ptdb.py \
--datasets ptdb \
--gpu $1



# using augmented data
python -m penn.train \
    --config config/crepe++.py \
    --gpu $1

python -m penn.train \
    --config config/fcnf0++.py \
    --gpu $1