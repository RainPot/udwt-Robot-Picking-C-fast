cd ext/nms/
make
cd ../dcn
python3 setup.py build develop
cd ../../
mkdir data
cd data
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/2019origin.zip
unzip -q 2019origin.zip
cd ..
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/geo/hourglass.pth


#mv ./backbones/hourglass.py ./backbones/hourglass_old.py
#mv ./backbones/hourglass_flip.py ./backbones/hourglass.py
#hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/hourglass_UNDER_39000.pth
#mv ./hourglass_UNDER_39000.pth ./hourglass.pth
#mv scripts/RRNet/train.py ./
#mv scripts/RRNet/eval.py ./

mkdir log
mkdir results
mkdir deepsort_holothurian
mkdir deepsort_scallop
mkdir deepsort_echinus
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/original_ckpt.t7
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/dla34-ba72cf86.pth
#mv ./dla34-ba72cf86.pth /root/.cache/torch/checkpoints/dla34-ba72cf86.pth
#cd log
#hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/undersync1block/ckp-49999.pth
#hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/undersync1block/ckp-44999.pth
#hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/undersync1block/ckp-59999.pth

