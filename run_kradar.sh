DIR=kradar
WORK=work_dirs
CONFIG="ra_img_echofusion_kradar_r50_trainval_24e"
bash tools/dist_train.sh projects/configs/$DIR/$CONFIG.py 1 --work-dir ./$WORK/$CONFIG/ 
#这里的8改成1 因为就只有一块卡
sleep 60               

# Evaluation
NUM=24
bash tools/dist_test.sh projects/configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/epoch_$NUM.pth 1 --eval bbox