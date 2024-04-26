
for len in 3 6 12
do
for rate in 0.001 0.005 0.0001 0.0005
do
python -u main.py \
  --model_name 'PA2GCN'\
  --dp_mode False\
  --data_name 'METR-LA' \
  --num_gcn 10\
  --d_model 64\
  --lr $rate \
  --patch_len 3\
  --stride 1\
  --batch_size 32 \
  --pred_len $len\
  --resume_dir None\
  --output_dir None\
  --info None\

done
done

for len in 3 6 12
do
for rate in 0.001 0.005 0.0001 0.0005
do
python -u main.py \
  --model_name 'PA2GCN'\
  --dp_mode False\
  --data_name 'PeMS-Bay' \
  --num_gcn 10\
  --d_model 64\
  --lr $rate \
  --patch_len 3\
  --stride 1\
  --batch_size 32 \
  --pred_len $len\
  --resume_dir None\
  --output_dir None\
  --info None\

done
done

for len in 3 6 12
do
for rate in 0.001 0.005 0.0001 0.0005
do
python -u main.py \
  --model_name 'PA2GCN'\
  --dp_mode False\
  --data_name 'PEMS04' \
  --num_gcn 11\
  --d_model 64\
  --lr $rate \
  --patch_len 3\
  --stride 1\
  --batch_size 32 \
  --pred_len $len\
  --resume_dir None\
  --output_dir None\
  --info None\

done
done

for len in 3 6 12
do
for rate in 0.001 0.005 0.0001 0.0005
do
python -u main.py \
  --model_name 'PA2GCN'\
  --dp_mode False\
  --data_name 'PEMS08' \
  --num_gcn 10\
  --d_model 64\
  --lr $rate \
  --patch_len 3\
  --stride 1\
  --batch_size 32 \
  --pred_len $len\
  --resume_dir None\
  --output_dir None\
  --info None\

done
done
