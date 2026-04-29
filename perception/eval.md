# Tinycheckpoints
x1            mae=0.001835  rmse=0.004437
y1            mae=0.000732  rmse=0.001152
z1            mae=0.000228  rmse=0.000337
box_base_yaw  mae=5.359 deg rmse=17.818 deg
lid_angle     mae=4.878 deg rmse=6.455 deg
flap_angle    mae=7.490 deg rmse=10.502 deg
lid_length    mae=0.000910  rmse=0.001348

# 10000dataset checkpoints
label                  mae          rmse         extra
x1                   0.001253   0.001563  
y1                   0.000535   0.000678  
z1                   0.000180   0.000231  
box_base_yaw         0.088700   0.223445  mae_deg=5.082 rmse_deg=12.802
lid_angle            0.096657   0.119708  mae_deg=5.538 rmse_deg=6.859
flap_angle           0.069860   0.113337  mae_deg=4.003 rmse_deg=6.494
lid_length           0.000719   0.000921  

# 10000dataset checkpoints epoch 200 lr 1e-3
label                  mae          rmse         extra
x1                   0.001094   0.001386  
y1                   0.000744   0.000907  
z1                   0.000208   0.000263  
box_base_yaw         0.056358   0.224827  mae_deg=3.229 rmse_deg=12.882
lid_angle            0.078746   0.099152  mae_deg=4.512 rmse_deg=5.681
flap_angle           0.058469   0.103557  mae_deg=3.350 rmse_deg=5.933
lid_length           0.000841   0.001061  


# 10000dataset checkpoints lr 3e-4
label                  mae          rmse         extra
x1                   0.001112   0.001410  
y1                   0.000496   0.000643  
z1                   0.000146   0.000197  
box_base_yaw         0.059218   0.215150  mae_deg=3.393 rmse_deg=12.327
lid_angle            0.062554   0.078910  mae_deg=3.584 rmse_deg=4.521
flap_angle           0.058055   0.084539  mae_deg=3.326 rmse_deg=4.844
lid_length           0.000590   0.000791  

# 10000dataset checkpoints lr 3e-4 sincos
label                  mae          rmse         extra
x1                   0.001237   0.001678  
y1                   0.000612   0.000782  
z1                   0.000211   0.000268  
box_base_yaw         0.013231   0.017824  mae_deg=0.758 rmse_deg=1.021
lid_angle            0.024562   0.032280  mae_deg=1.407 rmse_deg=1.849
flap_angle           0.048503   0.086037  mae_deg=2.779 rmse_deg=4.930
lid_length           0.000845   0.001070 

# 100kdataset checkpoints lr 3e-4sincos 20 epoch
label                  mae          rmse         extra
x1                   0.001224   0.001583  
y1                   0.000518   0.000685  
z1                   0.000196   0.000251  
box_base_yaw         0.013636   0.017510  mae_deg=0.781 rmse_deg=1.003
lid_angle            0.024008   0.030790  mae_deg=1.376 rmse_deg=1.764
flap_angle           0.056843   0.094501  mae_deg=3.257 rmse_deg=5.415
lid_length           0.000783   0.001002  

# 100kdataset checkpoints lr 3e-4sincos 40 epoch
x1                   0.001080   0.001357  
y1                   0.000419   0.000555  
z1                   0.000151   0.000200  
box_base_yaw         0.011488   0.014410  mae_deg=0.658 rmse_deg=0.826
lid_angle            0.026619   0.032596  mae_deg=1.525 rmse_deg=1.868
flap_angle           0.047868   0.076229  mae_deg=2.743 rmse_deg=4.368
lid_length           0.000606   0.000801 

# 100kdataset checkpoints lr 1e-4sincos 100 epoch
x1                   0.001824   0.002385  
y1                   0.001734   0.002262  
z1                   0.001394   0.001798  
box_base_yaw         0.013464   0.017626  mae_deg=0.771 rmse_deg=1.010
lid_angle            0.021243   0.027996  mae_deg=1.217 rmse_deg=1.604
flap_angle           0.045448   0.065275  mae_deg=2.604 rmse_deg=3.740
lid_length           0.001617   0.002065  

# 10kdataset keypointNet
device=cuda model=keypoint train=8000 val=2000 params=427989
epoch 001 train_mse=0.892985 val_mse=0.687064
epoch 010 train_mse=0.159920 val_mse=0.129246
epoch 020 train_mse=0.090503 val_mse=0.067647
epoch 030 train_mse=0.060147 val_mse=0.041188
epoch 040 train_mse=0.045837 val_mse=0.029001
epoch 050 train_mse=0.037242 val_mse=0.024427
epoch 060 train_mse=0.030957 val_mse=0.019607
epoch 070 train_mse=0.028194 val_mse=0.018037
epoch 080 train_mse=0.025056 val_mse=0.014907

# 100kdataset keypointNet
label                  mae          rmse         extra
x1                   0.003575   0.004801  
y1                   0.003744   0.004933  
z1                   0.002826   0.003728  
box_base_yaw         0.023466   0.032320  mae_deg=1.345 rmse_deg=1.852
lid_angle            0.038560   0.052183  mae_deg=2.209 rmse_deg=2.990
flap_angle           0.090293   0.132637  mae_deg=5.173 rmse_deg=7.600
lid_length           0.003001   0.003894  
key_x                0.006628   0.009744  
key_y                0.014363   0.019057  
key_z                0.007197   0.009371  
normal_x             0.033186   0.051294  
normal_y             0.059225   0.084348  
normal_z             0.076553   0.111333  
horizontal_x         0.007804   0.011279  
horizontal_y         0.023503   0.032122  
horizontal_z         0.000000   0.000000  
l1                   0.000842   0.001093  

x1                   0.002687   0.003440  
y1                   0.002386   0.003161  
z1                   0.001702   0.002206  
box_base_yaw         0.017697   0.022397  mae_deg=1.014 rmse_deg=1.283
lid_angle            0.021765   0.028823  mae_deg=1.247 rmse_deg=1.651
flap_angle           0.046279   0.071867  mae_deg=2.652 rmse_deg=4.118
lid_length           0.001564   0.002032  
key_x                0.003640   0.005162  
key_y                0.007234   0.009745  
key_z                0.004284   0.005808  
normal_x             0.019722   0.027128  
normal_y             0.026748   0.038765  
normal_z             0.038211   0.055513  
horizontal_x         0.004649   0.006374  
horizontal_y         0.017815   0.022482  
horizontal_z         0.000000   0.000000  
l1                   0.000435   0.000565 


x1                   0.001914   0.002433  
y1                   0.001663   0.002193  
z1                   0.001275   0.001666  
box_base_yaw         0.011466   0.014551  mae_deg=0.657 rmse_deg=0.834
lid_angle            0.016557   0.021677  mae_deg=0.949 rmse_deg=1.242
flap_angle           0.026385   0.041987  mae_deg=1.512 rmse_deg=2.406
lid_length           0.001030   0.001354  
key_x                0.002048   0.002911  
key_y                0.003091   0.004358  
key_z                0.001979   0.002901  
normal_x             0.011840   0.016332  
normal_y             0.016964   0.024163  
normal_z             0.022203   0.033329  
horizontal_x         0.003151   0.004261  
horizontal_y         0.011459   0.014541  
horizontal_z         0.000000   0.000000  
l1                   0.000286   0.000376