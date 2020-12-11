## 3D-CNN Model for EEG-based Emotion Recognition 
----
This is a final project for my 2019 Winter course "Pattern Recognition" based on DEAP dataset.
We preprocess original signal data via **CWT**(Continuous Wavelet Transform) and bulit a **3D-CNN** architecture as classifier, the accuracy of 'valence' label reaches 84.34%.

#### Data Preprocessing
1.  CWT analysis  
Baseline is removed, and original siginal data tramsformed into wavelet coefficients through CWT, then further into wavelet energy (scalograms).  In this step, we transform data shape from 32(channel)\*8064(sample points) into 32*64(scale)\*7680(sample points).
2.  Cut frames  
Next, we set 1s as frame length, thus 60 frames can be got within a 60s video. The shape for each frame is 32(channel)*64(scale).
3.  Select scales   
Then we calculated mean EER for all 64 scales in 32 channels. And 8th~39th scales are selected to reduce caculation.

4. 3D chunk     
We select several continous frames and stack them togther as a 3D chunk. The later experiments proved that 3 is the best.

#### Classifier: 3D-CNN
The network architecture is as follows.     
![6 Conv layers & 3 MaxPooling Layers & 1 Fc layer & SoftMax]( https://github.com/Daisybiubiubiu/EEG-Emotion-Recognition/tree/master/CWT/Figure/3D-CNN_architecture.png "3D-CNN Architeture")

#### About Code & Files
We use matlab_preprocessed_data, which is excluded from this repo.  
Run 'cwt_process.m' to get 'File_60frame_exscale'.    
Run 'scale_select.m' to get 'sumEER.mat' & 'scale_select.png'.
Run '3d_cnn.ipynb' based on files in 'File_60frame_exscale'.    
Besides, part of data are used to train 3D-CNN, some parameters of it are stored in  'model_statedict.pth' & 'optimizer_statedict.pth'.
Therefore, 'demo.py' can directly do the predict task based on the 'x_test.pth' & 'y_test.pth'.
##### Main Reference
+ Li, X., et al., Emotion Recognition from Multi-Channel EEG Data through Convolutional Recurrent Neural Network, in 2016 Ieee International Conference on Bioinformatics and Biomedicine, T. Tian, et al., Editors. 2016. p. 352-359.
+ Salama, E.S., et al., EEG-Based Emotion Recognition using 3D Convolutional Neural Networks. International Journal of Advanced Computer Science and Applications, 2018. 9(8): p. 329-337.
+ DEAP Dataset: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
