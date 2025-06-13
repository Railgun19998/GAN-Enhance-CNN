## ource_CNN.py  
input data.S3.csv <- real_data.csv  
train(epoch = 1000, batch_size = 8)  
-> cnn_Ti.h5  
  

## GAN.py  
input data.S3.csv  
train(latent_dim = 100, epoch = 8000, batch_size = 800)  
-> generator_model.h5  
-> filtered_generated_features.csv -> samp_data.csv  
  
all_data.csv = samp_real.csv = real_data.csv + samp_data.csv  
  
## CNN_Predict.py  
input all_data.csv  
train(epoch = 1000, batch_size = 8)  
-> cnn_sr.h5  
predict(data.S3.csv)  
  
## Post_Process.py  
print effect factor