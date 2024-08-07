# Correlation Encoder-Decoder Model for Text Generation
Based on the work of Hareesh Bahuleyan et al. we modified the structure of the model so that the model can take into account the correlation in the encode-decode process.
## Datasets
The proposed model and baselines have been evaluated on two experiments:      
Neural Question Generation with the SQuAD dataset https://rajpurkar.github.io/SQuAD-explorer/   
Daily Dialog dataset http://yanran.li/dailydialog.html
## Requirements
tensorflow-gpu==1.3.0    
Keras==2.0.8    
numpy==1.12.1    
pandas==0.22.0   
gensim==3.1.2   
nltk==3.4.5   
tqdm==4.19.1   
## Acknowledgments
1. Hareesh Bahuleyan, Lili Mou, Hao Zhou, and Olga Vechtomova, “Stochastic wasserstein autoencoder for probabilistic sentence generation,” in NAACL, 2019, pp. 4068–4076.
https://github.com/HareeshBahuleyan/probabilistic_nlg 
2. Hareesh Bahuleyan, Lili Mou, Olga Vechtomova, and Pascal Poupart, “Variational attention for sequence-to- sequence models,” in COLING, 2018, pp. 1672–1682.
https://github.com/HareeshBahuleyan/tf-var-attention
3. 深度学习的互信息：无监督提取特征 https://kexue.fm/archives/6024, https://github.com/bojone/infomax



# Reference
If you find our source useful, please consider citing our work.

Zhang X, Li Y, Peng X, et al. Correlation encoder-decoder model for text generation[C]//2022 International Joint Conference on Neural Networks (IJCNN). IEEE, 2022: 1-7.
