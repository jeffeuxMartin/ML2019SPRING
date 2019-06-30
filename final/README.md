### Machine Learning Final Project
### Online News Intent Retrieval

+ #### How to run the code?
   - Install dependencies
   ```bash
      pip install -r requirements.txt
   ```
   (mainly *NumPy*, *Pandas*, *SciPy*, *scikit-learn*, *jieba*, *Gensim*)

   - Execute the whole testing process
   ```bash
       bash test.sh path/to/url2content.json path/to/TD.csv \
       path/to/QS_1.csv path/to/predict.csv
   ```
     Running through 
        downloading models -> TF-IDF BOW -> KeyWord Embedding -> Ensembling

   - Training the Word2Vec model
   ```bash
         bash train.sh path/to/url2content.json path/to/TD.csv \
       path/to/QS_1.csv
   ```
    This is necessary if training from scratch is needed. Other parts do not contain training processes.

   **If to reproduce the result, ONLY TESTING IS NECESSARY!**
