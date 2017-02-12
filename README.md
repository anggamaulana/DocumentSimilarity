# DocumentSimilarity
Calculating Document Similarity using tfidf and cosinus similarity

Menghitung Kemiripan dokumen dengan tf-idf dan cosinus similarity

Dokumen yang akan dibandingkan ada di source1.txt,source2.txt,source3.txt,source4.txt,source5.txt

run di terminal : python feature_extraction.py

Akan menghasilkan hasil kalkulasi di file tf-idf.csv buka menggunakan excel


library yang digunakan di python 
-sklearn
-numpy
-Sastrawi.Stemmer

install menggunakan pip


pip install numpy

pip install -U scikit-learn

pip install Sastrawi

Formula yg digunakan : 

tf = jumlah terms di dokumen tanpa normalisasi  (biasanya dilakukan normalisasi di library)
df = jumlah terms di dokumen mana saja yg muncul

formula tfidf yg digunakan sesuai di kelas
tf-idf[i,j] = tf[i,j]*log(Ndocument/df[i])

formula cosinus similarity mengmbil konsep perhitungan angle di vector cos(x)=a.b/||a||||b||

