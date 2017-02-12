# Menghitung Kemiripan Dokumen dengan cosinus similarity

from sklearn.feature_extraction.text import CountVectorizer 
# from sklearn.feature_extraction.text import TfidfVectorizer #aktifkan jika ingin mengitung otomatis dari library
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import math
import numpy as np


REGEX = re.compile(r"\s")
def tokenize(text):
    return [tok.strip().lower() for tok in REGEX.split(text)]

def stopwords(text):
	reg = re.compile(r"\n")
	return reg.split(text)



file = open("source1.txt","r");
raw1 = file.read()

file = open("source2.txt","r");
raw2 = file.read()

file = open("source3.txt","r");
raw3 = file.read()

file = open("source4.txt","r");
raw4 = file.read()

file = open("source5.txt","r");
raw5 = file.read()

# print "BEFORE======================================================================================="
# print "DOKUMEN 1===================================================================================="
# print raw1
# print "DOKUMEN 2===================================================================================="
# print raw2
# print "DOKUMEN 3===================================================================================="
# print raw3
# print "DOKUMEN 4===================================================================================="
# print raw4
# print "DOKUMEN 5===================================================================================="
# print raw5


# dibuat huruf kecil semua
raw1=raw1.lower()
raw2=raw2.lower()
raw3=raw3.lower()
raw4=raw4.lower()
raw5=raw5.lower()


# menghilangkan tanda baca
tandabaca = [".",",","-","%","(",")"]
for td in tandabaca:
	raw1=raw1.replace(td,"")
	raw2=raw2.replace(td,"")
	raw3=raw3.replace(td,"")
	raw4=raw4.replace(td,"")
	raw5=raw5.replace(td,"")


# menghilangkan stop words dari database bahasa indonesia
file = open("stopwords.txt","r");
st = file.read()
stopwords = stopwords(st)

for word in stopwords:
	raw1=raw1.replace(" "+word+" "," ")
	raw2=raw2.replace(" "+word+" "," ")
	raw3=raw3.replace(" "+word+" "," ")
	raw4=raw4.replace(" "+word+" "," ")
	raw5=raw5.replace(" "+word+" "," ")
# print "AFTER======================================================================================"
# print "DOKUMEN 1===================================================================================="
# print raw1
# print "DOKUMEN 2===================================================================================="
# print raw2
# print "DOKUMEN 3===================================================================================="
# print raw3
# print "DOKUMEN 4===================================================================================="
# print raw4
# print "DOKUMEN 5===================================================================================="
# print raw5

# stemming Bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()


hasilstem1 = stemmer.stem(raw1)
hasilstem2 = stemmer.stem(raw2)
hasilstem3 = stemmer.stem(raw3)
hasilstem4 = stemmer.stem(raw4)
hasilstem5 = stemmer.stem(raw5)

# print "AFTER======================================================================================"
# print "DOKUMEN 1===================================================================================="
# print hasilstem1
# print "DOKUMEN 2===================================================================================="
# print hasilstem2
# print "DOKUMEN 3===================================================================================="
# print hasilstem3
# print "DOKUMEN 4===================================================================================="
# print hasilstem4
# print "DOKUMEN 5===================================================================================="
# print hasilstem5




train_set = [hasilstem1,hasilstem2,hasilstem3,hasilstem4,hasilstem5]

count_vectorizer = CountVectorizer(tokenizer=tokenize)
data = count_vectorizer.fit_transform(train_set).toarray()
vocab = count_vectorizer.get_feature_names()

print "Jumlah Term FREQUENCY============================="
print data
print "VECTOR FITUR============================="
print vocab
print "JUMLAH VECTOR FITUR============================="
print len(vocab)

# export excel terms Frequency
xls="Term-Frequency,"
for vc in vocab:
	xls+=vc+","

xls += "\n"

i = 0
for dt in data:
	xls += "DOKUMEN "+str(i)+","
	for dta in dt:
		xls += str(dta)+","
	xls += "\n"	
	i += 1




xls += "\n\n\n\n\n"
# export excel Document Frequency
xls += "Document-Frequency,"
for vc in vocab:
	xls+=vc+","

xls += "\n"

dfs = []
for dt in data:
	index2 = 0
	xls += ","
	for dta in dt:
		count = 0
		if data[0][index2]>0:count += 1
		if data[1][index2]>0:count += 1
		if data[2][index2]>0:count += 1
		if data[3][index2]>0:count += 1
		if data[4][index2]>0:count += 1
		dfs.append(count)
		xls += str(count)+","
		index2 += 1
	xls += "\n"
	break


xls += "\n\n\n\n\n"
# export TF-IDF
xls += "TF-IDF,"
for vc in vocab:
	xls+=vc+","

xls += "\n"

i=0
TFIDF=[]
for dt in data:
	xls += "DOKUMEN "+str(i)+","
	j=0
	data_tfidf=[]
	for dta in dt:
		# calculate tfidf with formula tfidf = tf*log(Ndocument/df)
		tfidf = dta*math.log(5/dfs[j])
		data_tfidf.append(tfidf)
		xls += str(tfidf)+","
		j += 1
	
	TFIDF.append(data_tfidf)

	xls += "\n"
	i += 1
	


# calculate COSINE-SIMILARITY
# sumber : https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/
# penjelasan vector : http://www.mathsisfun.com/algebra/vectors-dot-product.html
# formula cos x = document1.document2 / |document1|x|document2|
# dot product = x1y1 + x2y2 + xiyi
# magnitude using pytaghoras
# magnitude = sqrt(x1^2+x2^2+xi^2) x sqrt(y1^2+y2^2+yi^2)


xls += "\n\n\n\n\n\nCOSINUS SIMILARITY,,Rumus yang digunakan = document1.document2 / |document1|x|document2|\n\n\n"

xls+= ","
for i in range(5):
	xls += "Dokumen"+str(i+1)+","
xls += "\n"

for i in range(5):
	xls += "Dokumen"+str(i+1)+","
	for j in range(5):
		dt1 = TFIDF[i]
		dt2 = TFIDF[j]
		# calculate dot product
		dotproduct = np.dot(dt1,dt2)
		# calculate 2 data magnitude
		magnitude1 = np.sqrt(np.dot(dt1,dt1))
		magnitude2 = np.sqrt(np.dot(dt2,dt2))

		cos_similarity = dotproduct/(magnitude1*magnitude2)
		xls += str(cos_similarity)+","
	xls += "\n"



file = open("tf-idf.csv","w")
file.write(xls)
print "exported"



# USING SKLEARN LIBRARY===========================

# tfidf = TfidfVectorizer().fit_transform(train_set)
# pairwise_similarity = tfidf * tfidf.T

# print "Jumlah Term FREQUENCY-Inverse Document Frequency============================="
# print tfidf
# print type(tfidf)

# print "Jumlah COSINE-SIMILARITY============================="
# print pairwise_similarity




