def iris_data ():
    from sklearn import datasets
    iris = datasets.load_iris()
    return iris

def birth_weight_data ():
    ## DOESNT WORK FORBIDDEN ERROR CODE 403 ON REQUEST
    import requests
    birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
    birth_file = requests.get(birthdata_url)
    birth_data = birth_file.text.split('\'r\n') [5:]
    birth_header = [x for x in birth_data[0].split() if len(x)>=1]
    birth_data = [[float(x) for x in y.split () if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
    return birth_header, birth_data

def housing_data ():
    import requests
    housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV0']
    housing_file = requests.get(housing_url)
    housing_data = [[float(x) for x in y.split() if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]
    return housing_header, housing_data

def mnist_data ():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    return mnist

def spam_ham_text_data ():
    import requests
    import io
    from zipfile import ZipFile
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x)>=1]
    [text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]
    return [text_data_target, text_data_train]

def movie_review_data ():
    import requests
    import io
    import tarfile
    movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    r = requests.get(movie_data_url)
    stream_data = io.BytesIO(r.content)
    tmp = io.BytesIO()
    while True:
        s = stream_data.read(16384)
        if not s:
            break
        tmp.write(s)
    stream_data.close()
    tmp.seek(0)
    tar_file = tarfile.open(fileobj=tmp, mode="r:gz")
    pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
    neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
    pos_data = []
    for line in pos:
        pos_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
    neg_data = []
    for line in neg:
        neg_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
    tar_file.close()
    return pos_data, neg_data

def shakespeare_text_data ():
    import requests
    shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
    response = requests.get(shakespeare_url)
    shakespeare_file = response.content
    shakespeare_text = shakespeare_file.decode('utf-8')
    shakespeare_text = shakespeare_text[7675:]
    return shakespeare_text

def english_german_trans_data ():
    import requests
    import io
    from zipfile import ZipFile
    sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
    r = requests.get(sentence_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('deu.txt')
    eng_ger_data = file.decode()
    eng_ger_data = eng_ger_data.encode('ascii', errors='ignore')
    eng_ger_data = eng_ger_data.decode().split('\n')
    eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x) >= 1]
    [english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
    return [english_sentence, german_sentence], eng_ger_data







