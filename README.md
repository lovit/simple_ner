# Named Entity Recognizer

카운팅과 Logistic Regression을 이용하여 Named Entity를 추출하는 패키지 입니다. Named Entity Recognition (NER)을 기존의 방법처럼 Conditional Random Field (CRF), Recurrent Neural Network (RNN) 등의 sequential labeling 모델을 이용하지 않고, 각각의 단어에 대하여 개별적인 classification을 수행함으로써 해당 단어가 named entity인지 판단합니다. Named entity를 판단하는데 굳이 문장의 전체적인 정보를 이용하지 않아도 된다고 생각하기 때문입니다. 대신에 한 단어에 대하여 한 문장이 아닌 여러 문장에서 context에 대한 정보를 종합적으로 살펴보면 이를 판단할 수 있습니다. 

    지금 [버스] 타고 가고 있어
    집에는 [버스] 타고 갈래? 
    더운데 [택시] 잡자

위의 예제에서 ['버스', '택시']가 교통수단이라는 named entity를 판단하는 근거는 뒤의 ['타고', '잡자', ...] 와 같은 아주 가까운 부분에 있는 몇 몇 단어들입니다. 그리고 다른 교통수단들 주위에는 이와 같은 공통된 단어가 많이 등장할 것입니다. 우리는 처음 사용자가 임의로 (혹은 Word2Vec과 같은 모델을 이용하여) 지정한 seed words를 positive class로 지정하여 이를 구분하는 classifier를 학습함으로써 seed words [버스, 택시, ...]를 구분하는 질 좋은 features를 학습합니다. 그리고 이를 이용하여 다른 named entity를 찾습니다. 마치 seed words와 비슷한 features를 지니는 다른 words로 확장하는 것과 같습니다. 

아직 작업중인 패키지이며, 인터페이스는 정해지지 않았습니다. 


## config.py 구성요소

tutorials에서 config.py 파일이 생략되어 있습니다. 이 파일에는 아래의 파일들을 만들어 입력하여야 합니다. 

1. sentence_fname: str or list of str. 학습에 이용할 토크나이징이 완료되어 있는 sentence 파일. 파일의 스냅샷은 아래와 같다. 

        감사합니다
        네
        아뇨 내꺼 아닌데 요
        ...

1. word2vec_model_fname: str. gensim.Word2Vec을 학습한 pickle 파일
1. feature_manager_fname: str. FeatureManager의 학습된 parameters들을 저장하는 pickle 파일
1. zcorpus_fname: str. FeatureManager를 통하여 zcorpus 형식으로 변형된 결과 파일. 파일의 스냅샷은 아래와 같다. 

        감사합니다
        네
        아뇨  4428
        내꺼  1666 1637 30231
        아닌데 1798 2783 185843 1
        요   4007 2202
        ...

'감사합니다', '네'는 유용한 features가 없기 때문에 변형되지 않았으며, 아뇨:는 4428 번 feature가 수식된다는 의미이다. 

1. z_mm_fname: str. Logistic Regression을 이용하여 NER을 추출할 때에는 sklearn의 CountVectorizer를 이용하여 Sparse matrix를 만든다. 이 파일의 이름.
1. z_mm_xwords_fname: str. z_mm의 row 가 어떤 단어인지 저장한 파일. 스냅샷은 아래와 같다. 

        카라반
        아팠으니까
        내껏두
        ...

1. z_mm_feature_vocab_fname: str. z_mm의 column 가 어떤 feature인지 저장한 파일. 스냅샷은 아래와 같다. 

        ('X[1]', '?')
        ('X[1]', '요')
        ('X[-1]', '나')
        ('X[-1]', '요')
        ('X[-2]', '나')
        ('X[-1]', '응')
        ...

## From text file to Zcorpus

Zcorpus는 단어 \t [encoded_feature, ...] 형식의 corpus입니다. Zcorpus를 만드는 방법은 아래와 같습니다. 

    from config import sentence_fname, zcorpus_fname
    from simple_ner import FeatureCountingNER, FeatureManager, ZCorpus, Sentences

    feature_manager = FeatureManager(templates=None,
                                     feature_begin=-2,
                                     feature_end=2)

    # sentence_fname = [sentence_fname] (list of str is possible)
    sentences = Sentences(sentence_fname)
    feature_manager.scanning_features(sentences,
                                      pruning_n_sents=1000000,
                                      pruning_min_count=5,
                                      min_count=50)
    feature_manager.save(feature_manager_fname)

학습된 FeatureManager는 sentence를 feature list로 변환해 줍니다. 

    for i, sent in enumerate(sentences):
        if i >= 5: break
        print('sent            :', sent)
        print('features        :', feature_manager.words_to_feature(sent))
        print('encoded features:', feature_manager.words_to_encoded_feature(sent), end='\n\n')

결과는 아래와 같습니다. '감사합니다', '네'의 경우에는 사용할만한 features가 없어서 empty list가 출력됩니다. 

    sent            : ['감사합니다']
    features        : [[]]
    encoded features: [[]]

    sent            : ['네']
    features        : [[]]
    encoded features: [[]]

    sent            : ['아뇨', '내꺼', '아닌데', '요']
    features        : [[('X[1]', '내꺼'), ('X[1,2]', ('내꺼', '아닌데'))],
                       [('X[-1]', '아뇨'), ('X[-1,1]', ('아뇨', '아닌데')), ('X[-1,2]', ('아뇨', '아닌데', '요')), ('X[1]', '아닌데'), ('X[1,2]', ('아닌데', '요'))],
                       [('X[-2]', '아뇨'), ('X[-2,-1]', ('아뇨', '내꺼')), ('X[-2,1]', ('아뇨', '내꺼', '요')), ('X[-1]', '내꺼'), ('X[-1,1]', ('내꺼', '요')), ('X[1]', '요')],
                       [('X[-2]', '내꺼'), ('X[-2,-1]', ('내꺼', '아닌데')), ('X[-1]', '아닌데')]
                      ]
    encoded features: [[4428],
                       [1666, 1637, 30231],
                       [1798, 2783, 185843, 1],
                       [4007, 2202]
                      ] 

## NER extraction from Logistic Regression and Zcorpus

    from config import zcorpus_fname, feature_manager_fname
    from config import z_mm_fname, z_mm_xwords_fname, z_mm_feature_vocab_fname
    from simple_ner import get_process_memory, write_list
    from simple_ner import FeatureManager, ZCorpus
    from simple_ner import zcorpus_to_sparsematrix

    zcorpus = ZCorpus(zcorpus_fname)
    feature_manager = FeatureManager()
    feature_manager.load(feature_manager_fname)

Zcorpus를 sparse matrix로 만듭니다. int2word는 row id에 대한 word, feature_vocab은 column id에 대한 feature의 list입니다. 

    x, int2word, feature_vocab = zcorpus_to_sparsematrix(zcorpus,
                                                         feature_manager,
                                                         ner_min_feature_count=20,
                                                         pruning_per_instance=1000000,
                                                         pruning_min_featuer_count=2)

Logistic Regression을 학습하기 위해서는 row가 unit vector로 normalize 되어 있어야 합니다. 그렇지 않을 경우에는 regression model의 coefficient가 scale에 영향을 받습니다. 

    from sklearn.preprocessing import normalize
    x = normalize(x)

ner seeds를 설정한 뒤, 이 단어들이 positive class가 되도록 y를 만듧니다. 

    ner_seeds = '햄벅  빈대떡  팥죽  채소  인스턴트  고로케  홍어  크레페  치폴레  수제비  떡튀순  문어  콜라  삼김  짜왕  꽃게  복숭아  요구르트  쌈도  야식  한우  칼국수  해물찜  쌀밥  육개장  골뱅이  후식  아구찜  컵밥  음료수  고기  낙지  유자차  조개구이'.split('  ')

    word2int = {w:i for i,w in enumerate(int2word)}
    ner_seeds_idx = {word2int[w] for w in ner_seeds if w in word2int}
    y = [1 if i in ner_seeds_idx else -1 for i in range(x.shape[0])]

Lotistic Regression을 이용하여 seed words를 positive class로 구분하는 판별기를 만듧니다. 

    from sklearn.linear_model import LogisticRegression

    logistic = LogisticRegression(penalty='l2', C=1.0, class_weight={1:0.98, -1:0.02})
    logistic.fit(x, y)
    prob = logistic.predict_proba(x)

    extracted_ners = sorted(enumerate(prob[:,1]), key=lambda x:x[1], reverse=True)
    extracted_ners = [(int2word[i], p) for i, p in extracted_ners]
