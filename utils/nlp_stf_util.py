
from stanfordcorenlp import StanfordCoreNLP
import nltk
nlp = StanfordCoreNLP(r'D:\\code\2020\event_detection_without_triggers-master\data\standford\\stanford-corenlp-full-2018-10-05',lang='en')
print('================')
nltk.download('punkt')