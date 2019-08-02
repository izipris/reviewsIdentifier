from utilities.LanguageUtils import LanguageUtils
from utilities.DataUtils import DataUtils

X,y = DataUtils.preprocess_data('C:\\Users\\idzipris\\Downloads\\testai.txt')
text = "I ain't doing anything"
print(LanguageUtils.remove_contractions(text))