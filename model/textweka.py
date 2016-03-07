from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.core.converters import Loader
import weka.core.jvm as jvm

jvm.start(system_cp=True, packages=True)

print 'saul'
data_dir='documents\Arff\svdtransfer'
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file("data/breast-cancer.arff")

data.class_is_last()
print type(data),data