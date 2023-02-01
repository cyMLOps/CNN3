import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import pickle, json

# plot diagnostic learning curves
# def summarize_diagnostics(history):
# 	# plot loss
# 	pyplot.subplot(211)
# 	pyplot.title('Cross Entropy Loss')
# 	pyplot.plot(history['loss'], color='blue', label='train')
# 	pyplot.plot(history['val_loss'], color='orange', label='test')
# 	# plot accuracy
# 	pyplot.subplot(212)
# 	pyplot.title('Classification Accuracy')
# 	pyplot.plot(history['accuracy'], color='blue', label='train')
# 	pyplot.plot(history['val_accuracy'], color='orange', label='test')
# 	# save plot to file
# 	filename = sys.argv[0].split('/')[-1]
# 	pyplot.savefig(filename + '_plot.png')
# 	pyplot.close()

# save scores 
def save_score(acc):
	with open('score.json', 'w') as f:
		json.dump({'acc': acc}, f)

# save plots
def save_plot(history):
	loss = history['loss']
	val_loss = history['val_loss']
	with open('plot.json', 'w') as f:
		proc_dict = {'proc': [{
			'loss': loss,
			'val_loss': val_loss
		}for loss, val_loss in zip(loss, val_loss)]}
		json.dump(proc_dict, f)
		


# evaluate the model
def run_test_harness():
	# define model & history
	model = load_model('model.h5')
	# history = pickle.load(open('history.pkl', 'rb'))
	# create data generator
	datagen = ImageDataGenerator(featurewise_center=True)
	# specify imagenet mean values for centering
	datagen.mean = [123.68, 116.779, 103.939]
	# prepare iterato
	test_it = datagen.flow_from_directory('dataset_dogs_and_cats/test/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('-----------evaluated--------')
	print('> %.3f' % (acc * 100.0))
	# learning curves
	history = pickle.load(open('history.pkl', 'rb'))
	# save score
	save_score(acc)
	# save plot
	save_plot(history)
	# summarize_diagnostics(history)



# entry point, run the test harness
run_test_harness()
print('plot drawn')
# test






    