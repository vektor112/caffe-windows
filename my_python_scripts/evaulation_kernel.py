import caffe
import sys
import os
import argparse
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

np.set_printoptions(threshold=np.nan)
caffe.set_mode_cpu()

def main(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument(
	    "input",
	    help="Input image or directory."
	)

	parser.add_argument(
	    "output",
	    help="Output directory."
	)

	parser.add_argument(
	    "deploy_txt_path",
	    help="deploy.txt path."
	)

	parser.add_argument(
	    "model_path",
	    help="model path."
	)

	parser.add_argument(
	    "ext",
	    help="Extension of files."
	)

	parser.add_argument(
	    "labels_path",
	    help="Path of labels."
	)

	#Size of image cropping.
	parser.add_argument(
	    "kernel_size",
	    type=int,
	    help="Size of kernel (Size of image cropping)."
	)

		#Size of image cropping.
	parser.add_argument(
	    "name_of_output_layer",
	    default="prob",
	    help="Out put layer name, default 'pron'."
	)

	args = parser.parse_args()

	kernel = args.kernel_size
	padding = (kernel-1)/2

	#load the model and transformer
	net = caffe.Net(args.deploy_txt_path, args.model_path, caffe.TEST)
	transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2, 0, 1))

	netOutputRange = net.blobs[args.name_of_output_layer].data.shape[1]

	args.input_file = os.path.expanduser(args.input)
	if os.path.isdir(args.input_file):
		print("Loading folder: %s" % args.input_file)
		inputs = [(Image.open(im_f), im_f) for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
	else:
		print("Loading file: %s" % args.input_file)
		inputs = [(Image.open(args.input_file), args.input_file)]

	print "File number loaded: %d" % len(inputs)

	#Create directories
	for name in range(0, netOutputRange):
		newpath = args.output + "\\" + str(name)
		if not os.path.exists(newpath):
			os.makedirs(newpath)
			print newpath, "Directory created."

	#loop
	for im in inputs:
		im_w, im_h = im[0].size
		background = Image.new('RGB', (im_w + padding, im_h + padding), (0, 0, 0))
		bg_w, bg_h = background.size
		offset = ((bg_w - im_w) / 2, (bg_h - im_h) / 2)
		background.paste(im[0], offset)
		paddedImage = background

		#Create arrays
		arrays = [np.zeros((im_h, im_w, 3), dtype=np.uint8) for l in range(0, netOutputRange)]
		
		print im[0] ,'Committed for processing'

		for j in range(0, bg_w - padding):
			print 'Row processing:' ,j
			for i in range(0, bg_h - padding):
				
				#column
				#i = 230
				#row
				#j = 50
				#print j,i
				crop_rectangle = (i, j, i + kernel, j + kernel)

				cropped_im = paddedImage.crop(crop_rectangle)

				#if(i == 230):
				#	cropped_im.show()

				predictThisPicturePls = np.asarray(cropped_im)
				#print predictThisPicturePls.shape

				net.blobs['data'].data[...] = transformer.preprocess('data', predictThisPicturePls)

				prediction = net.forward()
				results = prediction[args.name_of_output_layer]

				#print 'predicted classes:', results[0]

				for index, res in enumerate(results[0]):
					arrays[index][j][i] = np.array([res*255, res*255, res*255], dtype=np.uint8)

				

		#Create images
		for index, res_array in enumerate(arrays):
			res_img = Image.fromarray(res_array, 'RGB')
			res_img.save(args.output + "\\" + str(index) + "\\" + im[1].rsplit('\\', 1)[-1])
	

if __name__ == '__main__':
	main(sys.argv)