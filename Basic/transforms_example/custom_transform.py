class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

                

class shades_of_gray(object):
	 
	 #     Parameters
	 #    ----------
	 #   img: 2D numpy array
	 #         The original image with format of (h, w, c)
	 #     power: int
	 #         The degree of norm, 6 is used in reference paper
	 # 
		 
		
	def __call__(self, img):
			"""
			:param img: PIL): Image 

			:return: Normalized image
			"""
			img = numpy.asarray(img)
			img_dtype = img.dtype

			power = 6
			extra = 6

			img = img.astype('float32')
			img_power = numpy.power(img, power)
			rgb_vec = numpy.power(numpy.mean(img_power, (0, 1)), 1 / power)
			rgb_norm = numpy.power(numpy.sum(numpy.power(rgb_vec, extra)), 1 / extra)
			rgb_vec = rgb_vec / rgb_norm
			rgb_vec = 1 / (rgb_vec * numpy.sqrt(3))
			img = numpy.multiply(img, rgb_vec)
			img = img.astype(img_dtype)

			return Image.fromarray(img)

	def __repr__(self):
			return self.__class__.__name__+'()'