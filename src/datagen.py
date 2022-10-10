from keras.preprocessing.image import ImageDataGenerator

class DataGenerator:

  def __init__(self, seed, X_fit, y_fit):
    self.seed = seed
    self.augmentations = {
      "fill_mode" : "wrap",
      "rotation_range" : 360,
      "width_shift_range" : 0.2, 
      "height_shift_range" : 0.2,
      "horizontal_flip" : True, 
      "vertical_flip" : True   
    }
    self.image_generator = ImageDataGenerator(**self.augmentations)
    self.mask_generator = ImageDataGenerator(**self.augmentations)
    self.image_generator.fit(X_fit, seed = seed)
    self.mask_generator.fit(y_fit, seed = seed)
  
  
  def get_generator(self, X, y, batch_size):
    return zip(
    self.image_generator.flow(X, batch_size=batch_size, shuffle=True, seed=self.seed), 
    self.mask_generator.flow(y, batch_size=batch_size, shuffle=True, seed=self.seed)
  )